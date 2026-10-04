"""Original DiT and SD2 sampling with optional class conditioning and no CFG."""

import math

import torch
from torchvision.models import VGG16_Weights

from Sampling.BaseSampler import BaseSampler
from Sampling.LoadModels import load_dit_imagenet, load_sd2_img2img

IMAGENET_CATEGORIES = VGG16_Weights.IMAGENET1K_V1.meta["categories"]


def _num_inference_steps_for_exact_actual_steps(
    actual_steps: int, strength: float, max_inference_steps: int = 1000
) -> int:
    """Return a schedule length whose img2img suffix has exactly actual_steps."""
    actual_steps = int(actual_steps)
    strength = float(strength)

    if not 0.0 < strength <= 1.0:
        raise ValueError("strength must be in the interval (0, 1].")
    if actual_steps < 1:
        raise ValueError("actual_steps must be at least 1.")

    num_inference_steps = max(1, math.ceil(actual_steps / strength))
    while int(num_inference_steps * strength) < actual_steps:
        num_inference_steps += 1

    if num_inference_steps > max_inference_steps:
        raise ValueError(
            f"num_inference_steps ({num_inference_steps}) exceeds "
            f"{max_inference_steps}. Increase strength or reduce actual_steps."
        )
    if int(num_inference_steps * strength) != actual_steps:
        raise ValueError(
            "Could not represent actual_steps exactly with the requested strength."
        )

    return num_inference_steps


def _to_rgb(inputs):
    mean = torch.tensor([0.485, 0.456, 0.406], device=inputs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=inputs.device).view(1, 3, 1, 1)
    return torch.clamp(inputs * std + mean, 0, 1)


def _from_rgb(images):
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (torch.clamp(images, 0, 1) - mean) / std


class LatentDiffusionSampler(BaseSampler):
    def __init__(self, *, sampler_config, **kwargs):
        for key in ("strength", "actual_steps", "class_conditioning"):
            if key not in sampler_config:
                raise ValueError(f"Missing required setting: config.{key}")
        self.strength = sampler_config["strength"]
        if type(self.strength) not in (int, float):
            raise ValueError("config.strength must be a number in (0, 1].")
        self.actual_steps = sampler_config["actual_steps"]
        if type(self.actual_steps) is not int:
            raise ValueError("config.actual_steps must be a positive integer.")
        self.class_conditioning = sampler_config["class_conditioning"]
        if type(self.class_conditioning) is not bool:
            raise ValueError("config.class_conditioning must be true or false.")
        self.strength = float(self.strength)
        self.num_inference_steps = _num_inference_steps_for_exact_actual_steps(
            self.actual_steps, self.strength
        )
        super().__init__(sampler_config=sampler_config, **kwargs)
        self.storage_config.update(
            model_id=self.MODEL_ID,
            scheduler_name="DDIMScheduler",
            vae_posterior_mode="mode",
        )


class DiTSampler(LatentDiffusionSampler):
    MODEL_ID = "facebook/dit-xl-2-256"
    LATENT_SCALE = 0.18215

    def setup(self, inputs: torch.Tensor, labels: torch.Tensor = None):
        if labels is None:
            raise ValueError("DiT requires ImageNet labels.")
        self.model_pipeline = load_dit_imagenet(inputs.device)
        self.vae, self.transformer, self.scheduler = self.model_pipeline
        x_vae = _to_rgb(inputs) * 2.0 - 1.0
        with torch.no_grad():
            self.latents = self.vae.encode(x_vae).latent_dist.mode() * self.LATENT_SCALE
        self.scheduler.set_timesteps(self.num_inference_steps)
        self.init_timestep = max(1, int(self.num_inference_steps * self.strength))
        self.t_start = self.scheduler.timesteps[
            self.num_inference_steps - self.init_timestep
        ]
        self.class_labels = (
            labels.to(inputs.device)
            if self.class_conditioning
            else torch.ones_like(labels).to(inputs.device) * 1000
        )

    def compute_sample(
        self, inputs: torch.Tensor, labels: torch.Tensor = None
    ) -> torch.Tensor:
        with torch.no_grad():
            noise = torch.randn_like(self.latents)
            cur_latents = self.scheduler.add_noise(self.latents, noise, self.t_start)
            timesteps = self.scheduler.timesteps[
                self.num_inference_steps - self.init_timestep :
            ]
            for t in timesteps:
                latent_input = self.scheduler.scale_model_input(cur_latents, t)
                timestep_input = torch.full(
                    (latent_input.shape[0],), t, device=inputs.device, dtype=torch.long
                )
                output = self.transformer(
                    latent_input, timestep_input, class_labels=self.class_labels
                ).sample
                channels = cur_latents.shape[1]
                if output.shape[1] == channels * 2:
                    output, _ = torch.split(output, channels, dim=1)
                cur_latents = self.scheduler.step(output, t, cur_latents).prev_sample
            decoded = self.vae.decode(cur_latents / self.LATENT_SCALE).sample
            return _from_rgb((decoded + 1.0) / 2.0).to(inputs.dtype)

    def cleanup(self):
        self.vae = self.transformer = self.scheduler = None
        self.latents = self.class_labels = None
        super().cleanup()


class StableDiffusion2Sampler(LatentDiffusionSampler):
    MODEL_ID = "sd2-community/stable-diffusion-2-base"
    PROMPT_TEMPLATE = "a photo of {imagenet_class}"

    def setup(self, inputs: torch.Tensor, labels: torch.Tensor = None):
        if labels is None:
            raise ValueError("SD2 requires ImageNet labels.")
        if inputs.shape[-2:] != (256, 256):
            raise ValueError("SD2 expects 256x256 inputs after geometry processing.")
        self.model_pipeline = load_sd2_img2img(inputs.device)
        self.prompts = (
            [
                self.PROMPT_TEMPLATE.format(
                    imagenet_class=IMAGENET_CATEGORIES[label.item()]
                )
                for label in labels
            ]
            if self.class_conditioning
            else [""] * len(labels)
        )
        x_vae = _to_rgb(inputs) * 2.0 - 1.0
        vae = self.model_pipeline.vae
        x_vae = x_vae.to(device=inputs.device, dtype=next(vae.parameters()).dtype)
        with torch.no_grad():
            self.init_latents = (
                vae.encode(x_vae).latent_dist.mode() * vae.config.scaling_factor
            )

    def compute_sample(
        self, inputs: torch.Tensor, labels: torch.Tensor = None
    ) -> torch.Tensor:
        with torch.no_grad():
            images = self.model_pipeline(
                prompt=self.prompts,
                image=self.init_latents,
                strength=self.strength,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=1.0,
                eta=0.0,
                output_type="pt",
            ).images
            return _from_rgb(images).to(inputs.dtype)

    def cleanup(self):
        self.init_latents = self.prompts = None
        super().cleanup()
