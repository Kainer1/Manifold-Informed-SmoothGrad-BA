"""Original two-stage DeepFloyd and class-conditioned ADM sampling."""

import torch

from Sampling.BaseSampler import BaseSampler
from Sampling.LoadModels import load_deepfloyd_if, load_openai_adm
from Sampling.Samplers.LatentDiffusion import (
    IMAGENET_CATEGORIES,
    _from_rgb,
    _num_inference_steps_for_exact_actual_steps,
    _to_rgb,
)


class DeepFloydSampler(BaseSampler):
    STAGE1_DOWNSAMPLING = "bicubic_antialias_clamp"

    def __init__(self, *, sampler_config, **kwargs):
        for key in (
            "strength",
            "actual_steps",
            "strength_2",
            "actual_steps_2",
            "class_conditioning",
        ):
            if key not in sampler_config:
                raise ValueError(f"Missing required setting: config.{key}")
        for key in ("strength", "strength_2"):
            value = sampler_config[key]
            if type(value) not in (int, float) or not 0.0 < value <= 1.0:
                raise ValueError(f"config.{key} must be a number in (0, 1].")
        for key in ("actual_steps", "actual_steps_2"):
            value = sampler_config[key]
            if type(value) is not int or value < 1:
                raise ValueError(f"config.{key} must be a positive integer.")
        self.class_conditioning = sampler_config["class_conditioning"]
        if type(self.class_conditioning) is not bool:
            raise ValueError("config.class_conditioning must be true or false.")
        self.strength = float(sampler_config["strength"])
        self.strength_2 = float(sampler_config["strength_2"])
        self.num_inference_steps_1 = _num_inference_steps_for_exact_actual_steps(
            sampler_config["actual_steps"], self.strength
        )
        self.num_inference_steps_2 = _num_inference_steps_for_exact_actual_steps(
            sampler_config["actual_steps_2"], self.strength_2
        )
        super().__init__(sampler_config=sampler_config, **kwargs)
        self.storage_config["stage1_downsampling"] = self.STAGE1_DOWNSAMPLING

    def setup(self, inputs: torch.Tensor, labels: torch.Tensor = None):
        if labels is None:
            raise ValueError("DeepFloyd requires ImageNet labels.")
        self.model_pipeline = load_deepfloyd_if(inputs.device)
        self.pipe_stage1, self.pipe_stage2 = self.model_pipeline
        self.prompts = (
            [f"a photo of {IMAGENET_CATEGORIES[label.item()]}" for label in labels]
            if self.class_conditioning
            else [""] * len(labels)
        )
        self.x_minusoneone = _to_rgb(inputs) * 2.0 - 1.0
        self.x_64_minusoneone = torch.nn.functional.interpolate(
            self.x_minusoneone,
            size=(64, 64),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(-1.0, 1.0)
        with torch.no_grad():
            self.prompt_embeds, self.negative_prompt_embeds = (
                self.pipe_stage1.encode_prompt(
                    prompt=self.prompts, device=inputs.device
                )
            )

    def compute_sample(
        self, inputs: torch.Tensor, labels: torch.Tensor = None
    ) -> torch.Tensor:
        with torch.no_grad():
            image_64 = self.pipe_stage1(
                prompt_embeds=self.prompt_embeds,
                negative_prompt_embeds=self.negative_prompt_embeds,
                image=self.x_64_minusoneone,
                strength=self.strength,
                num_inference_steps=self.num_inference_steps_1,
                guidance_scale=1.0,
                output_type="pt",
            ).images
            if image_64.shape[0] == 0:
                image_64 = self.x_64_minusoneone
            image_256 = self.pipe_stage2(
                prompt_embeds=self.prompt_embeds,
                negative_prompt_embeds=self.negative_prompt_embeds,
                image=image_64,
                original_image=self.x_minusoneone,
                num_inference_steps=self.num_inference_steps_2,
                guidance_scale=1.0,
                output_type="pt",
                strength=self.strength_2,
            ).images
            return _from_rgb((image_256 + 1.0) / 2.0).to(inputs.dtype)

    def cleanup(self):
        self.pipe_stage1 = self.pipe_stage2 = None
        self.prompt_embeds = self.negative_prompt_embeds = None
        self.x_minusoneone = self.x_64_minusoneone = self.prompts = None
        super().cleanup()


class ADMSampler(BaseSampler):
    def __init__(self, *, sampler_config, **kwargs):
        if "strength" not in sampler_config:
            raise ValueError("Missing required setting: config.strength")
        strength = sampler_config["strength"]
        if type(strength) not in (int, float) or not 0.0 < strength <= 1.0:
            raise ValueError("config.strength must be a number in (0, 1].")
        self.strength = float(strength)
        super().__init__(sampler_config=sampler_config, **kwargs)

    def setup(self, inputs: torch.Tensor, labels: torch.Tensor = None):
        if labels is None:
            raise ValueError("ADM requires ImageNet labels.")
        self.model_pipeline = load_openai_adm(inputs.device)
        self.model, self.diffusion = self.model_pipeline
        self.x_adm = _to_rgb(inputs) * 2.0 - 1.0
        self.num_denoising_steps = max(
            1, int(self.diffusion.num_timesteps * self.strength)
        )
        self.t_start = self.num_denoising_steps - 1
        self.model_kwargs = {"y": labels.to(inputs.device)}

    def compute_sample(
        self, inputs: torch.Tensor, labels: torch.Tensor = None
    ) -> torch.Tensor:
        with torch.no_grad():
            t_tensor = torch.tensor(
                [self.t_start] * inputs.shape[0], device=inputs.device
            )
            noise = torch.randn_like(self.x_adm)
            cur_x = self.diffusion.q_sample(self.x_adm, t_tensor, noise=noise)
            for t_idx in range(self.t_start, -1, -1):
                t_step = torch.tensor([t_idx] * inputs.shape[0], device=inputs.device)
                out = self.diffusion.p_sample(
                    self.model,
                    cur_x,
                    t_step,
                    clip_denoised=True,
                    model_kwargs=self.model_kwargs,
                )
                cur_x = out["sample"]
            return _from_rgb((cur_x + 1.0) / 2.0).to(inputs.dtype)

    def cleanup(self):
        self.model = self.diffusion = self.x_adm = self.model_kwargs = None
        super().cleanup()
