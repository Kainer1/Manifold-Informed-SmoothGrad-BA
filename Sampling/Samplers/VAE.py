"""Posterior-mode VAE sampling with range-scaled Gaussian latent noise."""

import os

import torch
from diffusers import AutoencoderKL

from Sampling.Samplers.SmoothGrad import SmoothGradSampler

VAE_MODEL_IDS = {
    "VAE_SD2": "stabilityai/sd-vae-ft-mse",
    "VAE_REPAE_ImageNet": "REPA-E/e2e-invae-hf",
}


class VAESampler(SmoothGradSampler):
    def __init__(self, *, sampler_config, **kwargs):
        super().__init__(sampler_config=sampler_config, **kwargs)
        self.model_id = VAE_MODEL_IDS[self.sampler_name]
        self.storage_config["model_id"] = self.model_id
        self.latents = None
        self.image_mean = None
        self.image_std = None

    def setup(self, inputs: torch.Tensor, labels: torch.Tensor = None):
        self.vae_model = AutoencoderKL.from_pretrained(
            self.model_id, token=os.environ.get("HF_TOKEN")
        ).to(inputs.device)
        self.vae_model.eval()
        self.image_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=inputs.device
        ).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225], device=inputs.device).view(
            1, 3, 1, 1
        )
        pixels = torch.clamp(inputs * self.image_std + self.image_mean, 0, 1)
        with torch.no_grad():
            self.latents = self.vae_model.encode(pixels * 2.0 - 1.0).latent_dist.mode()

    def compute_sample(
        self, inputs: torch.Tensor, labels: torch.Tensor = None
    ) -> torch.Tensor:
        if self.latents is None:
            raise ValueError("Call setup() before generating VAE samples.")
        noisy_latents = super().compute_sample(self.latents)
        with torch.no_grad():
            decoded = self.vae_model.decode(noisy_latents).sample
        pixels = torch.clamp((decoded + 1.0) / 2.0, 0, 1)
        return ((pixels - self.image_mean) / self.image_std).to(inputs.dtype)

    def cleanup(self):
        self.image_mean = None
        self.image_std = None
        super().cleanup()
