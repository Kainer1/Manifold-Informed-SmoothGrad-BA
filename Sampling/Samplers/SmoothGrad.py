"""Original normal SmoothGrad noise with one range factor per input image."""

import math

import torch

from Sampling.BaseSampler import BaseSampler


class SmoothGradSampler(BaseSampler):
    def __init__(self, *, sampler_config, **kwargs):
        if "p" not in sampler_config:
            raise ValueError("Missing required setting: config.p")
        self.p = sampler_config["p"]
        if type(self.p) not in (int, float) or not math.isfinite(self.p) or self.p < 0:
            raise ValueError("config.p must be a finite, nonnegative number.")
        super().__init__(sampler_config=sampler_config, **kwargs)

    def compute_sample(
        self, inputs: torch.Tensor, labels: torch.Tensor = None
    ) -> torch.Tensor:
        xmax = inputs.amax(dim=(1, 2, 3), keepdim=True)
        xmin = inputs.amin(dim=(1, 2, 3), keepdim=True)
        std = self.p * (xmax - xmin)
        noise = torch.randn_like(inputs) * std
        return inputs + noise
