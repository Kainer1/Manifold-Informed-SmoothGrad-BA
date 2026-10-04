"""Original Vanilla Gradient sampler: return the unmodified model input."""

import torch

from Sampling.BaseSampler import BaseSampler


class VanillaGradientSampler(BaseSampler):
    def compute_sample(
        self, inputs: torch.Tensor, labels: torch.Tensor = None
    ) -> torch.Tensor:
        return inputs
