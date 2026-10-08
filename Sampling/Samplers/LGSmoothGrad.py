"""Signed-mean low-gradient mask and centered-circle sanity checks."""

import math

import torch

from Helper.model import load_vgg16
from Sampling.AttributionComputation import input_gradient
from Sampling.Samplers.SmoothGrad import SmoothGradSampler

P_TILDE = 0.09
N_TILDE = 50


class LGSmoothGradSampler(SmoothGradSampler):
    def __init__(self, *, sampler_config, **kwargs):
        super().__init__(sampler_config=sampler_config, **kwargs)
        if "q" not in sampler_config:
            raise ValueError("Missing required setting: config.q")
        self.q = sampler_config["q"]
        if type(self.q) not in (int, float) or not 0 <= self.q <= 1:
            raise ValueError("config.q must be a number between 0 and 1.")
        self.circle_mask = None
        if "CircleMask" in sampler_config:
            self.circle_mask = sampler_config["CircleMask"]
            if self.circle_mask not in ("in", "out"):
                raise ValueError('config.CircleMask must be "in" or "out".')
        self.storage_config.update(p_tilde=P_TILDE, n_tilde=N_TILDE)
        self.mask = None
        self.model = None

    def setup(self, inputs: torch.Tensor, labels: torch.Tensor = None):
        height, width = inputs.shape[-2:]
        if self.circle_mask is not None:
            radius = math.sqrt((1.0 - self.q) * height * width / math.pi)
            y = torch.arange(height, device=inputs.device, dtype=torch.float32).view(
                1, 1, height, 1
            )
            x = torch.arange(width, device=inputs.device, dtype=torch.float32).view(
                1, 1, 1, width
            )
            distance_sq = (y - height / 2.0) ** 2 + (x - width / 2.0) ** 2
            outside = (distance_sq > radius**2).float().expand(len(inputs), -1, -1, -1)
            self.mask = 1.0 - outside if self.circle_mask == "in" else outside
            return

        if labels is None:
            raise ValueError("LG-SmoothGrad mask estimation requires labels.")
        # As in the original sampler, setup loads its own fixed VGG16 classifier.
        self.model = load_vgg16(inputs.device)
        selector = torch.zeros((len(inputs), 1000), device=inputs.device)
        selector.scatter_(1, labels.unsqueeze(1), 1.0)
        std = P_TILDE * (
            inputs.amax(dim=(1, 2, 3), keepdim=True)
            - inputs.amin(dim=(1, 2, 3), keepdim=True)
        )
        average_gradient = torch.zeros_like(inputs)
        for step in range(1, N_TILDE + 1):
            # Retain the unperturbed first point from the original implementation.
            if step == 1:
                point = inputs.clone().detach()
            else:
                point = (inputs + torch.randn_like(inputs) * std).clone().detach()
            gradients = []
            for start in range(0, len(inputs), 5):
                gradients.append(
                    input_gradient(
                        self.model,
                        point[start : start + 5],
                        selector[start : start + 5],
                    )
                )
            gradient = torch.cat(gradients, dim=0)
            average_gradient = ((step - 1) * average_gradient + gradient) / step
        average_magnitude = torch.linalg.vector_norm(
            average_gradient, dim=1, keepdim=True
        )
        threshold = torch.quantile(
            average_magnitude.view(len(inputs), -1), self.q, dim=1
        ).view(-1, 1, 1, 1)
        self.mask = (average_magnitude <= threshold).float()

    def compute_sample(
        self, inputs: torch.Tensor, labels: torch.Tensor = None
    ) -> torch.Tensor:
        if self.mask is None:
            raise ValueError("Call setup() before generating LG-SmoothGrad samples.")
        std = self.p * (
            inputs.amax(dim=(1, 2, 3), keepdim=True)
            - inputs.amin(dim=(1, 2, 3), keepdim=True)
        )
        noise = torch.randn_like(inputs) * std
        noise = noise * self.mask
        return inputs + noise

    def cleanup(self):
        self.mask = None
        super().cleanup()
