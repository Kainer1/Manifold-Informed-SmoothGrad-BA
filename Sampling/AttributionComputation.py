"""Original signed-gradient averaging and HDF5 output without VAE projection."""

import h5py
import numpy as np
import torch

from Helper.paths import RESULTS_DIR
from Sampling.RawGradientStatistics import mean_absolute_grid


def input_gradient(model, inputs, output_selector):
    inputs = inputs.clone().detach().requires_grad_(True)
    logits = model(inputs)
    return torch.autograd.grad(logits, inputs, grad_outputs=output_selector)[0]


def compute_and_save_attributions(
    model,
    sampler_generator,
    sampler_name,
    sampler_config,
    output_selector,
    indices,
    raw_gradient_accumulator,
):
    grad_sum = None
    count = 0
    all_samples_abs_sum = None
    first_sample_abs_sum = None
    first_sample_grid = None
    grid_position = raw_gradient_accumulator.grid_position(indices)
    for sample in sampler_generator:
        gradient = input_gradient(model, sample, output_selector)
        if grad_sum is None:
            grad_sum = gradient
        else:
            grad_sum += gradient
        sample_abs_sum = torch.linalg.vector_norm(
            gradient.detach(), ord=1, dim=(1, 2, 3)
        ).to(torch.float64)
        if all_samples_abs_sum is None:
            all_samples_abs_sum = torch.zeros_like(sample_abs_sum)
            first_sample_abs_sum = sample_abs_sum.clone()
            if grid_position is not None:
                first_sample_grid = mean_absolute_grid(
                    gradient[grid_position],
                    grid_size=raw_gradient_accumulator.grid_size,
                )
        all_samples_abs_sum += sample_abs_sum
        count += 1
    if count == 0:
        print(f"[{sampler_name}] No samples produced; no attributions written.")
        return

    avg_grad = grad_sum / count
    final_attribution_abs_sum = torch.linalg.vector_norm(
        avg_grad.detach(), ord=1, dim=(1, 2, 3)
    ).to(torch.float64)
    raw_gradient_accumulator.record_batch(
        batch_indices=indices,
        all_samples_abs_sum=all_samples_abs_sum,
        first_sample_abs_sum=first_sample_abs_sum,
        final_attribution_abs_sum=final_attribution_abs_sum,
        sample_count=count,
        first_sample_grid=first_sample_grid,
        gradient_shape=tuple(avg_grad.shape[1:]),
    )
    attributions = avg_grad.cpu().detach().numpy()
    output_dir = RESULTS_DIR / "Attributions"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"attributions_{sampler_name}.h5"
    try:
        file = h5py.File(path, "a")
    except OSError:
        path.unlink(missing_ok=True)
        file = h5py.File(path, "a")
    with file:
        file.attrs["group_name"] = sampler_config["group_name"]
        for position, index in enumerate(indices):
            key = str(index)
            if key in file:
                del file[key]
            group = file.create_group(key)
            group.attrs["sampler_config"] = str(sampler_config)
            group.create_dataset("attributions", data=attributions[position])
    print(f"[{sampler_name}] Saved attributions for {len(indices)} images.")


def pool(attribution: np.ndarray) -> np.ndarray:
    """Pool attributions by computing the norm over color channels."""
    if attribution.ndim == 3:
        return np.linalg.norm(attribution, axis=0)
    elif attribution.ndim == 4:
        return np.linalg.norm(attribution, axis=1)
    else:
        raise ValueError(f"Unexpected attribution shape: {attribution.shape}")


def normalize_heatmap(attr: np.ndarray) -> np.ndarray:
    """Normalize a single heatmap to [0, 1] using its 99.9th percentile."""
    absmax = np.percentile(attr, 99.9)
    return np.clip(attr / (absmax or 1e-10), 0, 1)
