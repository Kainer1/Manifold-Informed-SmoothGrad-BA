"""Run the selected sampler with the ImageNet/VGG16 thesis pipeline."""

import argparse
import glob
import json
from pathlib import Path

import h5py
import torch
from torchvision.models import VGG16_Weights, vgg16
from torchvision.transforms import InterpolationMode, Normalize
from torchvision.transforms import functional as TF

from Evaluation.evaluation import run_evaluation
from Sampling.AttributionComputation import compute_and_save_attributions
from Sampling.RawGradientStatistics import (
    RawGradientAccumulator,
    save_raw_gradient_values,
)
from Sampling.Samplers.LatentDiffusion import DiTSampler, StableDiffusion2Sampler
from Sampling.Samplers.LGSmoothGrad import LGSmoothGradSampler
from Sampling.Samplers.PixelDiffusion import ADMSampler, DeepFloydSampler
from Sampling.Samplers.SmoothGrad import SmoothGradSampler
from Sampling.Samplers.VAE import VAESampler
from Sampling.Samplers.VanillaGradient import VanillaGradientSampler
from Storage.AllSamples import AllSampleWriter
from Visualisations.SampleVisualisation import (
    collect_visualization_samples,
    plot_saved_samples,
)
from Visualisations.heatmaps import plot_heatmaps

# Set this to your existing ImageNet HDF5 file before running the script.
# The required file structure and preprocessing are described in README.md.
IMAGENET_H5_PATH = Path("/path/to/ImageNet-Val1000-256x256.h5")

SAMPLER_CLASSES = {
    "VanillaGradient": VanillaGradientSampler,
    "SmoothGrad": SmoothGradSampler,
    "LG-SmoothGrad": LGSmoothGradSampler,
    "VAE_SD2": VAESampler,
    "VAE_REPAE_ImageNet": VAESampler,
    "DiT": DiTSampler,
    "SD2": StableDiffusion2Sampler,
    "DeepFloyd": DeepFloydSampler,
    "ADM": ADMSampler,
}

_MISSING = object()


def get_setting(settings, key, *, context="config", default=_MISSING):
    """Check presence before reading; missing required settings stop the run."""
    if not isinstance(settings, dict):
        raise ValueError(f"{context} must be a JSON object.")
    if key in settings:
        return settings[key]
    if default is not _MISSING:
        return default
    raise ValueError(f"Missing required setting: {context}.{key}")


def load_configs(config_path=None, config_dir=None):
    """Load all settings without dropping or rewriting configuration keys."""
    if config_path:
        paths = [config_path]
    elif config_dir:
        # Keep the original glob order; do not reorder sampler runs.
        paths = glob.glob(str(Path(config_dir) / "*.json"))
    else:
        raise ValueError("Specify --config or --config_dir.")

    configs = []
    for path in paths:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Each configuration in {path} must be a JSON object.")
            configs.append(entry)

    if not configs:
        raise ValueError("No configurations found.")
    return configs


def validate_config(config, position):
    """Check the shared fields; sampler-specific checks follow at their use."""
    context = f"config[{position}]"
    for key in ("sampler_name", "group_name"):
        value = get_setting(config, key, context=context)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context}.{key} must be a nonempty string.")
        if key == "sampler_name" and value not in SAMPLER_CLASSES:
            raise ValueError(f"Unknown sampler_name: {value!r}")

    n_samples = get_setting(config, "n_samples", context=context)
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError(f"{context}.n_samples must be a positive integer.")
    get_setting(config, "indices", context=context)

    sampler_config = get_setting(config, "config", context=context)
    if not isinstance(sampler_config, dict):
        raise ValueError(f"{context}.config must be a JSON object.")

    flags = get_setting(config, "flags", context=context)
    if not isinstance(flags, dict):
        raise ValueError(f"{context}.flags must be a JSON object.")
    for key in ("SampleVisualizationIndices", "SafeHeatmapIndices"):
        if key in flags:
            selection = get_setting(flags, key, context=f"{context}.flags")
            if not isinstance(selection, list) or any(
                type(index) is not int for index in selection
            ):
                raise ValueError(
                    f"{context}.flags.{key} must be a list of integer indices, e.g. [0, 1]."
                )


def load_images(indices):
    """Read processed ImageNet rows with the original pixel normalization."""
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images, labels = [], []
    with h5py.File(IMAGENET_H5_PATH, "r") as file:
        inputs_ds, labels_ds = file["inputs"], file["labels"]
        for index in indices:
            pixels = torch.from_numpy(inputs_ds[index]).float() / 255.0
            image = normalize(pixels)
            image = TF.resize(
                image, 256, interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            images.append(TF.center_crop(image, 256))
            labels.append(labels_ds[index])
    return images, torch.tensor(labels, dtype=torch.long)


def get_correct_indices(model, count, device):
    """Scan from row 0 in the original batches of 100, before sampler geometry."""
    with h5py.File(IMAGENET_H5_PATH, "r") as file:
        total = len(file["labels"])
    selected = []
    model.eval()
    for start in range(0, total, 100):
        if len(selected) >= count:
            break
        batch_indices = list(range(start, min(start + 100, total)))
        images, labels = load_images(batch_indices)
        with torch.no_grad():
            logits = model(torch.stack(images).to(device))
            matches = (torch.argmax(logits, dim=1) == labels.to(device)).cpu()
        for index, match in zip(batch_indices, matches):
            if match.item():
                selected.append(index)
                if len(selected) == count:
                    break
    if len(selected) < count:
        print(
            f"Warning: found only {len(selected)} of {count} correctly classified images."
        )
    return selected


def resolve_indices(value, model, device):
    """Retain the original index formats and inclusive range endpoints."""
    if isinstance(value, list):
        return value
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        value = value.strip()
        if "correct" in value.lower():
            count = next(
                (int(part) for part in value.split("_") if part.isdigit()), 100
            )
            return get_correct_indices(model, count, device)
        if "-" in value:
            start, end = map(int, value.split("-"))
            return list(range(start, end + 1))
        if value.isdigit():
            return [int(value)]
    raise ValueError(f"Invalid indices: {value!r}")


def main():
    parser = argparse.ArgumentParser(description="ImageNet/VGG16 thesis experiments")
    parser.add_argument("--config", help="Path to a config JSON object or list")
    parser.add_argument("--config_dir", help="Directory containing config JSON files")
    args = parser.parse_args()

    try:
        configs_to_run = load_configs(args.config, args.config_dir)
        for position, config in enumerate(configs_to_run):
            validate_config(config, position)
        if not IMAGENET_H5_PATH.is_file():
            raise FileNotFoundError(
                f"ImageNet file not found: {IMAGENET_H5_PATH}. "
                "Set IMAGENET_H5_PATH at the top of run.py; see README.md."
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()
    print(f"Loaded VGG16 on {device} ({model.classifier[-1].out_features} classes).")

    for position, config in enumerate(configs_to_run):
        try:
            indices = resolve_indices(
                get_setting(config, "indices", context=f"config[{position}]"),
                model,
                device,
            )
            sampler_config = get_setting(config, "config")
            flags = get_setting(config, "flags")
            visualization_indices = get_setting(
                flags, "SampleVisualizationIndices", default=indices[:10]
            )
            heatmap_indices = get_setting(
                flags, "SafeHeatmapIndices", default=indices[:10]
            )
            for key, selection in (
                ("SampleVisualizationIndices", visualization_indices),
                ("SafeHeatmapIndices", heatmap_indices),
            ):
                for index in selection:
                    if index not in indices:
                        raise ValueError(
                            f"{key} contains index {index}, which is not in this configuration's indices."
                        )
            inputs_list, labels = load_images(indices)
            sampler_name = get_setting(config, "sampler_name")
            sampler_class = SAMPLER_CLASSES[sampler_name]
            sampler = sampler_class(
                sampler_name=sampler_name,
                group_name=get_setting(config, "group_name"),
                n_samples=get_setting(config, "n_samples"),
                sampler_config=sampler_config,
                flags=flags,
            )
            save_all_samples = get_setting(flags, "saveAllSamples", default=False)
            if not isinstance(save_all_samples, bool):
                raise ValueError("flags.saveAllSamples must be true or false.")
            safe_heatmap_png = get_setting(flags, "SafeHeatmapPNG", default=False)
            if not isinstance(safe_heatmap_png, bool):
                raise ValueError("flags.SafeHeatmapPNG must be true or false.")
            raw_gradient_accumulator = RawGradientAccumulator(indices)
            writer = (
                AllSampleWriter(sampler.name, sampler.storage_config, sampler.n_samples)
                if save_all_samples
                else None
            )
            try:
                for start in range(0, len(indices), 100):
                    batch_indices = indices[start : start + 100]
                    batch_inputs = torch.stack(inputs_list[start : start + 100]).to(
                        device
                    )
                    batch_labels = labels[start : start + 100].to(device)
                    output_selector = torch.zeros(
                        (len(batch_indices), 1000), device=device
                    )
                    for position_in_batch, label in enumerate(batch_labels):
                        output_selector[position_in_batch, label] = 1.0
                    generator = sampler.generate(
                        batch_inputs,
                        batch_labels,
                        indices=batch_indices,
                        evaluation_model=model,
                        all_sample_writer=writer,
                    )
                    generator = collect_visualization_samples(
                        generator,
                        model,
                        batch_inputs,
                        batch_labels,
                        batch_indices,
                        visualization_indices,
                        sampler.name,
                        sampler.accuracy_dict,
                    )
                    try:
                        compute_and_save_attributions(
                            model,
                            generator,
                            sampler.name,
                            sampler.storage_config,
                            output_selector,
                            batch_indices,
                            raw_gradient_accumulator,
                        )
                    finally:
                        generator.close()
            finally:
                if writer is not None:
                    writer.close()

            save_raw_gradient_values(
                accumulator=raw_gradient_accumulator,
                sampler_name=sampler.name,
                group_name=sampler.group_name,
            )
            run_evaluation(
                group_name=sampler.group_name,
                model=model,
                original_inputs=inputs_list,
                labels=labels,
                indices=indices,
                device=device,
            )
            heatmap_positions = [indices.index(index) for index in heatmap_indices]
            plot_heatmaps(
                group_name=sampler.group_name,
                indices=heatmap_indices,
                original_inputs=[
                    inputs_list[position] for position in heatmap_positions
                ],
                SafeHeatmapPNG=safe_heatmap_png,
            )
            positions = [indices.index(index) for index in visualization_indices]
            plot_saved_samples(
                sampler.name,
                sampler.group_name,
                visualization_indices,
                [inputs_list[position] for position in positions],
                labels[positions],
                model,
            )
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            parser.error(f"config[{position}]: {error}")

    print("Pipeline calls complete.")


if __name__ == "__main__":
    main()
