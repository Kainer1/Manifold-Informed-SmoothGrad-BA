"""Original ordinary heatmap grid and optional individual PNG exports."""

import cmcrameri.cm as cmc
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from Sampling.AttributionComputation import normalize_heatmap, pool
from Helper.paths import RESULTS_DIR


def get_samplers_for_group(group_name):
    samplers = []
    for path in (RESULTS_DIR / "Attributions").glob("attributions_*.h5"):
        try:
            with h5py.File(path, "r") as file:
                if file.attrs.get("group_name") == group_name:
                    samplers.append(path.stem.removeprefix("attributions_"))
        except OSError as error:
            print(f"Warning: could not read {path.name} ({error})")
    return sorted(samplers)


def save_individual_heatmaps(
    attr_data: dict,
    sampler_names: list,
    indices: list,
    output_dir,
):
    """Save native-resolution heatmaps directly, without a Matplotlib figure."""
    saved_count = 0
    for sampler_name in sampler_names:
        sampler_dir = output_dir / sampler_name
        sampler_dir.mkdir(parents=True, exist_ok=True)

        for idx in indices:
            if idx not in attr_data.get(sampler_name, {}):
                continue

            attr = attr_data[sampler_name][idx]
            if attr.ndim > 2:
                attr = pool(attr)
            normalized = normalize_heatmap(attr)
            rgba = cmc.batlow(normalized, bytes=True)
            image = Image.fromarray(rgba)
            image.save(
                sampler_dir / f"index_{int(idx):05d}.png",
                format="PNG",
                compress_level=0,
            )
            saved_count += 1

    print(
        f"-> {saved_count} einzelne Heatmaps ohne Plot-Resizing gespeichert unter: "
        f"{output_dir}"
    )


def plot_heatmaps(group_name, indices, original_inputs, *, SafeHeatmapPNG=False):
    """Plot every requested index and optionally save each displayed heatmap."""
    if not indices:
        return
    samplers = get_samplers_for_group(group_name)
    if not samplers:
        print(f"No attribution samplers found for group '{group_name}'.")
        return
    baselines = get_samplers_for_group("baseline")
    samplers = [name for name in baselines if name not in samplers] + samplers

    attr_data = {name: {} for name in samplers}
    img_size = 256
    for name in samplers:
        path = RESULTS_DIR / "Attributions" / f"attributions_{name}.h5"
        with h5py.File(path, "r") as file:
            for index in indices:
                if str(index) in file:
                    attr_data[name][index] = file[str(index)]["attributions"][:]
                    img_size = attr_data[name][index].shape[-1]

    inputs_batch = torch.stack(original_inputs)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    inputs_zo = torch.clamp(inputs_batch * std + mean, 0, 1).cpu().numpy()
    n_images = len(indices)
    fig, axes = plt.subplots(
        nrows=1 + len(samplers),
        ncols=n_images,
        figsize=(2.5 * n_images, 2.5 * (1 + len(samplers))),
        squeeze=False,
    )
    for col, index in enumerate(indices):
        axes[0, col].axis("off")
        axes[0, col].imshow(np.clip(inputs_zo[col].transpose(1, 2, 0), 0, 1))
        axes[0, col].set_title(f"Image {index}", fontsize=10, fontweight="bold")

    for row, name in enumerate(samplers, start=1):
        for col, index in enumerate(indices):
            ax = axes[row, col]
            if index in attr_data[name]:
                attr = attr_data[name][index]
                if attr.ndim > 2:
                    attr = pool(attr)
                ax.imshow(normalize_heatmap(attr), cmap=cmc.batlow, vmin=0, vmax=1)
            else:
                ax.imshow(
                    np.zeros((img_size, img_size)), cmap="gray", vmin=0, vmax=1, alpha=0
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(
                    name,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontweight="bold",
                    fontsize=9,
                )
                ax.yaxis.set_visible(True)

    plt.tight_layout()
    out_dir = RESULTS_DIR / "Heatmaps" / "OnlyHeatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"heatmaps_{group_name}.png"
    if SafeHeatmapPNG:
        save_individual_heatmaps(
            attr_data,
            samplers,
            indices,
            RESULTS_DIR / "Heatmaps" / "IndividualHeatmaps" / group_name,
        )
    plt.savefig(out_file, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"-> Plot saved to: {out_file}")
