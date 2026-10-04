"""Original SSIM representatives and analysis plots for ImageNet/VGG16."""

import os
import sys
from pathlib import Path

import cmcrameri.cm as cmc
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim

from Sampling.AttributionComputation import input_gradient, normalize_heatmap, pool
from Helper.paths import RESULTS_DIR

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_lpips_model = None


def get_lpips_model(device):
    global _lpips_model
    if _lpips_model is None:
        import lpips

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            _lpips_model = lpips.LPIPS(net="alex").to(device)
            _lpips_model.eval()
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
    else:
        _lpips_model = _lpips_model.to(device)
    return _lpips_model


def _unclamp_to_zeroone(
    tensor: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor
) -> torch.Tensor:
    """Wandelt einen netzwerkspezifischen Tensor linear in den ungeclampeten [0, 1] Farbraum um."""
    return tensor * std_t + mean_t


def compute_radial_profile(data: np.ndarray) -> np.ndarray:
    """
    Computes the radial profile (1D FFT representation) of a 2D image or Fourier spectrum.
    """
    y, x = np.indices((data.shape))
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1])

    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = data.flat[ind]

    r_int = r_sorted.astype(int)
    deltar = r_int[1:] - r_int[:-1]
    rind = np.where(deltar)[0]
    nr = rind[1:] - rind[:-1]

    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]
    radial_prof = np.divide(tbin, nr, out=np.zeros_like(tbin), where=nr > 0)
    return radial_prof


def compute_2d_fft(image: torch.Tensor) -> np.ndarray:
    """
    Computes the 2D Fourier Magnitude Spectrum of a given 2D tensor.
    Expects a 2D tensor (e.g. grayscale image).
    Returns the magnitude on CPU as a NumPy array.
    """
    image_centered = image - image.mean()
    amp = torch.abs(torch.fft.fftshift(torch.fft.fft2(image_centered))).cpu().numpy()
    return amp


def collect_visualization_samples(
    sampler_generator,
    model,
    original_inputs,
    labels,
    indices,
    visualization_indices,
    sampler_name,
    accuracy_dict,
):
    """Forward every sample and collect the selected images after consumption."""
    selected = indices[:10] if visualization_indices is None else visualization_indices
    positions = [
        position for position, index in enumerate(indices) if index in selected
    ]
    collected_samples = []
    sample_count = 0
    try:
        for sample in sampler_generator:
            yield sample
            if positions:
                collected_samples.append(sample[positions].detach().cpu())
            sample_count += 1
    finally:
        if hasattr(sampler_generator, "close"):
            sampler_generator.close()

    if sample_count == 0:
        return
    if collected_samples:
        save_visualization_samples(
            sampler_name=sampler_name,
            original_inputs=original_inputs[positions],
            collected_samples=collected_samples,
            indices=[indices[position] for position in positions],
            labels=labels[positions],
            model=model,
            accuracy_dict=accuracy_dict,
        )


def save_visualization_samples(
    sampler_name: str,
    original_inputs: torch.Tensor,
    collected_samples: list,
    indices: list,
    labels: torch.Tensor,
    model,
    accuracy_dict: dict,
):
    """Berechnet Metriken auf den ungeclampeten Samples und sichert 4 Extrem-Samples pro Index in .h5."""
    if not collected_samples:
        return
    vis_file = RESULTS_DIR / "Samples" / "rawOutput" / f"outputs_{sampler_name}.h5"
    vis_file.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"  -> Berechne Metriken für Visualisierungen ({len(collected_samples)} Iterationen gesammelt)..."
    )

    device = original_inputs.device
    limit = collected_samples[0].size(0)
    lpips_model = get_lpips_model(device)
    eval_model = model.eval()

    mean_vals, std_vals = IMAGENET_MEAN, IMAGENET_STD
    mean_t = torch.tensor(mean_vals, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std_vals, device=device).view(1, 3, 1, 1)
    lpips_dev = next(lpips_model.parameters()).device

    with h5py.File(vis_file, "a") as f_vis:
        for b in range(limit):
            idx, lbl = indices[b], int(labels[b])
            orig_img = original_inputs[b].unsqueeze(0).to(device)
            orig_unclamped = _unclamp_to_zeroone(orig_img, mean_t, std_t)
            orig_m1p1 = orig_unclamped * 2.0 - 1.0

            samples_raw_b = [
                batch_tensor[b].to(device) for batch_tensor in collected_samples
            ]
            batch_s = torch.stack(samples_raw_b)  # [N, C, H, W]
            batch_unclamped = _unclamp_to_zeroone(batch_s, mean_t, std_t)

            with torch.no_grad():
                orig_attr_score = float(eval_model(orig_img)[0, lbl].item())

                # Batch evaluation for LPIPS and Model Logits
                s_m1p1 = batch_unclamped * 2.0 - 1.0
                orig_m1p1_rep = orig_m1p1.expand(batch_s.size(0), -1, -1, -1).to(
                    lpips_dev
                )
                d_lpips_list = (
                    lpips_model(s_m1p1.to(lpips_dev), orig_m1p1_rep)
                    .view(-1)
                    .cpu()
                    .tolist()
                )

                logits = eval_model(batch_s)
                attr_list = logits[:, lbl].cpu().tolist()
                max_list = torch.max(logits, dim=1).values.cpu().tolist()

            l1_list = (
                torch.linalg.vector_norm(
                    orig_unclamped - batch_unclamped, ord=1, dim=(1, 2, 3)
                )
                .cpu()
                .tolist()
            )
            l2_list = (
                torch.linalg.vector_norm(
                    orig_unclamped - batch_unclamped, ord=2, dim=(1, 2, 3)
                )
                .cpu()
                .tolist()
            )
            ssim_list = []
            for start in range(0, batch_s.size(0), 10):
                samples_chunk = batch_unclamped[start : start + 10]
                ssim_list.extend(
                    ssim(
                        samples_chunk,
                        orig_unclamped.expand_as(samples_chunk),
                        data_range=1.0,
                        reduction="none",
                    )
                    .cpu()
                    .tolist()
                )

            metrics = {
                "l1": l1_list,
                "l2": l2_list,
                "ssim": ssim_list,
                "lpips": d_lpips_list,
                "attr": attr_list,
                "max_score": max_list,
            }

            stats = {
                f"{m}_{st}": float(getattr(np, st)(metrics[m]))
                for m in ["l1", "l2", "ssim", "lpips", "attr"]
                for st in ["mean", "std", "min", "max"]
            }

            sorted_idx = np.argsort(metrics["ssim"])
            sel_idx = (
                np.array([sorted_idx[-1], sorted_idx[-2], sorted_idx[1], sorted_idx[0]])
                if len(sorted_idx) >= 4
                else sorted_idx
            )
            selected_samples = batch_s[sel_idx].cpu().numpy()

            grp_name = str(idx)
            if grp_name in f_vis:
                del f_vis[grp_name]

            grp = f_vis.create_group(grp_name)
            grp.create_dataset("samples", data=selected_samples)
            grp.create_dataset("original_sample_indices", data=sel_idx)
            for m in metrics:
                grp.create_dataset(f"{m}_scores", data=np.array(metrics[m])[sel_idx])

            grp.attrs["orig_attr_score"] = orig_attr_score
            grp.attrs["accuracy"] = float(accuracy_dict[idx])
            for k, v in stats.items():
                grp.attrs[k] = v

    print(f"  -> Visualisierungs-Samples in {vis_file} gesichert.")


def plot_saved_samples(
    sampler_name: str,
    group_name: str,
    indices: list,
    original_inputs,
    labels: torch.Tensor,
    model,
):
    """Lädt Visualisierungs-Samples und plottet das 6-Zeilen Analyse-Gitter pro Index."""
    vis_file = RESULTS_DIR / "Samples" / "rawOutput" / f"outputs_{sampler_name}.h5"
    if not vis_file.exists():
        print(f"Datei nicht gefunden: {vis_file}")
        return

    with h5py.File(vis_file, "r") as f:
        target_indices = [
            idx
            for idx in indices
            if str(idx) in f
            and "samples" in f[str(idx)]
            and f[str(idx)]["samples"].shape[0] > 0
        ]
    if not target_indices:
        print(f"Keine Visualisierungs-Samples in {vis_file} gefunden.")
        return

    viz_dir = RESULTS_DIR / "Samples" / group_name
    viz_dir.mkdir(parents=True, exist_ok=True)

    positions = [indices.index(index) for index in target_indices]
    inputs_list = [original_inputs[position] for position in positions]
    labels_list = labels[positions]
    device = next(model.parameters()).device
    model.eval()

    mean_vals, std_vals = IMAGENET_MEAN, IMAGENET_STD
    mean_t = torch.tensor(mean_vals).view(1, 3, 1, 1)
    std_t = torch.tensor(std_vals).view(1, 3, 1, 1)

    with h5py.File(vis_file, "r") as f:
        sample_shape = f[str(target_indices[0])]["samples"].shape
        num_samples = sample_shape[0]
        original_images = (
            torch.stack(inputs_list) if isinstance(inputs_list, list) else inputs_list
        )
        num_cols, n_images = num_samples + 1, len(target_indices)

        fig, axes = plt.subplots(
            n_images * 6,
            num_cols,
            figsize=(3 * num_cols, 13 * n_images),
            gridspec_kw={"height_ratios": [0.22, 1.0, 0.6, 1.0, 1.0, 1.0] * n_images},
        )
        if len(axes.shape) == 1:
            axes = axes.reshape(-1, num_cols)

        for i, idx in enumerate(target_indices):
            grp_name = str(idx)
            base_row, label = i * 6, int(labels_list[i])
            output_selector = torch.zeros((1, 1000), device=device)
            output_selector[0, label] = 1.0

            samples_np = f[grp_name]["samples"][:]
            scores_h5 = {
                m: f[grp_name][f"{m}_scores"][:] for m in ["l1", "l2", "ssim", "lpips"]
            }

            orig = original_images[i].unsqueeze(0)
            orig_unclamped = _unclamp_to_zeroone(orig, mean_t, std_t)
            orig_vis = (
                torch.clamp(orig_unclamped, 0.0, 1.0)
                .squeeze(0)
                .permute(1, 2, 0)
                .numpy()
            )
            orig_b = orig.to(device)

            with torch.no_grad():
                orig_logits = model(orig_b)
                orig_attr_score = float(orig_logits[0, label].item())
                orig_softmax_score = float(
                    torch.softmax(orig_logits, dim=1)[0, label].item() * 100.0
                )
            orig_attr_score = float(
                f[grp_name].attrs.get("orig_attr_score", orig_attr_score)
            )

            amp_orig = compute_2d_fft(orig_unclamped.squeeze(0).mean(dim=0))
            rad_orig = compute_radial_profile(amp_orig)
            heatmap_orig = normalize_heatmap(
                pool(input_gradient(model, orig_b, output_selector).cpu().numpy()[0])
            )

            diffs, amp_s_list, log_amp_s_list, heatmaps_s, max_scores = (
                [],
                [],
                [],
                [],
                [],
            )
            percentiles_fft = [np.percentile(np.log1p(amp_orig), 99.9)]
            percentiles_diff = []
            s_vis_list = []

            for j in range(num_samples):
                s_np = samples_np[j]
                while s_np.ndim > 3:
                    s_np = s_np.squeeze(0)
                s_raw = torch.from_numpy(s_np).unsqueeze(0)
                s_unclamped = _unclamp_to_zeroone(s_raw, mean_t, std_t)
                s_vis = torch.clamp(s_unclamped, 0.0, 1.0).squeeze(0)
                s_vis_list.append(s_vis.permute(1, 2, 0).numpy())

                diff = (
                    (orig_unclamped - s_unclamped).squeeze(0).mean(dim=0).cpu().numpy()
                )
                diffs.append(diff)
                percentiles_diff.append(np.percentile(np.abs(diff), 99.9))

                amp_s = compute_2d_fft(s_unclamped.squeeze(0).mean(dim=0))
                amp_s_list.append(amp_s)
                log_amp_s = np.log1p(amp_s)
                log_amp_s_list.append(log_amp_s)
                percentiles_fft.append(np.percentile(log_amp_s, 99.9))

                s_b = s_raw.to(device)
                with torch.no_grad():
                    logits = model(s_b)
                max_scores.append(
                    (
                        float(torch.max(logits).item()),
                        int(torch.argmax(logits, dim=1).item()),
                    )
                )
                heatmaps_s.append(
                    normalize_heatmap(
                        pool(
                            input_gradient(model, s_b, output_selector).cpu().numpy()[0]
                        )
                    )
                )

            attr_scores = (
                list(f[grp_name]["attr_scores"][:])
                if "attr_scores" in f[grp_name]
                else [m[0] for m in max_scores]
            )
            acc = float(f[grp_name].attrs.get("accuracy", 0.0))

            vlim = max(max(percentiles_diff), 1e-4)
            vmax_fft = max(max(percentiles_fft), 1e-5)
            all_rads = [rad_orig] + [compute_radial_profile(a) for a in amp_s_list]
            valid_mins = [
                r[3:-3][r[3:-3] > 0].min()
                for r in all_rads
                if len(r[3:-3][r[3:-3] > 0]) > 0
            ]
            ymin_1d = max(min(valid_mins), 1e-5) if valid_mins else 1e-5
            ymax_1d = max([r[3:-3].max() for r in all_rads if len(r) > 6])

            # --- SPALTE 0: ORIGINAL ---
            axes[base_row, 0].text(
                0.5,
                0.5,
                f"Original (ID: {idx})\nTarget Logit: {orig_attr_score:.2f}"
                f"\nSoftmax: {orig_softmax_score:.1f}%\nAcc: {acc:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )
            axes[base_row + 1, 0].imshow(orig_vis)

            # 1D FFT helper
            def _setup_1d_fft(ax, rad, color="black", ls="-", alpha=1.0):
                ax.plot(rad, color=color, ls=ls, alpha=alpha, lw=1, clip_on=False)
                ax.set_yscale("log")
                ax.set_ylim(ymin_1d, ymax_1d * 1.5)
                ax.set_xlim(left=3, right=len(rad) - 4)
                ax.set_xticks([])
                ax.tick_params(axis="y", labelsize=6)
                for sp in ["top", "right", "bottom"]:
                    ax.spines[sp].set_visible(False)

            _setup_1d_fft(axes[base_row + 2, 0], rad_orig)
            axes[base_row + 2, 0].set_ylabel("1D FFT", fontsize=8)
            axes[base_row + 3, 0].imshow(
                np.log1p(amp_orig), cmap="viridis", vmin=0, vmax=vmax_fft
            )

            # Difference Box / Stat-Tabelle
            stats_attrs = f[grp_name].attrs
            axes[base_row + 4, 0].text(
                0.5,
                0.88,
                f"LIMITS\nFFT vmax: {vmax_fft:.2f}  |  Diff vlim: [-{vlim:.2f}, +{vlim:.2f}]",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )
            metric_fmt = [
                ("l1", ".1f"),
                ("l2", ".1f"),
                ("ssim", ".3f"),
                ("lpips", ".3f"),
                ("attr", ".2f"),
            ]
            cell_text = [
                [
                    f"{stats_attrs.get(f'{m}_{st}', 0.0):{fmt}}"
                    for st in ["mean", "std", "min", "max"]
                ]
                for m, fmt in metric_fmt
            ]
            table = axes[base_row + 4, 0].table(
                cellText=cell_text,
                rowLabels=["L1", "L2", "SSIM", "LPIP", "ATTR"],
                colLabels=["MEAN", "STD", "MIN", "MAX"],
                cellLoc="center",
                bbox=[0.05, 0.0, 0.95, 0.72],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(6)

            axes[base_row + 5, 0].imshow(heatmap_orig, cmap=cmc.batlow, vmin=0, vmax=1)
            for r in range(6):
                axes[base_row + r, 0].axis("off")

            # --- SPALTEN 1..N: SAMPLES ---
            cats = (
                ["Best 1", "Best 2", "Worst 2", "Worst 1"]
                if num_samples == 4
                else [f"Sample {j + 1}" for j in range(num_samples)]
            )
            for j in range(num_samples):
                col = j + 1
                s_max, pred_c = max_scores[j]
                max_str = f"{s_max:.2f}" + (
                    f" (Cls {pred_c})" if pred_c != label else ""
                )
                hdr_txt = f"{cats[j]}\nL1: {scores_h5['l1'][j]:.1f} | L2: {scores_h5['l2'][j]:.1f}\nSSIM: {scores_h5['ssim'][j]:.3f} | LP: {scores_h5['lpips'][j]:.3f}\nAttr: {attr_scores[j]:.2f} | Max: {max_str}"

                axes[base_row, col].text(
                    0.5,
                    0.5,
                    hdr_txt,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
                axes[base_row + 1, col].imshow(s_vis_list[j])

                _setup_1d_fft(
                    axes[base_row + 2, col], rad_orig, color="gray", ls="--", alpha=0.5
                )
                axes[base_row + 2, col].plot(
                    all_rads[j + 1],
                    color="blue" if j < 2 else "red",
                    alpha=0.7,
                    clip_on=False,
                )
                axes[base_row + 3, col].imshow(
                    log_amp_s_list[j], cmap="viridis", vmin=0, vmax=vmax_fft
                )
                axes[base_row + 4, col].imshow(
                    diffs[j], cmap=cmc.vik, vmin=-vlim, vmax=vlim
                )
                axes[base_row + 5, col].imshow(
                    heatmaps_s[j], cmap=cmc.batlow, vmin=0, vmax=1
                )

                for r in range(6):
                    axes[base_row + r, col].axis("off")

        plt.subplots_adjust(hspace=0.2, wspace=0.1)
        out_png = viz_dir / f"viz_{sampler_name}.png"
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close("all")
        print(f"-> Plot erfolgreich generiert und gespeichert unter: {out_png}")
