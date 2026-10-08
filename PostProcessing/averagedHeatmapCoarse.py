"""Export Coarse scores of saved signed-mean attribution heatmaps, as in Fig. C.1."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.fft import dctn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Helper.paths import RESULTS_ROOT  # noqa: E402

SCALES = (8, 16, 32, 64)


def coarse_scores(heatmap: np.ndarray) -> dict[int, float]:
    """Fraction of nonconstant DCT-II energy at wavelengths >= each scale."""
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ValueError("Expected a finite two-dimensional heatmap.")
    heatmap = heatmap.astype(np.float32, copy=False)
    centered = heatmap - heatmap.mean()
    coefficients = dctn(centered, axes=(-2, -1), norm="ortho")
    power = np.square(coefficients.astype(np.float64))
    power[0, 0] = 0.0
    total = power.sum()
    if total <= np.finfo(np.float64).eps:
        return {scale: float("nan") for scale in SCALES}
    height, width = heatmap.shape
    radius = np.hypot(
        np.arange(height)[:, None] / height,
        np.arange(width)[None, :] / width,
    )
    return {
        scale: float(power[radius <= 2.0 / scale].sum() / total) for scale in SCALES
    }


def analyze(args):
    if not args.samplers or len(set(args.samplers)) != len(args.samplers):
        raise ValueError("Provide distinct sampler names.")
    base = args.results_dir / "VGG16" / "IMAGENET_256"
    rows, summaries = [], []
    with ExitStack() as stack:
        files = {}
        for name in args.samplers:
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError("Supply exact sampler names without paths.")
            files[name] = stack.enter_context(
                h5py.File(base / "Attributions" / f"attributions_{name}.h5", "r")
            )
        coverage = {
            name: sorted(int(key) for key in source if key.isdigit())
            for name, source in files.items()
        }
        indices = args.indices
        if indices is None:
            indices = coverage[args.samplers[0]]
            if any(covered != indices for covered in coverage.values()):
                raise ValueError(
                    "Samplers have different image indices; select --indices explicitly."
                )
        if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
            raise ValueError("Provide distinct nonnegative image indices.")
        for name, source in files.items():
            scores = []
            for index in indices:
                if str(index) not in source:
                    raise ValueError(
                        f"Missing image index {index} in {source.filename}."
                    )
                # This array already contains the signed mean gradient: pool only now.
                attribution = source[str(index)]["attributions"][:]
                if attribution.shape != (3, 256, 256):
                    raise ValueError(
                        f"Invalid attribution shape at image {index}: {attribution.shape}."
                    )
                if not np.isfinite(attribution).all():
                    raise ValueError(
                        f"Non-finite attribution in {source.filename}, image {index}."
                    )
                heatmap = np.linalg.norm(attribution, axis=0)
                metrics = coarse_scores(heatmap)
                scores.append([metrics[scale] for scale in SCALES])
                rows.append(
                    {
                        "Sampler": name,
                        "Group": str(source.attrs.get("group_name", "")),
                        "Image Index": index,
                        **{f"Coarse {scale}": metrics[scale] for scale in SCALES},
                    }
                )
            values = np.asarray(scores)
            summary = {
                "Sampler": name,
                "Images": len(indices),
                "Source": str(Path(source.filename).resolve()),
            }
            for column, scale in enumerate(SCALES):
                # Preserve Fig. C.1 semantics: an undefined image makes the mean undefined.
                summary[f"Coarse {scale} Mean"] = float(values[:, column].mean())
                summary[f"Coarse {scale} Valid Images"] = int(
                    np.isfinite(values[:, column]).sum()
                )
            summaries.append(summary)
            print(
                f"{name}: {len(indices)} images, mean Coarse 32 = {summary['Coarse 32 Mean']:.6f}"
            )
    output = args.output_dir or base / "Stats" / "AveragedHeatmapCoarse"
    output.mkdir(parents=True, exist_ok=True)
    per_image, summary = pd.DataFrame(rows), pd.DataFrame(summaries)
    per_image.to_csv(output / "per_image.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    definitions = pd.DataFrame(
        [
            (
                "Representation",
                "Saved signed mean of ground-truth-logit input gradients, then RGB-L2 pooling. No display normalization.",
            ),
            (
                "Coarse",
                "Orthonormal 2D DCT-II of the mean-centered pooled heatmap; DC excluded. Energy at radial frequency <= 2/scale divided by total nonconstant energy.",
            ),
            ("Scales", "8, 16, 32 and 64 pixels. Coarse 32 is used in Fig. C.1."),
            (
                "Aggregation",
                "Arithmetic mean of one score per image, with equal image weights and identical selected indices across samplers.",
            ),
            (
                "Missing values",
                "Constant maps have undefined scores (blank). An undefined image makes the summary mean undefined; valid image counts are provided.",
            ),
        ],
        columns=["Item", "Definition"],
    )
    target = output / "averaged_heatmap_coarse.xlsx"
    temporary = target.with_suffix(".tmp.xlsx")
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        for name, frame in [
            ("Summary", summary),
            ("PerImage", per_image),
            ("Definitions", definitions),
        ]:
            frame.to_excel(writer, sheet_name=name, index=False, freeze_panes=(1, 0))
            sheet = writer.sheets[name]
            sheet.auto_filter.ref = sheet.dimensions
            sheet.column_dimensions["A"].width = 55
    temporary.replace(target)
    print(f"Saved Coarse results to {target}.")
    return target


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samplers", "--sampler-names", nargs="+", required=True)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
