"""Compare every saved sample heatmap with its image's Vanilla Gradient."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Helper.model import load_vgg16  # noqa: E402
from Helper.paths import RESULTS_ROOT  # noqa: E402
from PostProcessing.attributionSampleSimilarity import (  # noqa: E402
    _resolve_device,
    compute_l2_pooled_attributions,
)
from PostProcessing.sampleFourierAnalysis import check_group, open_samples  # noqa: E402

COLUMNS = ["Sampler", "Image Index", "Sample Index", "Label", "Cosine", "Spearman"]


def reference_similarities(heatmaps, reference):
    """Cosine and average-tie-rank Spearman on raw, flattened L2 maps."""
    maps = np.asarray(heatmaps, dtype=np.float64)
    baseline = np.asarray(reference, dtype=np.float64)
    if maps.ndim != 3 or baseline.shape != maps.shape[1:]:
        raise ValueError(
            "Sample and reference heatmaps must have matching spatial shapes."
        )
    if not np.isfinite(maps).all() or not np.isfinite(baseline).all():
        raise ValueError("Heatmaps contain non-finite values.")
    flat = maps.reshape(len(maps), -1)
    ref = baseline.ravel()

    def cosine_to_reference(values, target):
        denominator = np.linalg.norm(values, axis=1) * np.linalg.norm(target)
        result = np.divide(
            values @ target,
            denominator,
            out=np.full(len(values), np.nan),
            where=denominator > 0,
        )
        return np.clip(result, -1, 1)

    cosine = cosine_to_reference(flat, ref)
    ranks = rankdata(flat, axis=1, method="average")
    ref_ranks = rankdata(ref, method="average")
    spearman = cosine_to_reference(
        ranks - ranks.mean(axis=1, keepdims=True), ref_ranks - ref_ranks.mean()
    )
    return cosine, spearman


def validate_inputs(files, reference_name, sampler_names, requested_indices):
    """Require matching image coverage and labels, with one reference per image."""
    coverage = {
        name: sorted(
            int(key)
            for key in files[name]
            if key.isdigit() and files[name][key].attrs.get("complete", False)
        )
        for name in sampler_names
    }
    indices = requested_indices
    if indices is None:
        indices = coverage[sampler_names[0]]
        if any(covered != indices for covered in coverage.values()):
            raise ValueError(
                "Samplers have different complete image indices; select --indices explicitly."
            )
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError("Provide distinct nonnegative image indices.")
    for index in indices:
        reference, label = check_group(files[reference_name], index)
        if len(reference) != 1:
            raise ValueError(
                f"Vanilla reference needs exactly one sample at image {index}."
            )
        for name in sampler_names:
            samples, target = check_group(files[name], index)
            if target != label:
                raise ValueError(f"Target label mismatch at image {index}: {name}.")
            if samples.shape[1:] != reference.shape[1:]:
                raise ValueError(f"Input geometry mismatch at image {index}: {name}.")
    return indices


def write_results(output, rows, reference_name):
    data = pd.DataFrame(rows, columns=COLUMNS)
    summary = (
        data.groupby("Sampler", sort=False)
        .agg(
            Images=("Image Index", "nunique"),
            Samples=("Sample Index", "size"),
            Cosine_Mean=("Cosine", "mean"),
            Cosine_Valid=("Cosine", "count"),
            Spearman_Mean=("Spearman", "mean"),
            Spearman_Valid=("Spearman", "count"),
        )
        .reset_index()
    )
    per_image = (
        data.groupby(["Sampler", "Image Index"], sort=False)
        .agg(
            Samples=("Sample Index", "size"),
            Cosine_Mean=("Cosine", "mean"),
            Cosine_Valid=("Cosine", "count"),
            Spearman_Mean=("Spearman", "mean"),
            Spearman_Valid=("Spearman", "count"),
        )
        .reset_index()
    )
    definitions = pd.DataFrame(
        [
            (
                "Representation",
                "Ground-truth-logit input gradients; RGB-L2 pooling, then flatten. No display normalization.",
            ),
            (
                "Reference",
                f"One saved unperturbed input from {reference_name}, matched by image index and label.",
            ),
            (
                "Cosine",
                "Dot product divided by the product of L2 norms; undefined for a zero vector.",
            ),
            (
                "Spearman",
                "Pearson correlation of pixel ranks, with average ranks for ties; undefined for constant maps.",
            ),
            (
                "Aggregation",
                "Summary averages all finite individual sample coefficients. Images with more valid samples receive more weight; PerImage gives within-image means.",
            ),
            (
                "Missing values",
                "Undefined coefficients are blank; valid counts record each mean's denominator.",
            ),
            ("Sample Index", "Zero-based position in the saved samples dataset."),
        ],
        columns=["Item", "Definition"],
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("samples", data),
        ("per_image", per_image),
        ("summary", summary),
    ]:
        frame.to_csv(output / f"sample_to_vanilla_{name}.csv", index=False)
    target = output / "sample_to_vanilla_similarity.xlsx"
    temporary = target.with_suffix(".tmp.xlsx")
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        for name, frame in [
            ("Summary", summary),
            ("PerImage", per_image),
            ("Samples", data),
            ("Definitions", definitions),
        ]:
            frame.to_excel(writer, sheet_name=name, index=False, freeze_panes=(1, 0))
            sheet = writer.sheets[name]
            sheet.auto_filter.ref = sheet.dimensions
            sheet.column_dimensions["A"].width = 55
    temporary.replace(target)
    return target


def analyze(args, model=None):
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if not args.samplers or len(set(args.samplers)) != len(args.samplers):
        raise ValueError("Provide distinct sampler names.")
    if args.reference_sampler in args.samplers:
        raise ValueError("The reference must not be included as a sample sampler.")
    base = args.results_dir / "VGG16" / "IMAGENET_256"
    with ExitStack() as stack:
        files = {
            name: open_samples(stack, base / "Samples" / "allSamples", name)
            for name in [args.reference_sampler, *args.samplers]
        }
        indices = validate_inputs(
            files, args.reference_sampler, args.samplers, args.indices
        )
        device = _resolve_device(args.device)
        model = load_vgg16(device) if model is None else model.to(device).eval()
        rows = []
        for index in indices:
            reference, label = check_group(files[args.reference_sampler], index)
            baseline = compute_l2_pooled_attributions(
                model, reference, label, device, 1
            ).numpy()[0]
            for name in args.samplers:
                samples, _ = check_group(files[name], index)
                maps = compute_l2_pooled_attributions(
                    model, samples, label, device, args.batch_size
                ).numpy()
                cosine, spearman = reference_similarities(maps, baseline)
                rows.extend(
                    [name, index, sample, label, float(c), float(s)]
                    for sample, (c, s) in enumerate(zip(cosine, spearman, strict=True))
                )
            print(f"Image {index}: compared samples with Vanilla Gradient.", flush=True)
    output = args.output_dir or base / "Stats" / "SampleToVanillaSimilarity"
    target = write_results(output, rows, args.reference_sampler)
    print(f"Saved {len(rows)} comparisons to {target}.")
    return target


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samplers", "--sampler-names", nargs="+", required=True)
    parser.add_argument("--reference-sampler", required=True)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
