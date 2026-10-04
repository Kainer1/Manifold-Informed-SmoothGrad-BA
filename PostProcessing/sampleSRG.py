"""SRG of individual saved sample gradients, evaluated on original images."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import quantus
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Helper.model import load_vgg16  # noqa: E402
from Helper.paths import RESULTS_ROOT  # noqa: E402
from PostProcessing.sampleFourierAnalysis import check_group, open_samples  # noqa: E402
from Sampling.AttributionComputation import input_gradient, pool  # noqa: E402
from run import load_images  # noqa: E402


def srg(metric, model, original, label, gradients, device):
    count = len(gradients)
    scores = np.asarray(
        metric(
            model=model,
            x_batch=np.repeat(original[None], count, axis=0),
            y_batch=np.full(count, label, dtype=np.int64),
            a_batch=pool(gradients)[:, None],
            device=device,
        ),
        dtype=float,
    ).reshape(-1)
    if len(scores) != count:
        raise ValueError("SRG returned an unexpected number of scores.")
    return scores


def analyze(args, model=None):
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    model = load_vgg16(device) if model is None else model.to(device).eval()
    base = args.results_dir / "VGG16" / "IMAGENET_256"
    output = base / "Stats" / "SampleSRG"
    output.mkdir(parents=True, exist_ok=True)
    for sampler in args.samplers:
        sample_rows = []
        metric = quantus.SymmetricRelevanceGain(features_in_step=256, abs=True)
        with ExitStack() as stack:
            source = open_samples(stack, base / "Samples" / "allSamples", sampler)
            group = str(source.attrs.get("group_name", ""))
            indices = (
                list(dict.fromkeys(args.indices))
                if args.indices is not None
                else sorted(
                    int(key)
                    for key in source
                    if key.isdigit() and source[key].attrs.get("complete", False)
                )
            )
            if not indices:
                raise ValueError(f"No complete image indices in {source.filename}.")
            for index in indices:
                samples, label = check_group(source, index)
                originals, labels = load_images([index])
                if int(labels[0]) != label:
                    raise ValueError(
                        f"Original/sample label mismatch at image {index}."
                    )
                original = originals[0].cpu().numpy()
                for start in range(0, len(samples), args.batch_size):
                    batch = torch.from_numpy(
                        samples[start : start + args.batch_size]
                    ).to(device)
                    selector = torch.zeros((len(batch), 1000), device=device)
                    selector[:, label] = 1.0
                    gradients = input_gradient(model, batch, selector).detach()
                    scores = srg(
                        metric,
                        model,
                        original,
                        label,
                        gradients.cpu().numpy(),
                        device,
                    )
                    sample_rows.extend(
                        {
                            "Samplername": sampler,
                            "Gruppenname": group,
                            "Index": index,
                            "SampleIndex": start + position,
                            "Label": label,
                            "SRG": float(score),
                        }
                        for position, score in enumerate(scores)
                    )
                print(f"{sampler}, image {index}: evaluated {len(samples)} samples.")
        pd.DataFrame(sample_rows).to_csv(output / f"samples_{sampler}.csv", index=False)
        print(f"Saved SRG results for {sampler} to {output}.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samplers", nargs="+", required=True)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
