"""Time sample generation on one ImageNet image using the Code/ samplers.

Retains the legacy CUDA timing, warm-up, median, and batch-one projection.
The retained samplers use the current class-conditioning settings without CFG;
the original benchmark used stronger guidance for DiT, SD2, and DeepFloyd.
Checkpoint loading is excluded from both setup and sample-generation timing.
"""

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

if "--allow-download" not in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch  # noqa: E402

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from Sampling.LoadModels import ADM_WEIGHTS_PATH  # noqa: E402
from Sampling.Samplers.LatentDiffusion import (  # noqa: E402
    DiTSampler,
    StableDiffusion2Sampler,
)
from Sampling.Samplers.PixelDiffusion import ADMSampler, DeepFloydSampler  # noqa: E402
from Sampling.Samplers.SmoothGrad import SmoothGradSampler  # noqa: E402
from Sampling.Samplers.VAE import VAESampler  # noqa: E402
from run import load_images  # noqa: E402

CHECKPOINTS = {
    "smoothgrad": "none (Gaussian pixel noise)",
    "repae": "REPA-E/e2e-invae-hf",
    "sd_vae": "stabilityai/sd-vae-ft-mse",
    "dit": "facebook/dit-xl-2-256",
    "sd2": "sd2-community/stable-diffusion-2-base",
    "adm": str(ADM_WEIGHTS_PATH),
    "deepfloyd": "DeepFloyd/IF-I-XL-v1.0 + DeepFloyd/IF-II-L-v1.0",
}


def make_sampler(name):
    common = dict(group_name="runtime_benchmark", n_samples=1, flags={})
    if name == "smoothgrad":
        return SmoothGradSampler(
            **common, sampler_name="SmoothGrad", sampler_config={"p": 0.09}
        )
    if name in ("repae", "sd_vae"):
        return VAESampler(
            **common,
            sampler_name="VAE_REPAE_ImageNet" if name == "repae" else "VAE_SD2",
            sampler_config={"p": 0.04 if name == "repae" else 0.025},
        )
    if name == "dit":
        return DiTSampler(
            **common,
            sampler_name="DiT",
            sampler_config={
                "strength": 0.10,
                "actual_steps": 5,
                "class_conditioning": True,
            },
        )
    if name == "sd2":
        return StableDiffusion2Sampler(
            **common,
            sampler_name="SD2",
            sampler_config={
                "strength": 0.05,
                "actual_steps": 20,
                "class_conditioning": True,
            },
        )
    if name == "adm":
        return ADMSampler(
            **common, sampler_name="ADM", sampler_config={"strength": 0.08}
        )
    if name == "deepfloyd":
        return DeepFloydSampler(
            **common,
            sampler_name="DeepFloyd",
            sampler_config={
                "strength": 0.10,
                "actual_steps": 20,
                "strength_2": 0.10,
                "actual_steps_2": 20,
                "class_conditioning": True,
            },
        )
    raise ValueError(name)


def reuse_loaded_models(sampler):
    """Reuse the initial models for the separately timed, second setup call.

    Legacy setup reused its checkpoints. Code/ setup loads them on every call;
    returning the existing objects here preserves the exclusion of loading.
    """
    if isinstance(sampler, VAESampler):
        return patch(
            "Sampling.Samplers.VAE.AutoencoderKL.from_pretrained",
            return_value=sampler.vae_model,
        )
    loaders = {
        DiTSampler: "Sampling.Samplers.LatentDiffusion.load_dit_imagenet",
        StableDiffusion2Sampler: "Sampling.Samplers.LatentDiffusion.load_sd2_img2img",
        ADMSampler: "Sampling.Samplers.PixelDiffusion.load_openai_adm",
        DeepFloydSampler: "Sampling.Samplers.PixelDiffusion.load_deepfloyd_if",
    }
    for sampler_class, loader in loaders.items():
        if isinstance(sampler, sampler_class):
            return patch(loader, return_value=sampler.model_pipeline)
    return nullcontext()


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start, result


def benchmark_one(name, image, label, warmup, repeats, images, samples):
    sampler = make_sampler(name)
    sampler.dataset_name = "IMAGENET_256"
    sampler.model_name = "VGG16"
    row = {
        "model": name,
        "checkpoint": CHECKPOINTS[name],
        "sampler_config": dict(sampler.sampler_config),
        "status": "ok",
    }
    try:
        # Load once outside timing. The fixed 256_projection output is identity.
        sampler.setup(image, labels=label)
        for _ in range(warmup):
            sampler.compute_sample(image, labels=label)
        with reuse_loaded_models(sampler):
            row["image_setup_s"], _ = timed(lambda: sampler.setup(image, labels=label))
        sample_times = []
        for _ in range(repeats):
            duration, _ = timed(lambda: sampler.compute_sample(image, labels=label))
            sample_times.append(duration)
        row["sample_s_median"] = statistics.median(sample_times)
        row["one_sample_s_estimate"] = row["image_setup_s"] + row["sample_s_median"]
        row["sample_s_values"] = sample_times
        row["projected_total_s_batch1"] = (
            images * row["image_setup_s"] + images * samples * row["sample_s_median"]
        )
        # Sample-generation-only projection used by the thesis runtime table.
        row["projected_sample_s_batch1"] = images * samples * row["sample_s_median"]
        row["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            sampler.cleanup()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=CHECKPOINTS,
        default=["smoothgrad", "repae", "dit", "adm"],
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--project-images", type=int, default=100)
    parser.add_argument("--project-samples", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit checkpoint downloads; the default uses local cache only.",
    )
    args = parser.parse_args()
    if (
        min(args.repeats, args.project_images, args.project_samples) < 1
        or args.warmup < 0
    ):
        parser.error(
            "repeats, project-images, project-samples must be positive; warmup >= 0"
        )
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for comparable timing.")
    torch.manual_seed(0)
    image_list, labels = load_images([args.index])
    image = torch.stack(image_list).to("cuda")
    label = labels.to("cuda")
    result = {
        "protocol": "sample generation only, one ImageNet image, batch size 1; CUDA synchronized; warmup and checkpoint loading excluded",
        "index": args.index,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hostname": platform.node(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "project_images": args.project_images,
        "project_samples_per_image": args.project_samples,
        "rows": [],
    }
    for name in args.models:
        torch.cuda.reset_peak_memory_stats()
        row = benchmark_one(
            name,
            image,
            label,
            args.warmup,
            args.repeats,
            args.project_images,
            args.project_samples,
        )
        result["rows"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n"
            )
    if args.output:
        print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
