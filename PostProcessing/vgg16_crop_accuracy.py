#!/usr/bin/env python3
"""Compare VGG16 accuracy on original 256x256 inputs and 224x224 center crops.

Retains the calculation from BA-Code-Final/tests/test_vgg16_crop_accuracy.py.
The default dataset is IMAGENET_H5_PATH from run.py.
"""

import argparse
import sys
from pathlib import Path

import h5py
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Normalize

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from Helper.model import load_vgg16  # noqa: E402


def evaluate_vgg16_accuracy(
    h5_path: Path = None,
    batch_size: int = 50,
    device: str = None,
):
    if h5_path is None:
        from run import IMAGENET_H5_PATH

        h5_path = IMAGENET_H5_PATH
    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"ImageNet file not found: {h5_path}. "
            "Set IMAGENET_H5_PATH in run.py or supply --h5-path."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading VGG16 on device: {device}...")
    model = load_vgg16(device)
    model.eval()

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print(f"Reading dataset: {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        inputs = f["inputs"][:]
        labels = f["labels"][:]

    total_samples = len(labels)
    if total_samples == 0:
        raise ValueError("The dataset contains no images.")
    inputs_tensor = torch.from_numpy(inputs).float() / 255.0
    labels_tensor = torch.from_numpy(labels).long()

    correct_256 = 0
    top5_256 = 0
    correct_224 = 0
    top5_224 = 0

    print(f"\nEvaluating {total_samples} samples (batch size {batch_size})...")
    with torch.no_grad():
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            x_batch = inputs_tensor[start_idx:end_idx]
            y_batch = labels_tensor[start_idx:end_idx].to(device)

            # Original 256x256 resolution.
            x_256 = normalize(x_batch).to(device)
            out_256 = model(x_256)
            pred_256 = out_256.argmax(dim=1)
            correct_256 += (pred_256 == y_batch).sum().item()
            _, top5_idx_256 = out_256.topk(5, dim=1)
            top5_256 += (top5_idx_256 == y_batch.unsqueeze(1)).any(dim=1).sum().item()

            # Center crop only: no resize, matching the original comparison.
            x_224_cropped = TF.center_crop(x_batch, [224, 224])
            x_224 = normalize(x_224_cropped).to(device)
            out_224 = model(x_224)
            pred_224 = out_224.argmax(dim=1)
            correct_224 += (pred_224 == y_batch).sum().item()
            _, top5_idx_224 = out_224.topk(5, dim=1)
            top5_224 += (top5_idx_224 == y_batch.unsqueeze(1)).any(dim=1).sum().item()

    acc1_256 = correct_256 / total_samples * 100.0
    acc5_256 = top5_256 / total_samples * 100.0
    acc1_224 = correct_224 / total_samples * 100.0
    acc5_224 = top5_224 / total_samples * 100.0

    print("\n" + "=" * 60)
    print(f"VGG16 Evaluation Results ({total_samples:,} ImageNet Validation Samples)")
    print("=" * 60)
    print("256x256 (Original):")
    print(f"  Top-1 Accuracy: {correct_256}/{total_samples} ({acc1_256:.2f}%)")
    print(f"  Top-5 Accuracy: {top5_256}/{total_samples} ({acc5_256:.2f}%)")
    print("-" * 60)
    print("224x224 (Center Crop [16:240, 16:240]):")
    print(f"  Top-1 Accuracy: {correct_224}/{total_samples} ({acc1_224:.2f}%)")
    print(f"  Top-5 Accuracy: {top5_224}/{total_samples} ({acc5_224:.2f}%)")
    print("=" * 60)

    return {
        "acc1_256": acc1_256,
        "acc5_256": acc5_256,
        "acc1_224": acc1_224,
        "acc5_224": acc5_224,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    evaluate_vgg16_accuracy(args.h5_path, args.batch_size, args.device)


if __name__ == "__main__":
    main()
