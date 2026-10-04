"""The original all-sample HDF5 writer, restricted to ImageNet/VGG16."""

from pathlib import Path

import h5py
import numpy as np
import torch

from Helper.paths import RESULTS_DIR


DATASET_NAME = "samples"
PENDING_GROUP = "_pending"


class AllSampleWriter:
    """Incrementally store final sampler outputs as float32 model inputs."""

    def __init__(
        self,
        sampler_name: str,
        sampler_config: dict,
        expected_samples: int,
        file_path: Path | None = None,
    ):
        self.file_path = (
            Path(file_path)
            if file_path
            else RESULTS_DIR / "Samples" / "allSamples" / f"samples_{sampler_name}.h5"
        )
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.h5_file = h5py.File(self.file_path, "a")
        self.h5_file.attrs["sampler_name"] = sampler_name
        self.h5_file.attrs["group_name"] = sampler_config.get("group_name", "unknown")
        self.h5_file.attrs["model_name"] = "VGG16"
        self.h5_file.attrs["dataset_name"] = "IMAGENET_256"
        self.h5_file.attrs["representation"] = "normalized model input"
        self.h5_file.attrs["dtype"] = "float32"
        self.h5_file.attrs["sampler_config"] = str(sampler_config)
        self.expected_samples = int(expected_samples)
        self.pending = self.h5_file.require_group(PENDING_GROUP)
        self.active_indices: list[str] = []
        self.sample_position = 0

    def begin_batch(self, indices: list, labels: torch.Tensor) -> None:
        if self.active_indices:
            raise RuntimeError("An all-sample batch is already active.")
        if len(indices) != labels.numel():
            raise ValueError(
                f"Received {labels.numel()} labels for {len(indices)} indices."
            )

        self.active_indices = [str(index) for index in indices]
        self.sample_position = 0
        label_values = labels.detach().cpu().flatten().tolist()
        for position, index in enumerate(self.active_indices):
            if index in self.pending:
                del self.pending[index]
            group = self.pending.create_group(index)
            group.attrs["label"] = int(label_values[position])
            group.attrs["complete"] = False

    def record_batch(self, samples: torch.Tensor) -> None:
        if not self.active_indices:
            raise RuntimeError("begin_batch must be called before record_batch.")
        if samples.ndim != 4:
            raise ValueError(
                "Samples must have shape [batch, channels, height, width], "
                f"but received {tuple(samples.shape)}."
            )
        if samples.shape[0] != len(self.active_indices):
            raise ValueError(
                f"Received {samples.shape[0]} samples for "
                f"{len(self.active_indices)} indices."
            )
        if self.sample_position >= self.expected_samples:
            raise ValueError(
                "The sampler produced more accepted samples than configured: "
                f"expected {self.expected_samples}."
            )

        sample_values = samples.detach().float().cpu().numpy()
        channels, height, width = sample_values.shape[-3:]
        for batch_position, index in enumerate(self.active_indices):
            group = self.pending[index]
            if DATASET_NAME not in group:
                group.create_dataset(
                    DATASET_NAME,
                    shape=(self.expected_samples, channels, height, width),
                    maxshape=(None, channels, height, width),
                    chunks=(1, channels, height, width),
                    dtype=np.float32,
                )
            group[DATASET_NAME][self.sample_position] = sample_values[batch_position]
        self.sample_position += 1

    def finish_batch(self) -> None:
        if not self.active_indices:
            return
        if self.sample_position == 0:
            raise ValueError("Cannot save an empty all-sample batch.")

        for index in self.active_indices:
            group = self.pending[index]
            dataset = group[DATASET_NAME]
            if dataset.shape[0] != self.sample_position:
                dataset.resize((self.sample_position, *dataset.shape[1:]))
            group.attrs["complete"] = True
            group.attrs["sample_count"] = self.sample_position
            if index in self.h5_file:
                del self.h5_file[index]
            self.h5_file.move(f"{PENDING_GROUP}/{index}", index)

        self.h5_file.flush()
        self.active_indices = []
        self.sample_position = 0

    def close(self) -> None:
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
