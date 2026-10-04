"""Shared sampling loop, model-output statistics, and optional AllSamples storage."""

from abc import ABC, abstractmethod
import gc

import torch

from Storage.SampleStatistics import save_accuracy


class BaseSampler(ABC):
    def __init__(self, *, sampler_name, group_name, n_samples, sampler_config, flags):
        self.sampler_name = sampler_name
        self.group_name = group_name
        self.n_samples = n_samples
        self.sampler_config = sampler_config
        self.flags = flags
        self.resample_all = True
        self.forceSampleSafe = True
        self.geometry_type = "256_projection"
        self.model_name = "VGG16"
        self.dataset_name = "IMAGENET_256"
        self.model_pipeline = None
        self.indices = None
        self.accuracy_dict = {}
        parameters = "".join(
            f"_{key}{value}" for key, value in sorted(sampler_config.items())
        )
        self.name = f"{group_name}_{sampler_name}_n{n_samples}{parameters}"
        self.storage_config = dict(
            sampler_config,
            group_name=group_name,
            n_samples=n_samples,
            geometry_type=self.geometry_type,
            sampler_class_name=self.__class__.__name__,
        )
        self.config = self.storage_config

    def setup(self, inputs, labels=None):
        """Concrete samplers prepare models or input-dependent masks here."""

    @abstractmethod
    def compute_sample(self, inputs, labels=None):
        """Return one normalized sample batch with the input batch's shape."""
        raise NotImplementedError

    def cleanup(self):
        """Retain the original cleanup of sampler-owned models and temporary data."""
        for attr in ["model_pipeline", "model", "vae", "vae_model"]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                obj = getattr(self, attr)
                if hasattr(obj, "to"):
                    try:
                        obj.to("cpu")
                    except Exception:
                        pass
                if hasattr(obj, "model") and hasattr(obj.model, "to"):
                    try:
                        obj.model.to("cpu")
                    except Exception:
                        pass
                delattr(self, attr)
        if hasattr(self, "latents") and self.latents is not None:
            del self.latents
            self.latents = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self, inputs, labels, *, indices, evaluation_model, all_sample_writer=None
    ):
        if not (len(inputs) == len(labels) == len(indices)):
            raise ValueError("Images, labels, and indices must have matching counts.")
        batch_inputs = torch.stack(inputs) if isinstance(inputs, list) else inputs
        batch_inputs = batch_inputs.to(labels.device)
        self.indices = indices
        self.accuracy_dict.clear()
        correct_counts = torch.zeros(len(indices), device=labels.device)
        target_logit_sums = torch.zeros(len(indices), device=labels.device)
        target_softmax_sums = torch.zeros(len(indices), device=labels.device)
        evaluation_model.eval()
        try:
            self.setup(batch_inputs, labels=labels)
            if all_sample_writer is not None:
                all_sample_writer.begin_batch(indices, labels)
            for _ in range(self.n_samples):
                sample = self.compute_sample(batch_inputs, labels=labels)
                # The fixed 256_projection output geometry is the identity.
                with torch.no_grad():
                    logits = evaluation_model(sample)
                    predictions = torch.argmax(logits, dim=1)
                    correct_counts += (predictions == labels).float()
                target_logit_sums += logits.gather(1, labels.unsqueeze(1)).squeeze(1)
                target_softmax_sums += (
                    torch.softmax(logits, dim=1)
                    .gather(1, labels.unsqueeze(1))
                    .squeeze(1)
                    * 100.0
                )
                yield sample
                if all_sample_writer is not None:
                    all_sample_writer.record_batch(sample)
            if all_sample_writer is not None:
                all_sample_writer.finish_batch()

            accuracies = (correct_counts / self.n_samples * 100.0).cpu().tolist()
            target_logits = (target_logit_sums / self.n_samples).cpu().tolist()
            softmax_values = (target_softmax_sums / self.n_samples).cpu().tolist()
            self.accuracy_dict.update(zip(indices, accuracies))
            save_accuracy(
                group_name=self.group_name,
                sampler_name=self.name,
                accuracy_dict=self.accuracy_dict,
                softmax_dict=dict(zip(indices, softmax_values)),
                target_logit_dict=dict(zip(indices, target_logits)),
            )
        finally:
            self.cleanup()
