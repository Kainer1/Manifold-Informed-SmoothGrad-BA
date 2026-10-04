"""Original SRG evaluation and per-group CSV output, without sampler rebuilding."""

import ast
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import quantus
import torch

from Sampling.AttributionComputation import pool
from Visualisations.heatmaps import get_samplers_for_group
from Helper.paths import RESULTS_DIR


def run_evaluation(group_name, model, original_inputs, labels, indices, device):
    """Recompute SRG for the group's stored attributions on the selected images."""
    if not indices:
        return
    samplers = get_samplers_for_group(group_name)
    if not samplers:
        print(f"No attribution samplers found for group '{group_name}'.")
        return

    x_batch = torch.stack(original_inputs).numpy()
    y_batch = labels.cpu().numpy()
    rows = []
    for sampler_name in samplers:
        path = RESULTS_DIR / "Attributions" / f"attributions_{sampler_name}.h5"
        with h5py.File(path, "r") as file:
            sampler_config = {}
            for key in file:
                config_string = file[key].attrs.get("sampler_config")
                if config_string:
                    sampler_config = ast.literal_eval(config_string)
                    break
            attributions = []
            valid_positions = []
            for position, index in enumerate(indices):
                if str(index) in file:
                    attributions.append(file[str(index)]["attributions"][:])
                    valid_positions.append(position)
        if not attributions:
            print(f"[{sampler_name}] No attributions for the selected indices.")
            continue

        a_batch = np.array(attributions)
        if a_batch.ndim == 3:
            a_batch = a_batch[:, np.newaxis, :, :]
        a_batch = pool(a_batch) if a_batch.shape[1] > 1 else a_batch
        if a_batch.ndim == 3:
            a_batch = a_batch[:, np.newaxis, :, :]
        x_batch_curr = x_batch[valid_positions]
        y_batch_curr = y_batch[valid_positions]
        metric = quantus.SymmetricRelevanceGain(features_in_step=256, abs=True)
        row = {"Samplername": sampler_name, "Gruppenname": group_name}
        try:
            scores = []
            for start in range(0, len(valid_positions), 10):
                end = min(start + 10, len(valid_positions))
                scores.extend(
                    metric(
                        model=model,
                        x_batch=x_batch_curr[start:end],
                        y_batch=y_batch_curr[start:end],
                        a_batch=a_batch[start:end],
                        device=device,
                    )
                )
                print(
                    f"[{sampler_name}] SRG: {end}/{len(valid_positions)} images evaluated.",
                    flush=True,
                )
            row["SymmetricRelevanceGain_mean"] = np.nanmean(scores)
            row["SymmetricRelevanceGain_std"] = np.nanstd(scores)
        except Exception as error:
            print(f"[{sampler_name}] [FAILED] SymmetricRelevanceGain: {error}")
            row["SymmetricRelevanceGain_mean"] = np.nan
            row["SymmetricRelevanceGain_std"] = np.nan

        excluded_keys = {
            "group_name",
            "save_visualization_samples",
            "resample_all",
            "forceSampleSafe",
        }
        for key, value in sampler_config.items():
            if key not in excluded_keys and key not in row:
                row[key] = value
        rows.append(row)

    stats_dir = RESULTS_DIR / "Stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_results(group_name, rows, stats_dir)


def save_evaluation_results(group: str, rows: list, stats_dir: Path):
    """
    Speichert die Liste von Evaluierungs-Ergebnissen als CSV.
    """
    if not rows:
        return

    csv_path = stats_dir / f"{group}.csv"
    existing_df = pd.DataFrame()

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)

    new_df = pd.DataFrame(rows)

    if not existing_df.empty:
        existing_df = existing_df[
            ~existing_df["Samplername"].isin(new_df["Samplername"])
        ]
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    cols = list(final_df.columns)
    first_cols = ["Samplername", "Gruppenname"]
    metric_cols = sorted([c for c in cols if "_mean" in c or "_std" in c])
    config_cols = [c for c in cols if c not in first_cols + metric_cols]

    prioritized_config = []
    for p in ["geometry_type", "n_samples"]:
        if p in config_cols:
            prioritized_config.append(p)
            config_cols.remove(p)

    config_cols = sorted(config_cols)

    final_cols = first_cols + metric_cols + prioritized_config + config_cols
    final_df = final_df[final_cols]

    final_df.to_csv(csv_path, index=False)
    print(f"  -> {len(rows)} Sampler in {csv_path} gespeichert/aktualisiert.")
