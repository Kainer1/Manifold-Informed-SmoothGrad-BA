"""Original per-image sample model-output Excel format for ImageNet/VGG16."""

import pandas as pd

from Helper.paths import RESULTS_DIR


def save_accuracy(
    group_name: str,
    sampler_name: str,
    accuracy_dict: dict,
    softmax_dict: dict = None,
    target_logit_dict: dict = None,
):
    """Save Accuracy, Softmax percent, and ground-truth TargetLogit image means."""
    metric_values = {
        "Accuracy": accuracy_dict,
        "Softmax": softmax_dict or {},
        "TargetLogit": target_logit_dict or {},
    }
    if not any(metric_values.values()):
        return

    out_dir = RESULTS_DIR / "Stats" / "AnalyzeAttributionAccuracy"
    out_dir.mkdir(parents=True, exist_ok=True)
    excel_path = out_dir / f"{group_name}.xlsx"

    df = (
        pd.read_excel(excel_path)
        if excel_path.exists()
        else pd.DataFrame(columns=["Samplername", "Metrik"])
    )
    df.columns = [str(c) for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

    for metric_name, values in metric_values.items():
        if not values:
            continue

        row_data = {
            "Samplername": sampler_name,
            "Metrik": metric_name,
            **{str(k): v for k, v in values.items()},
        }
        mask = (df["Samplername"] == sampler_name) & (df["Metrik"] == metric_name)

        if mask.any():
            for col, val in row_data.items():
                df.at[df[mask].index[0], col] = val
        else:
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

    idx_cols = sorted([c for c in df.columns if c.isdigit()], key=int)
    df["Mittelwert"] = (
        df[idx_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    )
    df = df[["Samplername", "Metrik", "Mittelwert"] + idx_cols].sort_values(
        by=["Samplername", "Metrik"]
    )
    df.to_excel(excel_path, index=False)
    print(f"  -> Accuracy und Modell-Scores in {excel_path} gesichert.")
