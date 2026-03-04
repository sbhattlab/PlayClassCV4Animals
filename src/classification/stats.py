from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def format_confusion_matrix(cm: np.ndarray, labels: list[str]) -> str:
    """Format a confusion matrix as an aligned text table."""
    header = f"{'':>12s}" + "".join(f"{l:>12s}" for l in labels)
    rows = [header]
    for i, row_data in enumerate(cm):
        rows.append(f"{labels[i]:>12s}" + "".join(f"{v:>12d}" for v in row_data))
    return "\n".join(rows)


def aggregate_scalars(fold_results: list[dict], scalar_keys: list[str], run_dir: Path):
    """Build summary CSV with MEAN/STD rows and log per-metric stats."""
    rows = [{k: r[k] for k in scalar_keys} for r in fold_results]
    df = pd.DataFrame(rows)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    mean_row = {c: df[c].mean() for c in numeric_cols}
    std_row = {c: df[c].std() for c in numeric_cols}
    for c in df.columns:
        if c not in numeric_cols:
            mean_row[c] = "MEAN" if c == df.columns[0] else ""
            std_row[c] = "STD" if c == df.columns[0] else ""

    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    summary_path = run_dir / "lovo_summary.csv"
    df.to_csv(summary_path, index=False)
    logger.info(f"LOVO summary saved to {summary_path}")

    for col in numeric_cols:
        values = [r[col] for r in fold_results if col in r]
        logger.info(f"LOVO {col}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


def aggregate_confusion_matrices(
    fold_results: list[dict], cm_keys: list[str], run_dir: Path, labels: list[str]
):
    """Sum and write confusion matrices across folds."""
    for key in cm_keys:
        summed_cm = sum(r[key] for r in fold_results)
        cm_text = format_confusion_matrix(summed_cm, labels)
        cm_path = run_dir / f"lovo_{key}.txt"
        cm_path.write_text(f"# Summed {key} (rows=true, cols=predicted):\n{cm_text}\n")
        logger.info(f"Summed {key} saved to {cm_path}")


def aggregate_metrics(
    fold_results: list[dict], run_dir: Path, labels: list[str]
) -> None:
    """Aggregate per-fold metrics and write summary files.

    Parameters
    ----------
    fold_results : list[dict]
        Per-fold dicts from ``evaluate_fold``.
    run_dir : Path
        Output directory.
    labels : list[str]
        Class names for confusion matrix formatting.
    """
    scalar_keys = []
    cm_keys = []
    for k, v in fold_results[0].items():
        if isinstance(v, (int, float, str)):
            scalar_keys.append(k)
        elif isinstance(v, np.ndarray):
            cm_keys.append(k)
        else:
            raise ValueError(f"Unexpected type {type(v)} for metric '{k}'")

    aggregate_scalars(fold_results, scalar_keys, run_dir)
    aggregate_confusion_matrices(fold_results, cm_keys, run_dir, labels)
