"""Render the tracker-eval LaTeX table from results CSVs.

Mean column = TrackEval COMBINED_SEQ aggregate (from `metrics_aggregate.csv`).
SD column   = per-video standard deviation of that variant's HOTA / IDF1
              (from `metrics_per_video.csv`). For ablation rows, the mean
              is the delta of aggregates (variant - proposed); the SD is
              still the per-video SD of the variant being ablated.

Usage:
    pixi run -e tracker-evaluation python -m src.tracker_eval.create_ablation_table
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .paths import RESULTS_DIR

PROPOSED = "E_sam3_adaptive"

# (label, variant_key, role)
#   role: "baseline" | "proposed" | "ablation"
ROWS = [
    ("YOLO26x + BoT-SORT", "A_yolo_botsort", "baseline"),
    (r"Grounded-SAM-2~\cite{renGroundedSAMAssembling2024}", "B_gs2_strict", "baseline"),
    (r"\quad + adaptive grounding", "B_gs2_fixed", "baseline"),
    (
        r"\makecell[l]{SAM 3 + adaptive grounding \\ \quad \& adaptive chunking}",
        PROPOSED,
        "proposed",
    ),
    (r"\quad $-$ adaptive grounding", "C_sam3_frame_zero", "ablation"),
    (r"\quad $-$ adaptive chunking", "D_sam3_fixed", "ablation"),
]


def fmt_cell(
    mean: float, sd: float, *, bold: bool = False, signed: bool = False
) -> str:
    if signed:
        sign = "$-$" if mean < 0 else ""
        mean_str = f"{sign}{abs(mean):.3f}"
    else:
        mean_str = f"{mean:.3f}"
    if bold:
        mean_str = rf"\textbf{{{mean_str}}}"
    return rf"{mean_str} \scriptsize{{$\pm${sd:.3f}}}"


def build_table(results_dir: Path) -> str:
    agg = pd.read_csv(results_dir / "metrics_aggregate.csv").set_index("variant")
    perv = pd.read_csv(results_dir / "metrics_per_video.csv")
    sd = perv.groupby("variant")[["HOTA", "idf1"]].std()

    e_hota = float(agg.loc[PROPOSED, "HOTA"])
    e_idf1 = float(agg.loc[PROPOSED, "idf1"])

    lines: list[str] = []
    lines += [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{6pt}",
        r"    \caption{Tracking evaluation and ablation (mean HOTA and IDF$_1$ $\pm$ SD "
        r"aggregated over sparse human-verified keyframes from five videos). "
        r"The best score is shown in \textbf{bold}.}",
        r"    \label{tab:tracker_eval}",
        r"    \begin{tabular}{lcc}",
        r"        \toprule",
        r"        \textbf{Tracking method} & \textbf{HOTA} & \textbf{IDF$_1$} \\",
        r"        \midrule",
        r"        \multicolumn{3}{l}{\emph{Baselines:}} \\",
    ]

    for label, key, role in ROWS:
        hota_mean = float(agg.loc[key, "HOTA"])
        idf1_mean = float(agg.loc[key, "idf1"])
        hota_sd = float(sd.loc[key, "HOTA"])
        idf1_sd = float(sd.loc[key, "idf1"])

        if role == "ablation":
            hota_mean -= e_hota
            idf1_mean -= e_idf1
            hota_cell = fmt_cell(hota_mean, hota_sd, signed=True)
            idf1_cell = fmt_cell(idf1_mean, idf1_sd, signed=True)
        elif role == "proposed":
            hota_cell = fmt_cell(hota_mean, hota_sd, bold=True)
            idf1_cell = fmt_cell(idf1_mean, idf1_sd, bold=True)
        else:
            hota_cell = fmt_cell(hota_mean, hota_sd)
            idf1_cell = fmt_cell(idf1_mean, idf1_sd)

        if role == "proposed":
            lines += [
                r"        \midrule",
                r"        \multicolumn{3}{l}{\emph{Proposed method:}} \\",
                f"        {label}",
                f"          & {hota_cell} & {idf1_cell} \\\\",
                r"        \midrule",
                r"        \multicolumn{3}{l}{\emph{Ablations on proposed method:}} \\",
            ]
        else:
            lines.append(f"        {label} & {hota_cell} & {idf1_cell} \\\\")

    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    print(build_table(args.results_dir))


if __name__ == "__main__":
    main()
