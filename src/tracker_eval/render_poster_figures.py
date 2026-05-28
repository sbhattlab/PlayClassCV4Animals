"""Poster-ready visualisations for the tracker evaluation and classifier.

Outputs (PNG @ 300 dpi + PDF vector) under ``img/``:
  - tracker_eval_aggregate.{png,pdf}          — grouped-bar of HOTA/DetA/AssA/IDF1 x variants
  - tracker_eval_aggregate_strip.{png,pdf}    — same with per-video dots overlaid
  - tracker_eval_per_cage_hota.{png,pdf}      — HOTA per cage x variants
  - tracker_eval_hota_decomposition.{png,pdf} — DetA vs AssA scatter (poster scale)
  - tracker_eval_robustness.{png,pdf}         — per-video HOTA across the 3 SAM 3 variants
  - tracker_eval_summary.{png,pdf}            — single 2x2 poster panel combining the above
  - confusion_matrix_poster.{png,pdf}         — row-normalised confusion matrix with marginal metrics

Usage:
    pixi run -e tracker-evaluation python -m src.tracker_eval.render_poster_figures

ID-switch counts are deliberately omitted: gs2 has very few switches but
worse tracking (HOTA, DetA), because mask-propagation failures register as
misses and persistent false-positives, not as re-assignments. Showing raw
switch counts invites the misleading "fewer is better" reading.

Variant palette is family-grouped: gray for A (YOLO), two greens for the
gs2 pair (strict / parity), two blues for SAM 3 strict + fixed, and orange
for the full method E. This groups same-family variants visually so the
within-family steps read as recovery-mechanism ablations and the between-
family steps read as model-backbone swaps.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import RESULTS_DIR, ROOT

IMG_DIR = ROOT / "img"

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------

VARIANT_ORDER = [
    "A_yolo_botsort",
    "B_gs2_strict",
    "B_gs2_fixed",
    "C_sam3_frame_zero",
    "D_sam3_fixed",
    "E_sam3_adaptive",
]
VARIANT_LABELS = {
    "A_yolo_botsort": "A: YOLO + BoT-SORT",
    "B_gs2_strict": "B-strict: Grounded-SAM-2 (no recovery)",
    "B_gs2_fixed": "B-parity: Grounded-SAM-2 (parity recovery)",
    "C_sam3_frame_zero": "C-strict: SAM 3 (frame-0, no fallback)",
    "D_sam3_fixed": "D: SAM 3 + adaptive grounding",
    "E_sam3_adaptive": "E: SAM 3 + adaptive grounding + chunking (proposed)",
}
# Shortened labels used in the poster-scale HOTA decomposition legend so the
# single-column legend stays compact.
VARIANT_LABELS_SHORT = {
    "A_yolo_botsort": "A: YOLO + BoT-SORT",
    "B_gs2_strict": "B-s: Grounded-SAM-2 (no recovery)",
    "B_gs2_fixed": "B-p: Grounded-SAM-2 (parity recovery)",
    "C_sam3_frame_zero": "C: SAM 3 (frame-0, no fallback)",
    "D_sam3_fixed": "D: SAM 3 + adaptive grounding",
    "E_sam3_adaptive": "E: SAM 3 + adapt. grounding + chunking",
}
# Family-grouped palette: gray YOLO, two greens for gs2 family, two blues for
# SAM 3 strict/fixed, orange for the full method.
VARIANT_COLORS = {
    "A_yolo_botsort": "#9A9A9A",  # gray
    "B_gs2_strict": "#B5D69E",  # pale green — gs2 no recovery
    "B_gs2_fixed": "#5E8B3C",  # green     — gs2 with parity recovery
    "C_sam3_frame_zero": "#A8C5E3",  # pale blue — SAM 3 strict
    "D_sam3_fixed": "#2E75B6",  # blue      — SAM 3 adaptive grounding
    "E_sam3_adaptive": "#ED7D31",  # orange    — full method
}
LETTER_MAP = {
    "A_yolo_botsort": "A",
    "B_gs2_strict": "B-s",
    "B_gs2_fixed": "B-p",
    "C_sam3_frame_zero": "C",
    "D_sam3_fixed": "D",
    "E_sam3_adaptive": "E",
}

BASE_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
}

# Poster-scale overrides used only by the HOTA-decomposition scatter. Scoped
# via plt.rc_context so other figures keep BASE_RC.
POSTER_RC = {
    **BASE_RC,
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
}

plt.rcParams.update(BASE_RC)


def _save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(IMG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def _annotate_bars(ax, bars, fmt="{:.3f}", offset=0.005, fontsize=8):
    for b in bars:
        h = b.get_height()
        if h >= 0:
            y = h + offset
            va = "bottom"
        else:
            y = h - offset
            va = "top"
        ax.text(
            b.get_x() + b.get_width() / 2,
            y,
            fmt.format(h),
            ha="center",
            va=va,
            fontsize=fontsize,
            color="#333",
        )


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------

agg = (
    pd
    .read_csv(RESULTS_DIR / "metrics_aggregate.csv")
    .set_index("variant")
    .loc[VARIANT_ORDER]
)
per_video = pd.read_csv(RESULTS_DIR / "metrics_per_video.csv")
per_cage = pd.read_csv(RESULTS_DIR / "metrics_per_cage.csv")

N_VARIANTS = len(VARIANT_ORDER)
BAR_WIDTH = 0.13  # 6 variants × 0.13 = 0.78, leaves 0.22 gap between groups


def _offset(i: int) -> float:
    """Within-group x-offset for the i-th variant out of N_VARIANTS."""
    return (i - (N_VARIANTS - 1) / 2) * BAR_WIDTH


# ----------------------------------------------------------------------------
# Figure 1 — aggregate grouped bar (HOTA, DetA, AssA, IDF1)
# ----------------------------------------------------------------------------


def fig_aggregate(ax=None, show_legend=True):
    metrics = ["HOTA", "DetA", "AssA", "idf1"]
    display = ["HOTA", "DetA", "AssA", "IDF1"]
    x = np.arange(len(metrics))

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 6))

    for i, v in enumerate(VARIANT_ORDER):
        vals = agg.loc[v, metrics].astype(float).values
        bars = ax.bar(
            x + _offset(i),
            vals,
            BAR_WIDTH,
            label=VARIANT_LABELS[v],
            color=VARIANT_COLORS[v],
            edgecolor="white",
            linewidth=0.5,
        )
        _annotate_bars(ax, bars, fontsize=7)

    ax.set_ylim(0, 0.85)
    ax.axhline(0, color="#888", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(display)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title(
        "Aggregate tracker metrics — 6-way ablation on 5 occlusion-stressed videos (sparse-keyframe GT)"
    )
    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            frameon=False,
            ncol=3,
            fontsize=9.5,
        )

    if standalone:
        fig.tight_layout()
        _save(fig, "tracker_eval_aggregate")
        plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 1b — aggregate grouped bar WITH per-video strip overlay
# ----------------------------------------------------------------------------


def _darken(hex_color: str, amount: float = 0.45) -> str:
    h = hex_color.lstrip("#")
    r, g, b = [int(h[i : i + 2], 16) for i in (0, 2, 4)]
    r, g, b = [max(0, int(c * (1 - amount))) for c in (r, g, b)]
    return f"#{r:02x}{g:02x}{b:02x}"


def fig_aggregate_strip(ax=None):
    metrics = ["HOTA", "DetA", "AssA", "idf1"]
    display = ["HOTA", "DetA", "AssA", "IDF1"]
    x = np.arange(len(metrics))

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13.5, 6))

    per_video_std = per_video.groupby("variant")[metrics].std().loc[VARIANT_ORDER]
    for i, v in enumerate(VARIANT_ORDER):
        vals = agg.loc[v, metrics].astype(float).values
        stds = per_video_std.loc[v, metrics].astype(float).values
        bar_x = x + _offset(i)
        ax.bar(
            bar_x,
            vals,
            BAR_WIDTH,
            label=VARIANT_LABELS[v],
            color=VARIANT_COLORS[v],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.55,
            zorder=2,
        )
        ax.errorbar(
            bar_x,
            vals,
            yerr=stds,
            fmt="none",
            ecolor="#222",
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.9,
            zorder=3,
        )

    videos_sorted = sorted(per_video["video_id"].unique())
    jitter_range = BAR_WIDTH * 0.55
    jitter = np.linspace(-jitter_range / 2, jitter_range / 2, len(videos_sorted))
    jitter_map = dict(zip(videos_sorted, jitter))

    for i, v in enumerate(VARIANT_ORDER):
        dot_color = _darken(VARIANT_COLORS[v], 0.40)
        sub = per_video[per_video["variant"] == v]
        for j, metric in enumerate(metrics):
            for _, row in sub.iterrows():
                vid = row["video_id"]
                xj = x[j] + _offset(i) + jitter_map[vid]
                ax.scatter(
                    xj,
                    row[metric],
                    s=18,
                    color=dot_color,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=4,
                )

    ax.set_ylim(0, 0.85)
    ax.axhline(0, color="#888", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(display)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title(
        "Aggregate tracker metrics with per-video distribution  (5 videos × 6 variants)"
    )

    bar_handles = [
        plt.Rectangle((0, 0), 1, 1, color=VARIANT_COLORS[v], alpha=0.55, ec="white")
        for v in VARIANT_ORDER
    ]
    bar_labels = [VARIANT_LABELS[v] for v in VARIANT_ORDER]
    ax.legend(
        handles=bar_handles,
        labels=bar_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )

    if standalone:
        fig.tight_layout()
        _save(fig, "tracker_eval_aggregate_strip")
        plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2 — per-cage HOTA grouped bar
# ----------------------------------------------------------------------------


def fig_per_cage_hota(ax=None, show_legend=True):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 5.5))

    cages = sorted(per_cage["cage"].unique())
    x = np.arange(len(cages))

    for i, v in enumerate(VARIANT_ORDER):
        sub = per_cage[per_cage["variant"] == v].set_index("cage").loc[cages]
        bars = ax.bar(
            x + _offset(i),
            sub["HOTA"].values,
            BAR_WIDTH,
            label=VARIANT_LABELS[v],
            color=VARIANT_COLORS[v],
            edgecolor="white",
            linewidth=0.5,
        )
        _annotate_bars(ax, bars, fmt="{:.2f}", offset=0.008, fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cages)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("Cage")
    ax.set_ylabel("HOTA")
    ax.set_title("HOTA per cage  —  cross-environment consistency")
    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            frameon=False,
            ncol=3,
            fontsize=9.5,
        )

    if standalone:
        fig.tight_layout()
        _save(fig, "tracker_eval_per_cage_hota")
        plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3 — DetA vs AssA scatter (HOTA decomposition story)
# ----------------------------------------------------------------------------


def _draw_hota_decomposition(ax, *, poster_scale: bool):
    """Shared body for the HOTA-decomposition scatter.

    ``poster_scale=True`` is the standalone (poster) variant: large markers,
    bolder annotations, single-column bottom legend with shortened labels.
    ``poster_scale=False`` keeps it compact for the 2x2 summary panel.
    """
    marker_size = 500 if poster_scale else 240
    edge_lw = 2 if poster_scale else 1.5
    letter_fs = 20 if poster_scale else 12
    iso_lw = 1.2 if poster_scale else 0.8
    iso_label_fs = 14 if poster_scale else 8
    label_offset = (10, 10) if poster_scale else (8, 8)

    for v in VARIANT_ORDER:
        ax.scatter(
            agg.loc[v, "DetA"],
            agg.loc[v, "AssA"],
            s=marker_size,
            color=VARIANT_COLORS[v],
            edgecolor="white",
            linewidth=edge_lw,
            label=VARIANT_LABELS_SHORT[v] if poster_scale else VARIANT_LABELS[v],
            zorder=3,
        )
        ax.annotate(
            LETTER_MAP[v],
            xy=(agg.loc[v, "DetA"], agg.loc[v, "AssA"]),
            xytext=label_offset,
            textcoords="offset points",
            fontsize=letter_fs,
            fontweight="bold",
            color=VARIANT_COLORS[v],
        )

    # Iso-HOTA contours (HOTA ≈ sqrt(DetA · AssA))
    xs = np.linspace(0.01, 0.85, 200)
    for hota_iso in (0.1, 0.3, 0.5, 0.7):
        ys = (hota_iso**2) / xs
        ax.plot(xs, ys, color="#bbb", lw=iso_lw, ls="--", zorder=1)
        x_lab = 0.83
        y_lab = (hota_iso**2) / x_lab
        if 0.02 < y_lab < 0.83:
            ax.text(
                x_lab,
                y_lab,
                f"HOTA={hota_iso}",
                fontsize=iso_label_fs,
                color="#888",
                va="bottom",
            )

    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("DetA  (detection quality)")
    ax.set_ylabel("AssA  (association quality)")


def fig_deta_vs_assa(ax=None):
    """Compact HOTA-decomposition scatter for use inside the 2x2 summary."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6.0, 7.0))

    _draw_hota_decomposition(ax, poster_scale=False)
    ax.set_title("Tracking HOTA performance decomposition")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        frameon=False,
        fontsize=9.5,
        ncol=1,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    if standalone:
        fig.tight_layout()
        _save(fig, "tracker_eval_hota_decomposition")
        plt.close(fig)


def fig_hota_decomposition_poster():
    """Poster-scale HOTA-decomposition scatter (overrides fig_deta_vs_assa output)."""
    with plt.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(10, 10))
        _draw_hota_decomposition(ax, poster_scale=True)
        ax.set_title(
            "Tracking HOTA performance\ndecomposition", fontweight="bold", pad=12
        )

        handles = [
            mlines.Line2D(
                [],
                [],
                marker="o",
                color=VARIANT_COLORS[v],
                markersize=14,
                linestyle="None",
                markeredgecolor="white",
                markeredgewidth=1.2,
            )
            for v in VARIANT_ORDER
        ]
        labels = [VARIANT_LABELS_SHORT[v] for v in VARIANT_ORDER]
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=1,
            frameon=False,
            fontsize=16,
            handletextpad=0.8,
            labelspacing=0.5,
        )

        fig.tight_layout()
        _save(fig, "tracker_eval_hota_decomposition")
        plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 4 — per-video HOTA trajectory across the 3 SAM 3 variants
# ----------------------------------------------------------------------------


def fig_per_video_robustness(ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6.2))

    variants_three = ["C_sam3_frame_zero", "D_sam3_fixed", "E_sam3_adaptive"]
    x_pos = {v: i for i, v in enumerate(variants_three)}

    pv = per_video.copy()
    sub = pv[pv["variant"].isin(variants_three)][["variant", "video_id", "HOTA"]]
    wide = sub.pivot(index="video_id", columns="variant", values="HOTA")
    wide = wide[variants_three]

    line_color = "#888"

    for vid, row in wide.iterrows():
        xs = [x_pos[v] for v in variants_three]
        ys = [row[v] for v in variants_three]
        ax.plot(xs, ys, color=line_color, lw=1.2, alpha=0.55, zorder=2)
        for v in variants_three:
            ax.scatter(
                x_pos[v],
                row[v],
                s=80,
                color=VARIANT_COLORS[v],
                edgecolor="white",
                linewidth=1.3,
                zorder=4,
            )

    label_x_anchor = x_pos["E_sam3_adaptive"] + 0.10
    label_x_text = x_pos["E_sam3_adaptive"] + 0.16
    min_gap = 0.028

    labelled = (
        wide
        .assign(vid=wide.index)
        .sort_values(by="E_sam3_adaptive", ascending=False)
        .reset_index(drop=True)
    )
    label_ys = labelled["E_sam3_adaptive"].astype(float).tolist()
    for i in range(1, len(label_ys)):
        if label_ys[i - 1] - label_ys[i] < min_gap:
            label_ys[i] = label_ys[i - 1] - min_gap

    for i, r in labelled.iterrows():
        vid = r["vid"]
        y_dot = r["E_sam3_adaptive"]
        y_lab = label_ys[i]
        if abs(y_dot - y_lab) > 1e-4:
            ax.plot(
                [x_pos["E_sam3_adaptive"] + 0.025, label_x_anchor],
                [y_dot, y_lab],
                color="#444",
                lw=0.7,
                alpha=0.7,
                zorder=3,
            )
        ax.text(
            label_x_text,
            y_lab,
            vid.replace("_day_", " · day "),
            fontsize=10,
            va="center",
            color="#444",
        )

    means = wide.mean()
    for v in variants_three:
        ax.hlines(
            means[v],
            x_pos[v] - 0.20,
            x_pos[v] + 0.20,
            colors="#222",
            linestyles="--",
            lw=1.2,
            zorder=3,
        )
        ax.text(
            x_pos[v] - 0.22,
            means[v],
            f"{means[v]:.3f}",
            ha="right",
            va="center",
            fontsize=9,
            color="#222",
        )

    stds = wide.std()
    for v in variants_three:
        ax.text(
            x_pos[v],
            0.04,
            f"std = {stds[v]:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#222",
            bbox=dict(boxstyle="round,pad=0.32", fc="#f3f3f3", ec="#ccc", lw=0.6),
        )

    ax.set_xlim(-0.55, 2.85)
    ax.set_ylim(0, 0.85)
    ax.set_xticks([x_pos[v] for v in variants_three])
    ax.set_xticklabels([
        "C-strict\n(no scan, no fallback)",
        "D\n(+ adaptive grounding)",
        "E\n(+ adaptive chunking)",
    ])
    ax.set_ylabel("HOTA")
    ax.set_title(
        "Per-video HOTA — SAM 3 variants  (adaptive grounding drives the big jump)",
        fontsize=12.5,
    )
    ax.grid(True, axis="y", alpha=0.25)

    if standalone:
        fig.tight_layout()
        _save(fig, "tracker_eval_robustness")
        plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 5 — single 2x2 poster panel
# ----------------------------------------------------------------------------


def fig_summary():
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        2, 2, hspace=0.46, wspace=0.22, top=0.83, bottom=0.07, left=0.06, right=0.98
    )

    ax1 = fig.add_subplot(gs[0, :])
    fig_aggregate(ax=ax1, show_legend=False)

    ax2 = fig.add_subplot(gs[1, 0])
    fig_per_cage_hota(ax=ax2, show_legend=False)

    ax3 = fig.add_subplot(gs[1, 1])
    fig_deta_vs_assa(ax=ax3)
    leg = ax3.get_legend()
    if leg is not None:
        leg.remove()

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=VARIANT_COLORS[v], ec="white")
        for v in VARIANT_ORDER
    ]
    labels = [VARIANT_LABELS[v] for v in VARIANT_ORDER]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    fig.suptitle(
        "Tracker evaluation — 6-way ablation on 5 occlusion-stressed videos (sparse-keyframe GT)",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    _save(fig, "tracker_eval_summary")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 6 — row-normalised confusion matrix with marginal metrics
# ----------------------------------------------------------------------------

# Values from Table 4 (3-class, social excluded).
CM_LABELS = ["Other", "Obj.", "Loc."]
CM_PCT = np.array([
    [93.5, 4.8, 1.7],
    [31.2, 66.1, 2.7],
    [8.5, 6.7, 84.8],
])
CM_F1 = [94.8, 61.9, 74.4]
CM_PRECISIONS = [96.2, 58.2, 66.2]
CM_SUPPORTS = [12_585, 1_345, 585]


def fig_confusion_matrix():
    fig, ax = plt.subplots(figsize=(6, 5.2), dpi=200)

    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0, vmax=100)
    im = ax.imshow(CM_PCT, cmap=cmap, norm=norm, aspect="equal")

    n = len(CM_LABELS)
    for i in range(n):
        for j in range(n):
            val = CM_PCT[i, j]
            color = "white" if val > 55 else "black"
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(range(n))
    ax.set_xticklabels(CM_LABELS, fontsize=13)
    ax.set_yticks(range(n))
    ax.set_yticklabels(CM_LABELS, fontsize=13)
    ax.set_xlabel("Predicted", fontsize=14, labelpad=8)
    ax.set_ylabel("True", fontsize=14, labelpad=8)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    margin = 0.15
    for i in range(n):
        ax.text(
            n - 0.5 + margin,
            i,
            f"F₁ {CM_F1[i]:.1f}",
            ha="left",
            va="center",
            fontsize=11,
            color="#333",
        )
        ax.text(
            n - 0.5 + margin,
            i + 0.30,
            f"n={CM_SUPPORTS[i]:,}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#888",
        )
        ax.text(
            i,
            n - 0.5 + margin,
            f"P {CM_PRECISIONS[i]:.1f}",
            ha="center",
            va="top",
            fontsize=11,
            color="#333",
        )

    ax.set_xlim(-0.5, n - 0.5 + 1.35)
    ax.set_ylim(n - 0.5 + 0.55, -0.5)

    ax.set_xticks([x - 0.5 for x in range(1, n)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n)], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, shrink=0.82)
    cbar.set_label("Row-normalised (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    _save(fig, "confusion_matrix_poster")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig_aggregate()
    fig_aggregate_strip()
    fig_per_cage_hota()
    fig_per_video_robustness()
    fig_summary()
    # Poster-scale variant runs last so it overrides the compact HOTA
    # decomposition output written by fig_summary().
    fig_hota_decomposition_poster()
    fig_confusion_matrix()
    print("Wrote:")
    for p in sorted(IMG_DIR.glob("tracker_eval_*.png")):
        print(f"  {p.relative_to(ROOT)}")
    for p in sorted(IMG_DIR.glob("tracker_eval_*.pdf")):
        print(f"  {p.relative_to(ROOT)}")
    for p in sorted(IMG_DIR.glob("confusion_matrix_poster.*")):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
