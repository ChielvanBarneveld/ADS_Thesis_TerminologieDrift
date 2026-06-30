"""SQ2 (ELAS Heavy / h3): render the combined WSS@95 + Loss heatmap figure.

Heavy-model counterpart of SQ2/make_heatmaps.py. Same two-panel layout
(WSS@95 left, normalized Loss right), same diverging colour scheme centred on
the baseline (0%, 0%) cell. Robust to a single trial per cell: when only one
trial is present the per-cell std is undefined, so the std annotation is
suppressed and the title flags the sweep as preliminary.

Reads:  Report/outputs/SQ2_heavy/grid_runs/trials.jsonl
Writes: Report/outputs/SQ2_heavy/figures/heatmap_grid_heavy.png

Loss formula matches asreview-insights (v1.6.1).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
TRIALS = REPORT_DIR / "outputs" / "SQ2_heavy" / "grid_runs" / "trials.jsonl"
FIG_DIR = REPORT_DIR / "outputs" / "SQ2_heavy" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def render_panel(
    ax,
    pivot_mean: pd.DataFrame,
    pivot_std: pd.DataFrame,
    baseline: float,
    panel_title: str,
    cbar_label: str,
    invert: bool = False,
    show_std: bool = True,
):
    """Render one diverging heatmap panel centred on *baseline*.

    Parameters
    ----------
    invert : bool
        If True the colour sense is reversed (blue = good, red = bad).
        Use for Loss where lower is better.
    show_std : bool
        If False the per-cell std annotation is omitted (single-trial sweep).
    """
    arr_mean = pivot_mean.values
    arr_std = pivot_std.values
    pos_levels = pivot_mean.index.tolist()
    neg_levels = pivot_mean.columns.tolist()

    vmax_dev = max(abs(arr_mean.max() - baseline), abs(arr_mean.min() - baseline))
    cmap = "RdBu" if invert else "RdBu_r"
    norm = mcolors.TwoSlopeNorm(
        vmin=baseline - vmax_dev, vcenter=baseline, vmax=baseline + vmax_dev,
    )

    im = ax.imshow(arr_mean, cmap=cmap, norm=norm, aspect="auto", origin="lower")

    ax.set_xticks(range(len(neg_levels)))
    ax.set_xticklabels([f"{int(v)}%" for v in neg_levels], fontsize=9)
    ax.set_yticks(range(len(pos_levels)))
    ax.set_yticklabels([f"{int(v)}%" for v in pos_levels], fontsize=9)
    ax.set_xlabel("Drift prevalence among negatives", fontsize=11)
    ax.set_ylabel("Drift prevalence among positives", fontsize=11)

    for i in range(arr_mean.shape[0]):
        for j in range(arr_mean.shape[1]):
            m = arr_mean[i, j]
            s = arr_std[i, j]
            bg_intensity = abs(m - baseline) / vmax_dev if vmax_dev > 0 else 0
            txt_color = "white" if bg_intensity > 0.55 else "black"
            y_mean = i if not show_std else i + 0.05
            ax.text(
                j, y_mean, f"{m:.3f}",
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=txt_color,
            )
            if show_std and np.isfinite(s):
                ax.text(
                    j, i - 0.24, f"±{s:.3f}",
                    ha="center", va="center",
                    fontsize=6.5, fontstyle="italic", color=txt_color, alpha=0.7,
                )

    ax.set_title(panel_title, fontsize=11, pad=10)

    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(cbar_label, fontsize=10)


def main():
    lines = [json.loads(line) for line in TRIALS.read_text().splitlines()]
    df = pd.DataFrame(lines)
    if "loss" not in df.columns and "normalized_loss" in df.columns:
        df["loss"] = df["normalized_loss"]
    print(f"loaded {TRIALS.name}: {len(lines)} trials")

    pos_levels = sorted(df["pp"].unique())
    neg_levels = sorted(df["nn"].unique())
    n_trials = int(df.groupby(["pp", "nn"]).size().iloc[0])
    n_total = len(df)
    show_std = n_trials > 1

    agg = df.groupby(["pp", "nn"]).agg(
        wss_mean=("wss_95", "mean"), wss_std=("wss_95", "std"),
        loss_mean=("loss", "mean"), loss_std=("loss", "std"),
    ).reset_index()

    base_wss = float(agg.loc[(agg["pp"] == 0) & (agg["nn"] == 0), "wss_mean"].iloc[0])
    base_loss = float(agg.loc[(agg["pp"] == 0) & (agg["nn"] == 0), "loss_mean"].iloc[0])

    wss_mean_piv = agg.pivot(index="pp", columns="nn", values="wss_mean").reindex(
        index=pos_levels, columns=neg_levels,
    )
    wss_std_piv = agg.pivot(index="pp", columns="nn", values="wss_std").reindex(
        index=pos_levels, columns=neg_levels,
    )
    loss_mean_piv = agg.pivot(index="pp", columns="nn", values="loss_mean").reindex(
        index=pos_levels, columns=neg_levels,
    )
    loss_std_piv = agg.pivot(index="pp", columns="nn", values="loss_std").reindex(
        index=pos_levels, columns=neg_levels,
    )

    fig, (ax_wss, ax_loss) = plt.subplots(1, 2, figsize=(16, 6.5))

    render_panel(
        ax_wss, wss_mean_piv, wss_std_piv, base_wss,
        panel_title=(
            f"WSS@95  ·  baseline ≈ {base_wss:.3f} (grey)\n"
            "blue = degraded  ·  red = shortcut-leak"
        ),
        cbar_label="Mean WSS@95",
        show_std=show_std,
    )

    render_panel(
        ax_loss, loss_mean_piv, loss_std_piv, base_loss,
        panel_title=(
            f"Normalized loss  ·  baseline ≈ {base_loss:.3f} (grey)\n"
            "blue = improved  ·  red = degraded"
        ),
        cbar_label="Mean Loss",
        invert=True,
        show_std=show_std,
    )

    if show_std:
        sub = (
            f"ELAS Heavy (h3): mean WSS@95 (left) and normalized loss (right) "
            f"across {n_trials} trials per cell  (N = {n_total} simulations total)"
            "  ·  small italic = std-dev across trials"
        )
    else:
        sub = (
            "ELAS Heavy (h3): WSS@95 (left) and normalized loss (right) across the "
            f"drift grid  ·  single trial per cell (N = {n_total}, preliminary)"
        )
    fig.suptitle(sub, fontsize=12.5, fontweight="bold", y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = FIG_DIR / "heatmap_grid_heavy.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPORT_DIR)}")


if __name__ == "__main__":
    main()
