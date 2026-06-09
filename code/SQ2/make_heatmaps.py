"""SQ2: render WSS@95 and Loss heatmaps from the grid sweep.

Reads:  Report/outputs/SQ2/grid_runs/trials.jsonl
Writes: Report/outputs/SQ2/figures/heatmap_wss95.png
        Report/outputs/SQ2/figures/heatmap_loss.png

Both heatmaps share the same 6x6 axes:
    x = nn (% of negatives rewritten)  — "Drift prevalence among negatives"
    y = pp (% of positives rewritten)  — "Drift prevalence among positives"

Diverging colour scheme centred on the baseline (0%, 0%) cell:
    blue = degraded vs baseline, grey = at baseline, red = improved / shortcut-leak.
Cell annotations: bold mean ± italic std across trials.

Loss formula matches asreview-insights (v1.6.1).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
TRIALS = REPORT_DIR / "outputs" / "SQ2" / "grid_runs" / "trials.jsonl"
FIG_DIR = REPORT_DIR / "outputs" / "SQ2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def render_heatmap(
    pivot_mean: pd.DataFrame,
    pivot_std: pd.DataFrame,
    baseline: float,
    title: str,
    subtitle: str,
    cbar_label: str,
    out_path: Path,
    invert: bool = False,
):
    """Render a single diverging heatmap centred on *baseline*.

    Parameters
    ----------
    invert : bool
        If True the colour sense is reversed (blue = good, red = bad).
        Use for Loss where lower is better.
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

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(arr_mean, cmap=cmap, norm=norm, aspect="auto", origin="lower")

    ax.set_xticks(range(len(neg_levels)))
    ax.set_xticklabels([f"{int(v)}%" for v in neg_levels], fontsize=10)
    ax.set_yticks(range(len(pos_levels)))
    ax.set_yticklabels([f"{int(v)}%" for v in pos_levels], fontsize=10)
    ax.set_xlabel("Drift prevalence among negatives", fontsize=12)
    ax.set_ylabel("Drift prevalence among positives", fontsize=12)

    # Cell annotations: bold mean + small italic std
    for i in range(arr_mean.shape[0]):
        for j in range(arr_mean.shape[1]):
            m = arr_mean[i, j]
            s = arr_std[i, j]
            bg_intensity = abs(m - baseline) / vmax_dev if vmax_dev > 0 else 0
            txt_color = "white" if bg_intensity > 0.55 else "black"
            ax.text(
                j, i + 0.05, f"{m:.3f}",
                ha="center", va="center",
                fontsize=10, fontweight="bold", color=txt_color,
            )
            ax.text(
                j, i - 0.22, f"±{s:.3f}",
                ha="center", va="center",
                fontsize=7, fontstyle="italic", color=txt_color, alpha=0.7,
            )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    ax.set_title(subtitle, fontsize=9, color="grey", pad=12)

    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(cbar_label, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPORT_DIR)}")


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

    agg = df.groupby(["pp", "nn"]).agg(
        wss_mean=("wss_95", "mean"), wss_std=("wss_95", "std"),
        loss_mean=("loss", "mean"), loss_std=("loss", "std"),
    ).reset_index()

    base_wss = float(agg.loc[(agg["pp"] == 0) & (agg["nn"] == 0), "wss_mean"].iloc[0])
    base_loss = float(agg.loc[(agg["pp"] == 0) & (agg["nn"] == 0), "loss_mean"].iloc[0])

    # Pivot tables (rows = pp, cols = nn)
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

    render_heatmap(
        wss_mean_piv, wss_std_piv, base_wss,
        title=f"Mean WSS@95 across {n_trials} trials per cell  (N = {n_total} simulations total)",
        subtitle=(
            f"baseline FORAS ≈ {base_wss:.3f} (grey)  ·  "
            "blue = degraded  ·  red = shortcut-leak\n"
            "small italic = std-dev across trials"
        ),
        cbar_label="Mean WSS@95",
        out_path=FIG_DIR / "heatmap_wss95.png",
    )

    render_heatmap(
        loss_mean_piv, loss_std_piv, base_loss,
        title=f"Mean Loss across {n_trials} trials per cell  (N = {n_total} simulations total)",
        subtitle=(
            f"baseline FORAS ≈ {base_loss:.3f} (grey)  ·  "
            "blue = improved  ·  red = degraded\n"
            "small italic = std-dev across trials"
        ),
        cbar_label="Mean Loss",
        out_path=FIG_DIR / "heatmap_loss.png",
        invert=True,
    )


if __name__ == "__main__":
    main()
