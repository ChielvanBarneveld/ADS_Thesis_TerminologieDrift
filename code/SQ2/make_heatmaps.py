"""SQ2: render WSS@95 and Loss heatmaps from the grid sweep.

Reads:  Report/outputs/SQ2/grid_summary.csv
Writes: Report/outputs/SQ2/figures/heatmap_wss95.png
        Report/outputs/SQ2/figures/heatmap_loss.png

Both heatmaps share the same 6x6 axes:
    x = pp (% of positives rewritten)
    y = nn (% of negatives rewritten)
Cell values: mean across N_TRIALS seeds.
Loss formula matches asreview-insights (v1.6.1).
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
SUMMARY    = REPORT_DIR / "outputs" / "SQ2" / "grid_summary.csv"
FIG_DIR    = REPORT_DIR / "outputs" / "SQ2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def render_heatmap(pivot, title, cbar_label, out_path, cmap="viridis", value_fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    arr = pivot.values
    im = ax.imshow(arr, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Positives rewritten (%)")
    ax.set_ylabel("Negatives rewritten (%)")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, value_fmt.format(v), ha="center", va="center",
                        color="white" if (arr.max() - v) > (arr.max() - arr.min()) / 2 else "black",
                        fontsize=9)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPORT_DIR)}")


def main():
    df = pd.read_csv(SUMMARY)
    print(f"loaded {SUMMARY.name}: {len(df)} cells")

    pos_levels = sorted(df["pp"].unique())
    neg_levels = sorted(df["nn"].unique())
    wss_pivot = df.pivot(index="nn", columns="pp", values="wss_95_mean").reindex(index=neg_levels, columns=pos_levels)
    loss_col = "loss_mean" if "loss_mean" in df.columns else "normalized_loss_mean"
    loss_pivot = df.pivot(index="nn", columns="pp", values=loss_col).reindex(index=neg_levels, columns=pos_levels)

    render_heatmap(wss_pivot,
                   title="ELAS u4 -- WSS@95 by drift mix",
                   cbar_label="WSS@95 (higher = better)",
                   out_path=FIG_DIR / "heatmap_wss95.png",
                   cmap="viridis", value_fmt="{:.3f}")

    render_heatmap(loss_pivot,
                   title="ELAS u4 -- Loss by drift mix (asreview-insights formula)",
                   cbar_label="Loss (lower = better)",
                   out_path=FIG_DIR / "heatmap_loss.png",
                   cmap="viridis_r", value_fmt="{:.3f}")


if __name__ == "__main__":
    main()
