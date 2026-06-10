"""Generate mean recall plots per condition for SQ1.

Reads recall_curve.csv from each run directory, interpolates to a common
x-axis (proportion of corpus reviewed), computes mean +/- 95% CI per
condition, and produces two figures:

  outputs/SQ1/figures/recall_curves_full.png     (full 0-100% view)
  outputs/SQ1/figures/recall_curves_zoomed.png   (zoomed to first 15%)

Styling follows the ASReview recall-plot conventions used in
asreview-insights (x = proportion reviewed, y = recall, diagonal =
random baseline). Uses asreview's plot_recall internally for
single-condition reference if available; falls back to matplotlib.

Reads:  Report/outputs/SQ1/simulations/runs/seed_*__<cond>/recall_curve.csv
Writes: Report/outputs/SQ1/figures/recall_curves_full.png
        Report/outputs/SQ1/figures/recall_curves_zoomed.png

Usage:  python make_recall_plot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- paths (relative to code/SQ1/this.py -> Report/) ----------
SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
RUNS_DIR = REPORT_DIR / "outputs" / "SQ1" / "simulations" / "runs"
SUMMARY = REPORT_DIR / "outputs" / "SQ1" / "simulations" / "summary.json"
FIG_DIR = REPORT_DIR / "outputs" / "SQ1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- ASReview-style configuration ----------
# Condition order + colors matching ASReview's palette
CONDITIONS = ["baseline", "raw", "period", "full"]
COND_LABELS = {
    "baseline": "Baseline (no injection)",
    "raw": "Raw (original text)",
    "period": "Period (era rewrite)",
    "full": "Full (modern rewrite)",
}
# Colourblind-safe palette (Okabe-Ito / Tol), chosen so no two conditions
# are similar in hue (feedback R1: previous red/orange pair was too close).
COND_COLORS = {
    "baseline": "#0072B2",  # blue
    "raw": "#AA3377",       # purple-magenta
    "period": "#E69F00",    # orange
    "full": "#009E73",      # green
}
# Distinct linestyles so overlapping curves stay tellable apart even in
# greyscale print (feedback R1: curves are similar in shape).
COND_LINESTYLES = {
    "baseline": "solid",
    "raw": (0, (1, 1.2)),          # dotted
    "period": (0, (3, 1.5, 1, 1.5)),  # dash-dot
    "full": (0, (5, 1.5)),         # long dash
}

# Common x-axis: 1000 evenly spaced points from 0 to 1
X_INTERP = np.linspace(0, 1, 1001)


def load_recall_curves() -> dict[str, list[np.ndarray]]:
    """Load all recall curves, grouped by condition.

    Returns {condition: [array of recall values interpolated to X_INTERP]}.
    After the last step in a recall_curve.csv, recall = 1.0 for the
    remainder of the corpus (AsReview truncates at last positive found).
    """
    curves: dict[str, list[np.ndarray]] = {c: [] for c in CONDITIONS}

    # Get n_docs per condition from summary.json
    summary = json.loads(SUMMARY.read_text())
    n_docs_map: dict[str, int] = {}
    for run in summary["runs"]:
        n_docs_map[run["condition"]] = run["n_docs"]

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("seed_"):
            continue

        # Parse condition from dirname: seed_042__baseline -> baseline
        parts = run_dir.name.split("__", 1)
        if len(parts) != 2:
            continue
        cond = parts[1]
        if cond not in curves:
            continue

        rc_path = run_dir / "recall_curve.csv"
        if not rc_path.exists():
            print(f"  WARN: {rc_path} not found, skipping")
            continue

        df = pd.read_csv(rc_path)
        n_docs = n_docs_map.get(cond, df["step"].max())

        # x = proportion of corpus reviewed (0..1)
        # y = recall (0..1)
        x_raw = df["step"].values / n_docs
        y_raw = df["recall"].values

        # Prepend (0, 0) and append (1, 1) for full interpolation range
        x_raw = np.concatenate([[0], x_raw, [1.0]])
        y_raw = np.concatenate([[0], y_raw, [1.0]])

        # Interpolate to common grid
        y_interp = np.interp(X_INTERP, x_raw, y_raw)
        curves[cond].append(y_interp)

    for cond, arrs in curves.items():
        print(f"  {cond}: {len(arrs)} recall curves loaded")

    return curves


def plot_recall_comparison(
    curves: dict[str, list[np.ndarray]],
    xlim: tuple[float, float] = (0, 1),
    ylim: tuple[float, float] = (0, 1.02),
    out_path: Path | None = None,
    title: str = "",
    zoom_label: str | None = None,
):
    """Plot mean recall +/- 95% CI per condition, ASReview style."""

    fig, ax = plt.subplots(figsize=(7, 5))

    # Random baseline (diagonal)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8, label="Random", zorder=1)

    for cond in CONDITIONS:
        arrs = curves.get(cond, [])
        if not arrs:
            continue
        mat = np.array(arrs)  # shape (n_seeds, 1001)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        n = mat.shape[0]
        ci95 = 1.96 * std / np.sqrt(n)  # 95% confidence interval of the mean

        color = COND_COLORS[cond]
        label = COND_LABELS[cond]

        ax.plot(X_INTERP, mean, color=color, linewidth=1.6,
                linestyle=COND_LINESTYLES[cond], label=label, zorder=3)
        ax.fill_between(
            X_INTERP, mean - ci95, mean + ci95,
            color=color, alpha=0.15, zorder=2,
        )

    # ASReview-style formatting
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Proportion of corpus reviewed", fontsize=11)
    ax.set_ylabel("Recall (proportion of relevant papers found)", fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Add percentage tick labels for readability
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    if zoom_label:
        ax.annotate(
            zoom_label, xy=(0.02, 0.97), xycoords="axes fraction",
            fontsize=9, color="grey", va="top",
        )

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"wrote {out_path.relative_to(REPORT_DIR)}")
    plt.close(fig)


def main():
    print("Loading recall curves...")
    curves = load_recall_curves()

    #