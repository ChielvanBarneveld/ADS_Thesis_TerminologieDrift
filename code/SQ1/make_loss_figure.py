"""Generate the SQ1 normalized-loss bar chart (loss by condition).

Standalone figure used in Results 4.1. SQ1 foregrounds time-to-discovery and
loss over WSS@95, so this replaces the earlier two-panel (WSS@95 + loss) figure
with a loss-only chart.

Reads:  Report/outputs/SQ1/summary_by_condition.csv
Writes: Report/outputs/SQ1/figures/sq1_loss_by_condition.png
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
SUMMARY = REPORT_DIR / "outputs" / "SQ1" / "summary_by_condition.csv"
OUT = REPORT_DIR / "outputs" / "SQ1" / "figures" / "sq1_loss_by_condition.png"

ORDER = ["baseline", "raw", "period", "full"]
LABELS = {
    "baseline": "Baseline\n(no injected papers)",
    "raw": "Raw\n(original text)",
    "period": "Period\n(era rewrite)",
    "full": "Full\n(modern rewrite)",
}
# Colourblind-safe palette (Okabe-Ito / Tol) — matches make_recall_plot.py (feedback R1).
COLORS = {"baseline": "#0072B2", "raw": "#AA3377", "period": "#E69F00", "full": "#009E73"}


def main() -> None:
    rows = {r["condition"]: r for r in csv.DictReader(open(SUMMARY))}
    means = [float(rows[c]["loss_mean"]) for c in ORDER]
    stds = [float(rows[c]["loss_std"]) for c in ORDER]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [LABELS[c] for c in ORDER], means, yerr=stds, capsize=5,
        color=[COLORS[c] for c in ORDER], edgecolor="black", linewidth=0.6,
    )
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.0012, f"{m:.4f}",
                ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Normalized loss", fontsize=12)
    ax.set_title("Normalized Loss by Condition", fontsize=14)
    ax.set_ylim(0, 0.037)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    