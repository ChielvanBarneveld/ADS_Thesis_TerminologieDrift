"""Generate the SQ2 historical-term distribution figure (horizontal bar chart).

The SQ2 regex rewrite replaces each modern PTSD-token with a historical variant
sampled from the candidate-pool distribution. This figure visualises that
sampling distribution (term -> probability weight).

Reads:  Report/data/SQ2/term_distribution.json   (produced by build_regex_dataset.py)
Writes: Report/outputs/SQ2/figures/term_distribution.png
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
DIST = REPORT_DIR / "data" / "SQ2" / "term_distribution.json"
OUT = REPORT_DIR / "outputs" / "SQ2" / "figures" / "term_distribution.png"


def main() -> None:
    dist = json.loads(DIST.read_text())
    # Sort ascending so the largest weight sits at the top of a horizontal bar.
    items = sorted(dist.items(), key=lambda kv: kv[1])
    terms = [t for t, _ in items]
    weights = [w * 100 for _, w in items]  # percentages

    fig, ax = plt.subplots(figsize=(8, 9))
    bars = ax.barh(terms, weights, color="#DD8452", edgecolor="black", linewidth=0.5)
    for b, w in zip(bars, weights):
        ax.text(w + 0.15, b.get_y() + b.get_height() / 2, f"{w:.1f}%",
                va="center", ha="left", fontsize=8)
    ax.set_xlabel("Sampling probability (% of historical-term draws)", fontsize=11)
    ax.set_title("SQ2 historical-term sampling distribution\n(candidate-pool frequency over 28 terms)", fontsize=12)
    ax.set_xlim(0, max(weights) + 2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150)
    print(f"wrote {OUT}: {len(terms)} terms")


if __name__ == "__main__":
    main()
