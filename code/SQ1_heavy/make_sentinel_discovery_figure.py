"""SQ1 (ELAS Heavy / h3): time-to-discovery bar chart (discovery point per paper).

Heavy-model counterpart of SQ1/make_sentinel_discovery_figure.py. Grouped bars:
one group per injected historical elusive paper, one bar per rewrite condition
(raw / period / full), height = mean % of the corpus screened at discovery across
the ten seeds, error bars = std across seeds. Same colourblind-safe palette as the
u4 version so the two figures read side by side.

Reads:  Report/outputs/SQ1_heavy/simulations/summary.json
Writes: Report/outputs/SQ1_heavy/figures/sq1_sentinel_discovery_heavy.png

Usage:  python make_sentinel_discovery_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
SUMMARY = REPORT_DIR / "outputs" / "SQ1_heavy" / "simulations" / "summary.json"
OUT = REPORT_DIR / "outputs" / "SQ1_heavy" / "figures" / "sq1_sentinel_discovery_heavy.png"

CONDITIONS = ["raw", "period", "full"]
COND_LABELS = {"raw": "Raw", "period": "Period", "full": "Full"}
# Matches SQ1/make_sentinel_discovery_figure.py COND_COLORS (feedback R1).
COND_COLORS = {"raw": "#AA3377", "period": "#E69F00", "full": "#009E73"}

SENTINELS = [
    ("solomon_1988", "Solomon 1988\n(era: post-1980)"),
    ("kardiner_1947", "Kardiner 1947\n(era: 1945–1979)"),
    ("war_neuroses_tunisian_1944", "Grinker & Spiegel 1944\n(era: pre-1945)"),
]


def main() -> None:
    summary = json.loads(SUMMARY.read_text())

    # Collect discovery positions (% of corpus) per (sentinel, condition).
    pos: dict[tuple[str, str], list[float]] = {}
    for run in summary["runs"]:
        cond = run["condition"]
        if cond not in CONDITIONS:
            continue
        n_docs = run["n_docs"]
        for sp in run.get("sentinel_positions", []) or []:
            pos.setdefault((sp["sentinel_id"], cond), []).append(
                100.0 * sp["step"] / n_docs
            )

    n_papers = len(SENTINELS)
    width = 0.25
    x = np.arange(n_papers)

    fig, ax = plt.subplots(figsize=(12, 6))
    for k, cond in enumerate(CONDITIONS):
        means = [np.mean(pos[(sid, cond)]) for sid, _ in SENTINELS]
        stds = [np.std(pos[(sid, cond)]) for sid, _ in SENTINELS]
        offs = x + (k - 1) * width
        bars = ax.bar(
            offs, means, width, yerr=stds, capsize=4,
            color=COND_COLORS[cond], edgecolor="black", linewidth=0.6,
            label=COND_LABELS[cond],
        )
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 1.2, f"{m:.1f}%",
                    ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in SENTINELS], fontsize=11)
    ax.set_ylabel("% of corpus screened at discovery", fontsize=12)
    ax.set_title("Time-to-Discovery of Historical Elusive Papers by Condition "
                 "(ELAS Heavy)", fontsize=14)
    ax.legend(fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT.relative_to(REPORT_DIR)}")


if __name__ == "__main__":
    main()
