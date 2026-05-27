#!/usr/bin/env python3
"""
make_intro_terminology_timeline.py — generates Figure 1.2 of the thesis:
the evolution of clinical terminology for post-traumatic stress, 1860–2025.

Output: outputs/paper/figures/intro_ptsd_terminology_timeline.png

Design choices:
  * Gantt-style horizontal bars per historical term, each bar spanning the
    period in which the term was a dominant clinical descriptor.
  * War-band background shading (US Civil War, WWI, WWII, Korea, Vietnam)
    contextualises each term's emergence.
  * Vertical dotted lines mark key publications and DSM revisions.
  * Era-labels on the right collapse the nine pre-DSM-III categories into
    the conceptual buckets the field used (cardiological, psychoanalytic, ...).

Run:
  python3 outputs/paper/figures/make_intro_terminology_timeline.py
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Data: one row per term, each with (label, start_year, end_year, color, era)
# Periods follow Jones & Wessely (2005) Shell Shock to PTSD and the
# era-style guide at outputs/SQ2/era_style_guide.md.
# ─────────────────────────────────────────────────────────────────────────────
TERMS = [
    ("Soldier's heart / irritable heart",        1862, 1895, "#5B7FB6", "Cardiological"),
    ("Effort syndrome / neurocirculatory asthenia", 1916, 1945, "#E08E3C", "Cardiovascular"),
    ("Shell shock (concussion)",                 1915, 1925, "#5FA065", "Neuro → psychogenic"),
    ("War neurosis / traumatic neurosis",        1915, 1950, "#B95C5C", "Psychoanalytic"),
    ("War hysteria",                             1915, 1940, "#9C7DB6", "Psychiatric-hysterical"),
    ("Combat / battle fatigue, battle exhaustion", 1941, 1960, "#8C7A6B", "Frontline psychiatry"),
    ("Operational fatigue",                      1943, 1960, "#D896B9", "Aviation medicine"),
    ("Gross stress reaction (DSM-I)",            1952, 1968, "#7BA8B2", "Psychiatric reactive"),
    ("Combat stress reaction",                   1973, 1995, "#B8B040", "Pre-DSM-III reactive"),
    ("Post-traumatic stress disorder (DSM-III+)", 1980, 2025, "#4A8C5A", "Modern DSM"),
]

# War bands (start, end, label).
WARS = [
    (1861, 1865, "US Civil War"),
    (1914, 1918, "WWI"),
    (1939, 1945, "WWII"),
    (1950, 1953, "Korea"),
    (1955, 1975, "Vietnam"),
]

# Key publications / DSM revisions.
LANDMARKS = [
    (1871, "Da Costa"),
    (1915, "Myers"),
    (1941, "Kardiner"),
    (1952, "DSM-I"),
    (1980, "DSM-III"),
    (1994, "DSM-IV"),
    (2005, "Jones &\nWessely"),
    (2013, "DSM-5"),
]

X_MIN, X_MAX = 1860, 2030


def main(out_path: Path | None = None) -> Path:
    out_path = out_path or Path(__file__).resolve().parent / "intro_ptsd_terminology_timeline.png"

    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=140)

    # War-band shading (drawn first, behind everything).
    for start, end, _label in WARS:
        ax.axvspan(start, end, facecolor="#F3E2C8", alpha=0.55, zorder=0)

    # Term bars.
    for i, (label, t0, t1, color, _era) in enumerate(TERMS):
        y = len(TERMS) - i - 1
        ax.barh(y, t1 - t0, left=t0, height=0.55, color=color, edgecolor="none", zorder=2)
        # Era-label on the right
        ax.text(X_MAX + 6, y, _era, va="center", ha="left", fontsize=8.5, color="#555")

    # Y-axis: term names
    ax.set_yticks(range(len(TERMS)))
    ax.set_yticklabels([t[0] for t in reversed(TERMS)], fontsize=9.5)
    ax.tick_params(axis="y", length=0, pad=2)

    # X-axis
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_xticks(range(1875, 2051, 25))
    ax.tick_params(axis="x", labelsize=9)

    # Landmark vertical lines + labels at top.
    y_top = len(TERMS) - 0.3
    for year, name in LANDMARKS:
        ax.axvline(year, color="#666", linewidth=0.7, linestyle=(0, (2, 2)), zorder=1, alpha=0.55)
        ax.text(year, y_top + 0.6, name, rotation=60, ha="left", va="bottom",
                fontsize=8, color="#333")

    # War-band labels at bottom.
    for start, end, label in WARS:
        ax.text((start + end) / 2, -0.95, label, ha="center", va="top",
                fontsize=7.5, color="#7A6037", fontweight="bold")

    # Cosmetics
    ax.set_title("Evolution of clinical terminology for post-traumatic stress, 1860 – 2025",
                 fontsize=12, pad=18)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#999")
    ax.set_ylim(-1.6, len(TERMS) + 1.3)
    ax.grid(False)

    # Footer
    footer = ("Each bar marks the period in which the term was a dominant clinical descriptor "
              "in the English-language military-psychiatric literature. Background shading marks "
              "the major wars that shaped clinical observation; vertical dotted lines mark key "
              "publications and DSM revisions. The DSM-III in 1980 folded the older categories "
              "into a single label — post-traumatic stress disorder — which thereafter "
              "dominates the vocabulary.")
    fig.text(0.07, -0.02, footer, fontsize=7.8, color="#555", wrap=True, ha="left")

    plt.tight_layout(rect=(0, 0.02, 0.85, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140, facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}")
