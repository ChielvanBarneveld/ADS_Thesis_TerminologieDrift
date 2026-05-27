#!/usr/bin/env python3
"""
make_drift_figure.py — generates the PTSD terminology timeline figure.

Data-driven: reads the historical elusive papers candidate pool
(data/SQ1/candidates_with_terms.csv) and computes year ranges per term
group from actual publication years. Term grouping and era labels follow
Crocq and Crocq (2000) and Jones and Wessely (2005).

Output: outputs/SQ1/figures/terminology_timeline.png

Run:
  python3 code/SQ1/make_drift_figure.py
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent.parent
CAND_CSV   = ROOT_DIR / "data" / "SQ1" / "candidates_with_terms.csv"
OUT_DIR    = ROOT_DIR / "outputs" / "SQ1" / "figures"

# ─────────────────────────────────────────────────────────────────────
# Term grouping: pool terms → display groups.
# Grouping is domain knowledge (Crocq & Crocq 2000, Jones & Wessely 2005);
# year ranges are computed from the candidate pool data.
# ─────────────────────────────────────────────────────────────────────
TERM_GROUPS = [
    # (display_label, [pool_terms], color, era_label)
    ("Disordered action of the heart",
     ["disordered action of the heart"],
     "#5B7FB6", "Cardiological"),
    ("Railway spine",
     ["railway spine"],
     "#4A7A8C", "Industrial trauma"),
    ("Soldier's heart",
     ["soldier's heart", "soldiers heart"],
     "#6B8DB5", "Cardiological"),
    ("Shell shock",
     ["shell shock", "shell shocked", "shell concussion"],
     "#5FA065", "WWI neuro-psychiatric"),
    ("War neurosis / war neuroses",
     ["war neurosis", "war neuroses"],
     "#B95C5C", "Psychoanalytic"),
    ("Traumatic neurosis",
     ["traumatic neurosis"],
     "#C47A5C", "Psychoanalytic"),
    ("War hysteria",
     ["war hysteria"],
     "#9C7DB6", "Psychiatric"),
    ("Effort syndrome",
     ["effort syndrome"],
     "#E08E3C", "Cardiovascular"),
    ("Da Costa's syndrome",
     ["Da Costa's syndrome", "Da Costa syndrome"],
     "#D4A05A", "Cardiovascular"),
    ("Kriegsneurose",
     ["Kriegsneurose"],
     "#8B6B5C", "German psychiatric"),
    ("Battle / combat fatigue",
     ["battle fatigue", "combat fatigue", "battle exhaustion"],
     "#8C7A6B", "Frontline psychiatry"),
    ("Combat neurosis",
     ["combat neurosis"],
     "#7A6B5C", "Frontline psychiatry"),
    ("Operational fatigue",
     ["operational fatigue", "operational exhaustion"],
     "#D896B9", "Aviation medicine"),
    ("Gross stress reaction (DSM-I)",
     ["gross stress reaction"],
     "#7BA8B2", "DSM-I reactive"),
    ("Transient situational disturbance",
     ["transient situational disturbance"],
     "#6B9BA5", "DSM-II reactive"),
    ("Post-Vietnam syndrome",
     ["post-Vietnam syndrome"],
     "#A5B840", "Pre-DSM-III"),
    ("Combat stress reaction",
     ["combat stress reaction"],
     "#B8B040", "Post-DSM-III reactive"),
    ("Névrose traumatique",
     ["névrose traumatique"],
     "#9B7A9C", "French psychiatric"),
    ("Névrose de guerre",
     ["névrose de guerre"],
     "#8A6A8B", "French psychiatric"),
    ("Obusite",
     ["obusite"],
     "#7A5A7B", "French WWI"),
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


def load_term_ranges(csv_path: Path) -> dict:
    """Read candidate pool, compute per-group year ranges from data."""
    df = pd.read_csv(csv_path)
    
    # Explode found_via_terms → one row per (paper, term)
    rows = []
    for _, r in df.iterrows():
        yr = r["year"]
        if pd.isna(yr) or pd.isna(r["found_via_terms"]):
            continue
        for t in str(r["found_via_terms"]).split(";"):
            rows.append({"term": t.strip(), "year": int(yr)})
    term_df = pd.DataFrame(rows)
    
    ranges = {}
    for label, pool_terms, color, era in TERM_GROUPS:
        sub = term_df[term_df["term"].isin(pool_terms)]["year"]
        if sub.empty:
            continue
        ranges[label] = {
            "n": len(sub),
            "min": int(sub.min()),
            "p5": int(sub.quantile(0.05)),
            "median": int(sub.median()),
            "p95": int(sub.quantile(0.95)),
            "max": int(sub.max()),
            "color": color,
            "era": era,
        }
    return ranges


def main(out_path: Path | None = None) -> Path:
    out_path = out_path or OUT_DIR / "terminology_timeline.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    ranges = load_term_ranges(CAND_CSV)
    
    # Sort by p5 year (earliest start at top of figure = bottom of list)
    sorted_terms = sorted(ranges.items(), key=lambda x: x[1]["p5"])
    
    X_MIN, X_MAX = 1860, 2030
    n_terms = len(sorted_terms)
    
    fig, ax = plt.subplots(figsize=(14, max(6.5, n_terms * 0.38)), dpi=140)
    
    # War-band shading
    for start, end, _label in WARS:
        ax.axvspan(start, end, facecolor="#F3E2C8", alpha=0.55, zorder=0)
    
    # Term bars: p5–p95 as solid bar, whiskers to min–max
    for i, (label, info) in enumerate(sorted_terms):
        y = n_terms - i - 1
        
        # Whisker: min to max (thin line)
        ax.plot([info["min"], info["max"]], [y, y],
                color=info["color"], linewidth=1, alpha=0.35, zorder=1)
        
        # Main bar: p5 to p95
        bar_width = info["p95"] - info["p5"]
        ax.barh(y, bar_width, left=info["p5"], height=0.5,
                color=info["color"], edgecolor="none", zorder=2, alpha=0.85)
        
        # Paper count annotation (fixed column at X_MAX - 30)
        ax.text(X_MAX - 25, y, f"n={info['n']}",
                va="center", ha="right", fontsize=7.5, color="#666")
        
        # Era-label on the far right
        ax.text(X_MAX + 6, y, info["era"],
                va="center", ha="left", fontsize=8, color="#555")
    
    # Y-axis: term names
    ax.set_yticks(range(n_terms))
    ax.set_yticklabels([t[0] for t in reversed(sorted_terms)], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=2)
    
    # X-axis
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xlabel("Publication year", fontsize=10)
    ax.set_xticks(range(1875, 2051, 25))
    ax.tick_params(axis="x", labelsize=9)
    
    # Landmarks
    y_top = n_terms - 0.3
    for year, name in LANDMARKS:
        ax.axvline(year, color="#666", linewidth=0.7,
                   linestyle=(0, (2, 2)), zorder=1, alpha=0.55)
        ax.text(year, y_top + 0.6, name, rotation=60, ha="left",
                va="bottom", fontsize=8, color="#333")
    
    # War-band labels
    for start, end, label in WARS:
        ax.text((start + end) / 2, -0.95, label, ha="center", va="top",
                fontsize=7.5, color="#7A6037", fontweight="bold")
    
    # Cosmetics
    ax.set_title(
        "Historical PTSD terminology in the candidate pool (n=2,288 papers, 28 terms)",
        fontsize=11.5, pad=18)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#999")
    ax.set_ylim(-1.6, n_terms + 1.3)
    ax.grid(False)
    
    # Footer
    footer = (
        "Bars span the 5th to 95th percentile of publication years; "
        "whiskers extend to the full range. Paper counts (n) reflect "
        "the number of candidate pool papers found via each term. "
        "Data source: candidates_with_terms.csv (OpenAlex snapshot, "
        "28 April 2026)."
    )
    fig.text(0.07, -0.02, footer, fontsize=7.8, color="#555",
             wrap=True, ha="left")
    
    plt.tight_layout(rect=(0, 0.02, 0.82, 1))
    fig.savefig(out_path, bbox_inches="tight", dpi=140, facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}")
