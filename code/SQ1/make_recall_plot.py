"""Generate mean recall plot per condition for SQ1.

Replicates the asreview-insights plot_recall style (step plot, optimal
recall line, black random baseline) with mean +/- 95% CI across seeds
and X-markers at elusive-paper discovery positions.

If asreviewcontrib.insights is installed, delegates axis styling to the
library. Otherwise uses a faithful replica based on the asreview-insights
v1.6 source (plot.py::_add_recall_info).

Reads:  Report/outputs/SQ1/simulations/runs/seed_*__<cond>/recall_curve.csv
        Report/outputs/SQ1/simulations/summary.json
Writes: Report/outputs/SQ1/figures/recall_curves.png

Usage:  python make_recall_plot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- paths (code/SQ1/this.py -> code/ -> Report/) ----------
SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
RUNS_DIR = REPORT_DIR / "outputs" / "SQ1" / "simulations" / "runs"
SUMMARY = REPORT_DIR / "outputs" / "SQ1" / "simulations" / "summary.json"
FIG_DIR = REPORT_DIR / "outputs" / "SQ1" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- condition config ----------
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

SENTINEL_LABELS = {
    "solomon_1988": "Solomon 1988",
    "kardiner_1947": "Kardiner 1947",
    "war_neuroses_tunisian_1944": "Grinker & Spiegel 1944",
}
SENTINEL_MARKERS = {
    "solomon_1988": "X",
    "kardiner_1947": "X",
    "war_neuroses_tunisian_1944": "X",
}

X_INTERP = np.linspace(0, 1, 1001)


def load_recall_curves() -> tuple[dict[str, list[np.ndarray]], int, int]:
    """Load recall curves grouped by condition.

    Returns (curves, n_docs_baseline, n_pos_baseline).
    """
    curves: dict[str, list[np.ndarray]] = {c: [] for c in CONDITIONS}

    summary = json.loads(SUMMARY.read_text())
    n_docs_map: dict[str, int] = {}
    n_pos_map: dict[str, int] = {}
    for run in summary["runs"]:
        n_docs_map[run["condition"]] = run["n_docs"]
        n_pos_map[run["condition"]] = run["n_positives"]

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("seed_"):
            continue
        parts = run_dir.name.split("__", 1)
        if len(parts) != 2:
            continue
        cond = parts[1]
        if cond not in curves:
            continue

        rc_path = run_dir / "recall_curve.csv"
        if not rc_path.exists():
            print(f"  WARN: {rc_path} missing, skipping")
            continue

        df = pd.read_csv(rc_path)
        n_docs = n_docs_map.get(cond, int(df["step"].max()))
        x_raw = np.concatenate([[0], df["step"].values / n_docs, [1.0]])
        y_raw = np.concatenate([[0], df["recall"].values, [1.0]])
        curves[cond].append(np.interp(X_INTERP, x_raw, y_raw))

    for c, arrs in curves.items():
        print(f"  {c}: {len(arrs)} curves")

    return curves, n_docs_map["baseline"], n_pos_map["baseline"]


def load_sentinel_positions() -> dict[str, dict[str, tuple[float, float]]]:
    """Load mean sentinel discovery positions per condition.

    Returns {condition: {sentinel_id: (mean_x_proportion, mean_recall)}}.
    """
    summary = json.loads(SUMMARY.read_text())
    # Collect per (cond, sentinel_id)
    positions: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for run in summary["runs"]:
        cond = run["condition"]
        n_docs = run["n_docs"]
        for sp in run.get("sentinel_positions", []) or []:
            key = (cond, sp["sentinel_id"])
            x = sp["step"] / n_docs
            y = sp["recall_at_step"]
            positions.setdefault(key, []).append((x, y))

    result: dict[str, dict[str, tuple[float, float]]] = {}
    for (cond, sid), vals in positions.items():
        xs, ys = zip(*vals)
        result.setdefault(cond, {})[sid] = (float(np.mean(xs)), float(np.mean(ys)))

    return result


def _apply_asreview_style(ax, n_pos):
    """Apply asreview-insights recall-plot axis styling.

    Tries to import asreviewcontrib.insights; if unavailable, replicates
    the _add_recall_info style from asreview-insights v1.6.
    """
    try:
        from asreviewcontrib.insights.plot import _add_recall_info
        # _add_recall_info expects a labels list; we fake it for styling only
        fake_labels = [1] * n_pos + [0] * (10000 - n_pos)
        _add_recall_info(ax, fake_labels, x_absolute=False, y_absolute=False)
        return
    except Exception:
        pass

    # Faithful replica of asreview-insights v1.6 _add_recall_info
    ax.set_title("Recall")
    ax.set(
        xlabel="Proportion of labeled records",
        ylabel="Recall",
    )
    ax.set_ylim([-0.05, 1.05])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])


def main():
    print("Loading recall curves...")
    curves, n_docs, n_pos = load_recall_curves()
    sent_pos = load_sentinel_positions()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # --- Random baseline (black step — asreview-insights convention) ---
    ax.step(X_INTERP, X_INTERP, color="black", linewidth=0.8,
            label="Random", where="post", zorder=1)

    # --- Optimal recall (grey step — reaches 1.0 at n_pos/n_docs) ---
    optimal_x = np.concatenate([[0], np.linspace(0, n_pos / n_docs, n_pos + 1), [1.0]])
    optimal_y = np.concatenate([[0], np.linspace(0, 1, n_pos + 1), [1.0]])
    ax.step(optimal_x, optimal_y, color="grey", linewidth=0.8,
            label="Optimal", where="post", zorder=1)

    # --- Mean recall curves with CI bands ---
    for cond in CONDITIONS:
        arrs = curves.get(cond, [])
        if not arrs:
            continue
        mat = np.array(arrs)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        ci95 = 1.96 * std / np.sqrt(mat.shape[0])

        color = COND_COLORS[cond]
        ax.step(X_INTERP, mean, color=color, linewidth=1.6,
                linestyle=COND_LINESTYLES[cond],
                label=COND_LABELS[cond], where="post", zorder=3)
        ax.fill_between(X_INTERP, mean - ci95, mean + ci95,
                        color=color, alpha=0.12, step="post", zorder=2)

    # --- X-markers at elusive-paper discovery positions ---
    marker_legend_added = set()
    for cond in ["raw", "period", "full"]:
        if cond not in sent_pos:
            continue
        color = COND_COLORS[cond]
        for sid, (sx, sy) in sent_pos[cond].items():
            label = SENTINEL_LABELS.get(sid, sid) if sid not in marker_legend_added else None
            ax.scatter(sx, sy, marker="X", s=80, color=color,
                       edgecolors="black", linewidths=0.5, zorder=5,
                       label=label)
            marker_legend_added.add(sid)

    # --- Axis styling (asreview-insights convention) ---
    _apply_asreview_style(ax, n_pos)

    # Override: our legend includes conditions + sentinels
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=2)

    plt.tight_layout()
    out = FIG_DIR / "recall_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPORT_DIR)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
