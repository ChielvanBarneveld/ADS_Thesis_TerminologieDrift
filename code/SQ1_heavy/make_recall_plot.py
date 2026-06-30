"""Mean recall plot per condition for SQ1 — HEAVY model (ELAS h3).

Mirror of ../SQ1/make_recall_plot.py, but reads the heavy-model outputs and
writes to Report/outputs/SQ1_heavy/figures/. Same asreview-insights style
(step plot, optimal + random baselines, mean +/- 95% CI across seeds,
X-markers at historical-elusive-paper discovery positions).

Reads:  Report/outputs/SQ1_heavy/simulations/runs/seed_*__<cond>/recall_curve.csv
        Report/outputs/SQ1_heavy/simulations/summary.json
Writes: Report/outputs/SQ1_heavy/figures/recall_curves_heavy.png

Usage:  python make_recall_plot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
RUNS_DIR = REPORT_DIR / "outputs" / "SQ1_heavy" / "simulations" / "runs"
SUMMARY = REPORT_DIR / "outputs" / "SQ1_heavy" / "simulations" / "summary.json"
FIG_DIR = REPORT_DIR / "outputs" / "SQ1_heavy" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["baseline", "raw", "period", "full"]
COND_LABELS = {
    "baseline": "Baseline (no injection)",
    "raw": "Raw (original text)",
    "period": "Period (era rewrite)",
    "full": "Full (modern rewrite)",
}
COND_COLORS = {
    "baseline": "#0072B2",
    "raw": "#AA3377",
    "period": "#E69F00",
    "full": "#009E73",
}
COND_LINESTYLES = {
    "baseline": "solid",
    "raw": (0, (1, 1.2)),
    "period": (0, (3, 1.5, 1, 1.5)),
    "full": (0, (5, 1.5)),
}
SENTINEL_LABELS = {
    "solomon_1988": "Solomon 1988",
    "kardiner_1947": "Kardiner 1947",
    "war_neuroses_tunisian_1944": "Grinker & Spiegel 1944",
}
X_INTERP = np.linspace(0, 1, 1001)


def load_recall_curves():
    curves = {c: [] for c in CONDITIONS}
    summary = json.loads(SUMMARY.read_text())
    n_docs_map, n_pos_map = {}, {}
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
            print(f"  WARN: {rc_path} missing, skipping"); continue
        df = pd.read_csv(rc_path)
        n_docs = n_docs_map.get(cond, int(df["step"].max()))
        x_raw = np.concatenate([[0], df["step"].values / n_docs, [1.0]])
        y_raw = np.concatenate([[0], df["recall"].values, [1.0]])
        curves[cond].append(np.interp(X_INTERP, x_raw, y_raw))
    for c, arrs in curves.items():
        print(f"  {c}: {len(arrs)} curves")
    return curves, n_docs_map["baseline"], n_pos_map["baseline"]


def load_sentinel_positions():
    summary = json.loads(SUMMARY.read_text())
    positions = {}
    for run in summary["runs"]:
        cond = run["condition"]; n_docs = run["n_docs"]
        for sp in run.get("sentinel_positions", []) or []:
            key = (cond, sp["sentinel_id"])
            x = sp["step"] / n_docs
            y = sp.get("recall_at_step")
            if y is None:
                # recall_at_step may be absent in reconstructed summaries; approximate
                y = None
            positions.setdefault(key, []).append((x, y))
    result = {}
    for (cond, sid), vals in positions.items():
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals if v[1] is not None]
        result.setdefault(cond, {})[sid] = (float(np.mean(xs)), float(np.mean(ys)) if ys else None)
    return result


def main():
    print("Loading recall curves...")
    curves, n_docs, n_pos = load_recall_curves()
    sent_pos = load_sentinel_positions()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.step(X_INTERP, X_INTERP, color="black", linewidth=0.8, label="Random", where="post", zorder=1)
    optimal_x = np.concatenate([[0], np.linspace(0, n_pos / n_docs, n_pos + 1), [1.0]])
    optimal_y = np.concatenate([[0], np.linspace(0, 1, n_pos + 1), [1.0]])
    ax.step(optimal_x, optimal_y, color="grey", linewidth=0.8, label="Optimal", where="post", zorder=1)

    for cond in CONDITIONS:
        arrs = curves.get(cond, [])
        if not arrs:
            continue
        mat = np.array(arrs)
        mean = mat.mean(axis=0); std = mat.std(axis=0)
        ci95 = 1.96 * std / np.sqrt(mat.shape[0])
        color = COND_COLORS[cond]
        ax.step(X_INTERP, mean, color=color, linewidth=1.6, linestyle=COND_LINESTYLES[cond],
                label=COND_LABELS[cond], where="post", zorder=3)
        ax.fill_between(X_INTERP, mean - ci95, mean + ci95, color=color, alpha=0.12, step="post", zorder=2)

    added = set()
    for cond in ["raw", "period", "full"]:
        if cond not in sent_pos:
            continue
        color = COND_COLORS[cond]
        for sid, (sx, sy) in sent_pos[cond].items():
            if sy is None:
                continue
            label = SENTINEL_LABELS.get(sid, sid) if sid not in added else None
            ax.scatter(sx, sy, marker="X", s=80, color=color, edgecolors="black",
                       linewidths=0.5, zorder=5, label=label)
            added.add(sid)

    ax.set_title("Recall - ELAS h3 (heavy / mxbai)")
    ax.set(xlabel="Proportion of labeled records", ylabel="Recall")
    ax.set_ylim([-0.05, 1.05]); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=2)
    plt.tight_layout()
    out = FIG_DIR / "recall_curves_heavy.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
