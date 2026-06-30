"""Compare ELAS u4 vs ELAS h3 (heavy) on the SQ1 sentinel-injection runs.

Reads both summary.json files and writes a side-by-side table:
  * per condition (baseline/raw/period/full): mean WSS@95, Loss, ATD (+ SD, n)
  * per historical elusive paper: mean discovery step (time-to-discovery) and
    mean % screened at discovery, per condition, for each model.

Pure pandas/json — does NOT import asreview, so it runs anywhere (incl. this
sandbox) once the heavy runs exist. Outputs:
  Report/outputs/SQ1_heavy/u4_vs_h3_comparison.csv
  Report/outputs/SQ1_heavy/u4_vs_h3_comparison.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
U4_SUMMARY = REPORT_DIR / "outputs" / "SQ1" / "simulations" / "summary.json"
H3_SUMMARY = REPORT_DIR / "outputs" / "SQ1_heavy" / "simulations" / "summary.json"
OUT_DIR = REPORT_DIR / "outputs" / "SQ1_heavy"
COND_ORDER = ["baseline", "raw", "period", "full"]


def load_runs(path, model_label):
    if not path.exists():
        return None
    runs = json.loads(path.read_text()).get("runs", [])
    return runs if runs else None


def runs_to_df(runs, model):
    rows = []
    for r in runs:
        rows.append({
            "model": model, "condition": r.get("condition"), "seed": r.get("seed"),
            "wss_95": r.get("wss_95"), "loss": r.get("loss"), "atd": r.get("atd"),
            "steps_to_95": r.get("steps_to_95"),
        })
    return pd.DataFrame(rows)


def sentinel_df(runs, model):
    rows = []
    for r in runs:
        for sp in (r.get("sentinel_positions") or []):
            rows.append({
                "model": model, "condition": r.get("condition"), "seed": r.get("seed"),
                "sentinel_id": sp.get("sentinel_id"),
                "step": sp.get("step"), "percent_screened": sp.get("percent_screened"),
            })
    return pd.DataFrame(rows)


def main():
    u4 = load_runs(U4_SUMMARY, "u4")
    h3 = load_runs(H3_SUMMARY, "h3")
    if u4 is None:
        print(f"No u4 runs at {U4_SUMMARY}")
        return
    if h3 is None:
        print(f"No heavy (h3) runs yet at {H3_SUMMARY}.")
        print("Run run_simulations_heavy.py first, then re-run this script.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([runs_to_df(u4, "u4"), runs_to_df(h3, "h3")], ignore_index=True)

    agg = (df.groupby(["condition", "model"])
             .agg(n=("seed", "count"),
                  wss95_mean=("wss_95", "mean"), wss95_sd=("wss_95", "std"),
                  loss_mean=("loss", "mean"), loss_sd=("loss", "std"),
                  atd_mean=("atd", "mean"))
             .reset_index())
    agg["__o"] = agg["condition"].map({c: i for i, c in enumerate(COND_ORDER)}).fillna(99)
    agg = agg.sort_values(["__o", "model"]).drop(columns="__o")
    agg.to_csv(OUT_DIR / "u4_vs_h3_comparison.csv", index=False)

    # sentinel time-to-discovery
    sdf = pd.concat([sentinel_df(u4, "u4"), sentinel_df(h3, "h3")], ignore_index=True)
    sent_agg = pd.DataFrame()
    if not sdf.empty:
        sent_agg = (sdf.groupby(["condition", "sentinel_id", "model"])
                       .agg(td_step_mean=("step", "mean"),
                            pct_screened_mean=("percent_screened", "mean"),
                            n=("seed", "count"))
                       .reset_index())

    # markdown report
    lines = ["# ELAS u4 vs ELAS h3 (heavy) — SQ1 comparison", "",
             f"u4 runs: {len(u4)} · h3 runs: {len(h3)}", "",
             "## Per-condition means (WSS@95 / Loss / ATD)", "",
             "| condition | model | n | WSS@95 | Loss | ATD |",
             "|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['condition']} | {r['model']} | {int(r['n'])} | "
            f"{r['wss95_mean']:.4f} | {r['loss_mean']:.4f} | {r['atd_mean']:.1f} |")
    if not sent_agg.empty:
        lines += ["", "## Historical elusive paper time-to-discovery (mean step)", "",
                  "| condition | sentinel | model | mean step | mean % screened | n |",
                  "|---|---|---|---|---|---|"]
        so = sent_agg.copy()
        so["__o"] = so["condition"].map({c: i for i, c in enumerate(COND_ORDER)}).fillna(99)
        for _, r in so.sort_values(["__o", "sentinel_id", "model"]).iterrows():
            lines.append(
                f"| {r['condition']} | {r['sentinel_id']} | {r['model']} | "
                f"{r['td_step_mean']:.0f} | {r['pct_screened_mean']:.2f}% | {int(r['n'])} |")
    (OUT_DIR / "u4_vs_h3_comparison.md").write_text("\n".join(lines) + "\n")

    print("Wrote:")
    print(" ", (OUT_DIR / "u4_vs_h3_comparison.csv").relative_to(REPORT_DIR))
    print(" ", (OUT_DIR / "u4_vs_h3_comparison.md").relative_to(REPORT_DIR))
    print("\n" + "\n".join(lines[:30]))


if __name__ == "__main__":
    main()
