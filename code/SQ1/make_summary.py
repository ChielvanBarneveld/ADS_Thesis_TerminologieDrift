"""Aggregate per-run summaries across seeds into a thesis-ready summary.

Reads:  Report/outputs/SQ1/simulations/summary.json   (written by run_simulations.py)
Writes: Report/outputs/SQ1/summary_by_condition.csv   (mean +- std per condition)
        Report/outputs/SQ1/summary_by_sentinel.csv    (per-sentinel TD distribution)

Run after `run_simulations.py` has produced >= 1 run per condition.
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
SIM_DIR    = REPORT_DIR / "outputs" / "SQ1" / "simulations"
RAW        = SIM_DIR / "summary.json"
OUT_COND   = REPORT_DIR / "outputs" / "SQ1" / "summary_by_condition.csv"
OUT_SENT   = REPORT_DIR / "outputs" / "SQ1" / "summary_by_sentinel.csv"


def _ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    if len(xs) == 1:
        return xs[0], None, 1
    return statistics.fmean(xs), statistics.stdev(xs), len(xs)


def main():
    data = json.loads(RAW.read_text())
    runs = data["runs"]
    df = pd.DataFrame(runs)

    cond_rows = []
    for cond, sub in df.groupby("condition"):
        wss95_m, wss95_s, n = _ms(sub["wss_95"].tolist())
        wss80_m, wss80_s, _ = _ms(sub["wss_80"].tolist())
        wss90_m, wss90_s, _ = _ms(sub["wss_90"].tolist())
        loss_col = "loss" if "loss" in sub.columns else "normalized_loss"
        loss_m, loss_s, _   = _ms(sub[loss_col].tolist())
        atd_m, atd_s, _     = _ms(sub.get("atd", pd.Series([None] * len(sub))).tolist())
        steps95_m, steps95_s, _ = _ms(sub["steps_to_95"].tolist())
        cond_rows.append({
            "condition": cond,
            "n_seeds": n,
            "wss_80_mean": wss80_m, "wss_80_std": wss80_s,
            "wss_90_mean": wss90_m, "wss_90_std": wss90_s,
            "wss_95_mean": wss95_m, "wss_95_std": wss95_s,
            "loss_mean": loss_m, "loss_std": loss_s,
            "atd_mean": atd_m, "atd_std": atd_s,
            "steps_to_95_mean": steps95_m, "steps_to_95_std": steps95_s,
        })
    cond_df = pd.DataFrame(cond_rows).sort_values("condition")
    cond_df.to_csv(OUT_COND, index=False)
    print(f"wrote {OUT_COND.relative_to(REPORT_DIR)}")
    print(cond_df.to_string(index=False))

    sent_rows = []
    for _, run in df.iterrows():
        for sp in run.get("sentinel_positions", []) or []:
            sent_rows.append({
                "condition": run["condition"],
                "seed": run["seed"],
                "sentinel_id": sp["sentinel_id"],
                "step": sp["step"],
                "percent_screened": sp["percent_screened"],
            })
    if sent_rows:
        sent_df = pd.DataFrame(sent_rows)
        agg = sent_df.groupby(["condition", "sentinel_id"]).agg(
            n_seeds=("seed", "count"),
            step_mean=("step", "mean"),
            step_std=("step", "std"),
            step_min=("step", "min"),
            step_max=("step", "max"),
            percent_screened_mean=("percent_screened", "mean"),
        ).reset_index().sort_values(["condition", "sentinel_id"])
        agg.to_csv(OUT_SENT, index=False)
        print(f"\nwrote {OUT_SENT.relative_to(REPORT_DIR)}")
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
