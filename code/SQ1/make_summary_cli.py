"""Aggregate metrics from .asreview files via `asreview metrics` CLI.

Reads:  Report/outputs/SQ1/simulations/asreview_files/*.asreview
Writes: Report/outputs/SQ1/simulations/asreview_files/<name>__metrics.json  (raw insights output, per run)
        Report/outputs/SQ1/summary_by_condition_cli.csv                     (aggregated, parallel to summary_by_condition.csv)

Compare summary_by_condition_cli.csv with summary_by_condition.csv to verify
that our Python-API metrics match the official asreview-insights numbers.
"""
from __future__ import annotations
import json
import re
import statistics
import subprocess
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
ASREVIEW_DIR = REPORT_DIR / "outputs" / "SQ1" / "simulations" / "asreview_files"
OUT_COND_CLI = REPORT_DIR / "outputs" / "SQ1" / "summary_by_condition_cli.csv"


def extract_metric(items, metric_id, recall_target=None):
    """Pull a metric value from `asreview metrics` JSON output.

    metric_id: 'recall', 'wss', 'loss', 'atd', 'td', 'erf', etc.
    recall_target: for wss/erf, the recall threshold (e.g. 0.95).
    """
    for item in items:
        if item["id"] == metric_id:
            v = item["value"]
            if metric_id in ("loss", "atd"):
                return float(v)
            if metric_id in ("wss", "erf", "recall"):
                # value is list of [target, value] pairs
                for target, val in v:
                    if recall_target is None or abs(target - recall_target) < 1e-6:
                        return float(val)
            return v
    return None


def metrics_for_file(asreview_path):
    """Run `asreview metrics` on a .asreview file; return parsed JSON dict."""
    out_json = asreview_path.with_suffix(".metrics.json")
    if not out_json.exists():
        res = subprocess.run(
            ["asreview", "metrics", str(asreview_path), "-o", str(out_json),
             "--wss", "0.80", "0.90", "0.95"],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            print(f"  metrics FAILED for {asreview_path.name}: {res.stderr[-300:]}")
            return None
    return json.loads(out_json.read_text())["data"]["items"]


def _ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    if len(xs) == 1:
        return xs[0], None, 1
    return statistics.fmean(xs), statistics.stdev(xs), len(xs)


def main():
    files = sorted(ASREVIEW_DIR.glob("*.asreview"))
    if not files:
        print(f"No .asreview files in {ASREVIEW_DIR.relative_to(REPORT_DIR)}. Run `run_simulations_cli.py` first.")
        return
    print(f"Processing {len(files)} .asreview files...\n")

    rows = []
    for f in files:
        m = re.match(r"(\w+)__seed(\d+)\.asreview$", f.name)
        if not m:
            print(f"  skip (unrecognised name): {f.name}")
            continue
        condition = m.group(1)
        seed = int(m.group(2))
        print(f"  {condition} seed={seed} ... ", end="", flush=True)
        items = metrics_for_file(f)
        if items is None:
            print("FAILED")
            continue
        rows.append({
            "condition": condition,
            "seed": seed,
            "wss_80": extract_metric(items, "wss", 0.80),
            "wss_90": extract_metric(items, "wss", 0.90),
            "wss_95": extract_metric(items, "wss", 0.95),
            "loss":   extract_metric(items, "loss"),
            "atd":    extract_metric(items, "atd"),
        })
        print(f"WSS@95={rows[-1]['wss_95']:.4f} loss={rows[-1]['loss']:.4f} ATD={rows[-1]['atd']:.1f}")

    df = pd.DataFrame(rows)
    cond_rows = []
    for cond, sub in df.groupby("condition"):
        wss95_m, wss95_s, n = _ms(sub["wss_95"].tolist())
        wss80_m, wss80_s, _ = _ms(sub["wss_80"].tolist())
        wss90_m, wss90_s, _ = _ms(sub["wss_90"].tolist())
        loss_m,  loss_s,  _ = _ms(sub["loss"].tolist())
        atd_m,   atd_s,   _ = _ms(sub["atd"].tolist())
        cond_rows.append({
            "condition": cond, "n_seeds": n,
            "wss_80_mean": wss80_m, "wss_80_std": wss80_s,
            "wss_90_mean": wss90_m, "wss_90_std": wss90_s,
            "wss_95_mean": wss95_m, "wss_95_std": wss95_s,
            "loss_mean":   loss_m,  "loss_std":   loss_s,
            "atd_mean":    atd_m,   "atd_std":    atd_s,
            "_metrics_source": "asreview metrics CLI",
        })
    out_df = pd.DataFrame(cond_rows).sort_values("condition")
    out_df.to_csv(OUT_COND_CLI, index=False)
    print(f"\nwrote {OUT_COND_CLI.relative_to(REPORT_DIR)}")
    print(out_df.to_string(index=False))
    print(f"\n>>> Cross-check: diff this with summary_by_condition.csv (Python-API route).")
    print(f"    The numbers should match within rounding error.")


if __name__ == "__main__":
    main()
