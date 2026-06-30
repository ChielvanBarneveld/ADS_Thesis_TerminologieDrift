"""SQ1 sentinel-injection simulations via the asreview CLI (official .asreview route).

Parallel to run_simulations.py — uses the same elas_u4 config and identical
prior-selection logic (seed-matched), but invokes `asreview simulate ...` as a
subprocess so each run produces a canonical .asreview project file that
`asreview metrics` and `asreview plot` can read directly.

Why have both this and run_simulations.py?
- run_simulations.py (Python API): fast, gives us a custom JSON-summary with
  loss/WSS/ATD computed via formulas that match asreview-insights v1.6.1.
- run_simulations_cli.py (this file): slower (CLI startup ~5s/run), produces
  .asreview files that are the canonical reproducibility artefact. Used to
  cross-check the Python-API metrics against `asreview metrics` output.

Reads:  Report/outputs/SQ1/simulations/datasets/foras_xlsx_*.csv
Writes: Report/outputs/SQ1/simulations/asreview_files/<cond>__seed<NNN>.asreview

Defaults: 10 seeds per condition = 40 runs.
Override via env:
  CONDS=raw SEEDS=3 python run_simulations_cli.py
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
SIM_DIR    = REPORT_DIR / "outputs" / "SQ1" / "simulations"
DATA_DIR   = SIM_DIR / "datasets"
OUT_DIR    = SIM_DIR / "asreview_files"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_priors(df: pd.DataFrame, seed: int, condition: str) -> tuple[int, int]:
    """Pick one non-sentinel positive + one random negative, deterministic per seed.

    MATCHES the prior-selection logic in run_simulations.py so CLI runs are
    directly comparable to Python-API runs at the same seed.
    """
    labels = df["included"].values.astype(int)
    is_sent = df["is_sentinel"].astype(bool).values
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    pos_non_sent = pos_idx[~is_sent[pos_idx]] if condition != "baseline" else pos_idx
    prior_pos = int(rng.choice(pos_non_sent, 1, replace=False)[0])
    prior_neg = int(rng.choice(neg_idx, 1, replace=False)[0])
    return prior_pos, prior_neg


def run_one(condition: str, seed: int) -> Path:
    csv_path = DATA_DIR / ("foras_xlsx_baseline.csv" if condition == "baseline"
                            else f"foras_xlsx_with_sentinels_{condition}.csv")
    out_path = OUT_DIR / f"{condition}__seed{seed:03d}.asreview"
    if out_path.exists():
        print(f"  [{condition} seed={seed}] skip (already exists)")
        return out_path
    df = pd.read_csv(csv_path, low_memory=False)
    df["is_sentinel"] = df["is_sentinel"].astype(bool).fillna(False)
    prior_pos, prior_neg = compute_priors(df, seed, condition)

    cmd = [
        "asreview", "simulate", str(csv_path),
        "-o", str(out_path),
        "--ai", "elas_u4",
        "--prior-idx", str(prior_pos), str(prior_neg),
        "--seed", str(seed),
        "--n-query", "1",
    ]
    print(f"  [{condition} seed={seed}] priors=({prior_pos},{prior_neg})  RUN: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"    FAILED in {elapsed:.1f}s")
        print(f"    stderr (last 500): {res.stderr[-500:]}")
        return None
    print(f"    OK in {elapsed:.1f}s -> {out_path.relative_to(REPORT_DIR)}")
    return out_path


def main():
    conditions = os.environ.get("CONDS", "baseline,raw,period,full").split(",")
    n_seeds = int(os.environ.get("SEEDS", "10"))
    seeds = list(range(42, 42 + n_seeds))
    total = len(conditions) * len(seeds)
    print(f"Conditions: {conditions}")
    print(f"Seeds:      {seeds}")
    print(f"Total runs: {total}\n")

    ok = 0
    for cond in conditions:
        print(f"\n=== {cond.upper()} ===")
        for seed in seeds:
            result = run_one(cond, seed)
            if result is not None:
                ok += 1

    print(f"\nDone. {ok}/{total} runs succeeded.")
    print(f"Output dir: {OUT_DIR.relative_to(REPORT_DIR)}")
    print(f"\nNext: run `python make_summary_cli.py` to extract metrics via `asreview metrics`.")


if __name__ == "__main__":
    main()
