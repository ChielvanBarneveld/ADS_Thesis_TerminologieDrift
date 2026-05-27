"""SQ2: run the 6x6x20 grid of regex-rewrite simulations.

Grid axes (SYMMETRIC -- fix vs old SQ2 sweep where y-axis maxed at 25):
    POS_GRID = [0, 5, 10, 20, 50, 100]   # % of positives rewritten
    NEG_GRID = [0, 5, 10, 20, 50, 100]   # % of negatives rewritten
    N_TRIALS = 5                          # seeds per (pp, nn) cell

Engine: ELAS u4 (SVM + TF-IDF + Balanced + Max) via get_ai_config.
n_query=1 HARDCODED for paper-by-paper review (not batched).

Reads:   Report/data/SQ2/foras_regex_rewritten.parquet
Writes:  Report/outputs/SQ2/grid_runs/trials.jsonl        (append-only, resumable)
         Report/outputs/SQ2/grid_summary.csv              (mean per cell)

Resumable: skips (pp, nn, trial) combos already in trials.jsonl.
Estimated runtime: ~18 min per trial x 180 trials ~ 54 hours (a long weekend).
Override defaults via env vars:
    POS_GRID=0,5,10 NEG_GRID=0,5,10 N_TRIALS=3 python run_simulations_grid.py
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from asreview import ActiveLearningCycle, Simulate
from asreview.models.balancers import Balanced
from asreview.models.classifiers import SVM
from asreview.models.feature_extractors import Tfidf
from asreview.models.models import get_ai_config
from asreview.models.queriers import Max

SCRIPT_DIR  = Path(__file__).resolve().parent
REPORT_DIR  = SCRIPT_DIR.parent.parent
SRC_PARQUET = REPORT_DIR / "data" / "SQ2" / "foras_regex_rewritten.parquet"
GRID_DIR    = REPORT_DIR / "outputs" / "SQ2" / "grid_runs"
GRID_DIR.mkdir(parents=True, exist_ok=True)
OUT_TRIALS  = GRID_DIR / "trials.jsonl"
OUT_SUMMARY = REPORT_DIR / "outputs" / "SQ2" / "grid_summary.csv"

DEFAULT_POS = [0, 5, 10, 20, 50, 100]
DEFAULT_NEG = [0, 5, 10, 20, 50, 100]
DEFAULT_N_TRIALS = 5

_ELAS_U4 = get_ai_config("elas_u4")["value"]


def build_cycle():
    """n_query=1 hardcoded: paper-by-paper, no batches."""
    return ActiveLearningCycle(
        querier=Max(**_ELAS_U4.querier_param),
        classifier=SVM(**_ELAS_U4.classifier_param),
        balancer=Balanced(**_ELAS_U4.balancer_param),
        feature_extractor=Tfidf(**_ELAS_U4.feature_extractor_param),
        n_query=1,
    )


def loss(labels_in_order, n_docs):
    """asreview-insights Loss formula (v1.6.1). Takes n_docs to compensate for
    AsReview truncating the simulation when the last positive is found —
    extending cum_pos at n_pos for remaining n_docs - n_truncated positions."""
    n_pos = int(labels_in_order.sum())
    n_trunc = len(labels_in_order)
    if n_pos == 0 or n_pos == n_docs:
        return None
    cum_pos = np.cumsum(labels_in_order)
    actual_auc = float(cum_pos.sum()) + n_pos * (n_docs - n_trunc)
    optimal_auc = n_pos * n_docs - n_pos * (n_pos - 1) / 2
    worst_auc = n_pos * (n_pos + 1) / 2
    return float((optimal_auc - actual_auc) / (optimal_auc - worst_auc))


def atd(labels_in_order):
    """Average Time to Discovery -- asreview-insights atd()."""
    pos = np.where(labels_in_order == 1)[0] + 1
    return float(pos.mean()) if len(pos) else None


def build_corpus(df_src, pos_pct, neg_pct, rng):
    df = df_src.copy()
    pos_idx = df.index[df["included"] == 1].to_numpy()
    neg_idx = df.index[df["included"] == 0].to_numpy()
    n_pd = int(round(len(pos_idx) * pos_pct / 100))
    n_nd = int(round(len(neg_idx) * neg_pct / 100))
    pos_drifted = set(rng.choice(pos_idx, size=n_pd, replace=False)) if n_pd else set()
    neg_drifted = set(rng.choice(neg_idx, size=n_nd, replace=False)) if n_nd else set()
    drifted = pos_drifted | neg_drifted
    df["is_drifted"] = df.index.isin(drifted)
    df["title"]    = np.where(df["is_drifted"], df["rewritten_title"],    df["original_title"]).astype(str)
    df["abstract"] = np.where(df["is_drifted"], df["rewritten_abstract"], df["original_abstract"]).astype(str)
    return df[["title", "abstract", "included", "is_drifted"]].copy()


def run_trial(df_src, pp, nn, trial):
    seed = pp * 10000 + nn * 100 + trial
    rng = np.random.default_rng(seed)
    df_mix = build_corpus(df_src, pp, nn, rng)
    labels = df_mix["included"].values.astype(int)
    is_drift = df_mix["is_drifted"].values
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    pos_nd = pos[~is_drift[pos]]
    neg_nd = neg[~is_drift[neg]]
    pp_pos = int(rng.choice(pos_nd if len(pos_nd) else pos, 1)[0])
    pp_neg = int(rng.choice(neg_nd if len(neg_nd) else neg, 1)[0])

    t0 = time.time()
    sim = Simulate(
        X=df_mix[["title", "abstract"]].fillna(""),
        labels=labels, cycles=[build_cycle()], print_progress=False,
    )
    sim.label([pp_pos, pp_neg])
    sim.review()
    elapsed = time.time() - t0

    order = sim._results["record_id"].astype(int).values
    labels_in_order = labels[order]
    n = len(df_mix)
    total_pos = int(labels.sum())

    cum_pos = np.cumsum(labels_in_order)
    recall = cum_pos / total_pos
    s95 = next((i for i, r in enumerate(recall, 1) if r >= 0.95), None)
    wss = round(0.95 - s95 / n, 4) if s95 else None
    loss_val = loss(labels_in_order, n)
    atd_val = atd(labels_in_order)

    drift_idx = np.where((labels == 1) & is_drift)[0]
    step_of = {int(r): i for i, r in enumerate(order, start=1)}
    drift_steps = [step_of[int(i)] for i in drift_idx if int(i) in step_of]
    drift_med = float(np.median(drift_steps)) if drift_steps else None

    return {
        "pp": pp, "nn": nn, "trial": trial, "seed": seed,
        "n_docs": n, "n_positives": total_pos,
        "n_drifted_pos": int(((labels == 1) & is_drift).sum()),
        "wss_95": wss, "loss": loss_val, "atd": atd_val,
        "drift_median_step": drift_med,
        "elapsed_seconds": round(elapsed, 1),
    }


def main():
    pos_grid = [int(x) for x in os.environ.get("POS_GRID", ",".join(map(str, DEFAULT_POS))).split(",")]
    neg_grid = [int(x) for x in os.environ.get("NEG_GRID", ",".join(map(str, DEFAULT_NEG))).split(",")]
    n_trials = int(os.environ.get("N_TRIALS", DEFAULT_N_TRIALS))
    print(f"POS_GRID = {pos_grid}")
    print(f"NEG_GRID = {neg_grid}")
    print(f"N_TRIALS = {n_trials}")
    total = len(pos_grid) * len(neg_grid) * n_trials
    print(f"total trials = {total}\n")

    df_src = pd.read_parquet(SRC_PARQUET)
    print(f"loaded {SRC_PARQUET.name}: {len(df_src)} rows")

    done = set()
    if OUT_TRIALS.exists():
        for line in OUT_TRIALS.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["pp"], r["nn"], r["trial"]))
            except Exception:
                pass
        print(f"resuming: {len(done)} trials already done\n")

    n_run = 0
    with OUT_TRIALS.open("a") as fh:
        for pp in pos_grid:
            for nn in neg_grid:
                for trial in range(n_trials):
                    if (pp, nn, trial) in done:
                        continue
                    res = run_trial(df_src, pp, nn, trial)
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    n_run += 1
                    print(f"[{n_run:>3}/{total - len(done)}] pp={pp:>3} nn={nn:>3} trial={trial:>2}  wss95={res['wss_95']}  loss={res['loss']:.4f}  ATD={res['atd']:.1f}  t={res['elapsed_seconds']}s")

    rows = [json.loads(l) for l in OUT_TRIALS.read_text().splitlines()]
    grid_df = pd.DataFrame(rows)
    if "loss" not in grid_df.columns and "normalized_loss" in grid_df.columns:
        grid_df["loss"] = grid_df["normalized_loss"]
    agg_kwargs = dict(
        n_trials=("trial", "count"),
        wss_95_mean=("wss_95", "mean"),
        wss_95_std=("wss_95", "std"),
        loss_mean=("loss", "mean"),
        loss_std=("loss", "std"),
        drift_median_step_mean=("drift_median_step", "mean"),
    )
    if "atd" in grid_df.columns:
        agg_kwargs["atd_mean"] = ("atd", "mean")
    summary = grid_df.groupby(["pp", "nn"]).agg(**agg_kwargs).reset_index()
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"\nwrote {OUT_SUMMARY.relative_to(REPORT_DIR)}")


if __name__ == "__main__":
    main()
