"""SQ1 sentinel-injection simulations with ELAS u4.

Pipeline:
  1. Read pre-built dataset CSVs from outputs/SQ1/simulations/datasets/
     (baseline, with_sentinels_raw/period/full).
  2. For each (condition, seed) in {baseline,raw,period,full} x {seeds},
     run a SINGLE-CYCLE AsReview simulation with the official elas_u4 config
     (SVM + TF-IDF + Balanced + Max, n_query=1 hardcoded for paper-by-paper).
  3. Per run, record:
       - review order (CSV)
       - recall curve  (CSV)
       - per-sentinel time-to-discovery
       - WSS@95, Loss, ATD  (asreview-insights formulas)
     into Report/outputs/SQ1/simulations/runs/seed_<NNN>__<cond>/
  4. Aggregate across seeds into summary.json.

Defaults: 10 seeds per condition (baseline + raw + period + full) => 40 total runs.
  (Reduced from 20 after observing ~18 min/run on FORAS — see SIMULATION_RUNBOOK.md.)

Usage (override via env vars):
  CONDS=baseline,raw,period,full SEEDS=10 python run_simulations.py
  CONDS=raw SEEDS=5 python run_simulations.py        # quick smoke test
  SEEDS_LIST=42,7,99 python run_simulations.py       # specific seeds

Required: asreview>=3.0.
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
SIM_DIR    = REPORT_DIR / "outputs" / "SQ1" / "simulations"
DATA_DIR   = SIM_DIR / "datasets"
RUNS_DIR   = SIM_DIR / "runs"
SUMMARY    = SIM_DIR / "summary.json"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

_ELAS_U4 = get_ai_config("elas_u4")["value"]
ENGINE_LABEL = "asreview_3.0_elas_u4_SVM_TFIDF_Balanced_Max_nq1"


def build_cycle() -> ActiveLearningCycle:
    """Build a single elas_u4 active-learning cycle. Fresh per simulation.

    n_query is hardcoded to 1 (paper-by-paper review, NOT batched). This
    matches the teacher's example and the standard FORAS-paper simulation
    protocol. All other parameters come from the official elas_u4 config.
    """
    return ActiveLearningCycle(
        querier=Max(**_ELAS_U4.querier_param),
        classifier=SVM(**_ELAS_U4.classifier_param),
        balancer=Balanced(**_ELAS_U4.balancer_param),
        feature_extractor=Tfidf(**_ELAS_U4.feature_extractor_param),
        n_query=1,
    )


# Metrics — definitions match asreview-insights v1.6.1 exactly.
def wss_at_recall(recall_curve, target, n_docs):
    """WSS@r (Cohen 2006) — matches asreview-insights wss().

    WSS@r = r - (step_at_r / N_total). CRITICAL: N must be total corpus size,
    NOT the truncated recall-curve length (asreview stops after the last
    positive is found, so recall_curve["step"].max() < total_docs).
    Higher is better.
    """
    hits = recall_curve[recall_curve["recall"] >= target]
    if hits.empty:
        return None
    step_at_r = int(hits.iloc[0]["step"])
    return round(float(target - step_at_r / n_docs), 4)


def loss(labels_in_order, n_docs):
    """Normalized Loss — matches asreview-insights insights.loss() exactly.

        Loss = (Optimal_AUC - Actual_AUC) / (Optimal_AUC - Worst_AUC)
        Optimal_AUC = Ny*Nx - Ny*(Ny-1)/2     (positives reviewed first)
        Worst_AUC   = Ny*(Ny+1)/2             (positives reviewed last)
        Actual_AUC  = Sum of cumulative recall over all N positions

    CRITICAL: N must be total corpus size (n_docs), NOT len(labels_in_order)
    — asreview truncates the simulation when the last positive is found, so
    `labels_in_order` is shorter than the full corpus. To get the correct Loss
    we extend the cumulative-positives count at n_pos for the remaining
    n_docs - n_truncated positions (which are all unreviewed negatives).
    Range: 0 (perfect) to 1 (worst). Lower is better.
    """
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
    """Average Time to Discovery — mean 1-indexed position of positives."""
    pos = np.where(labels_in_order == 1)[0] + 1
    return float(pos.mean()) if len(pos) else None


def simulate_once(df, condition, seed):
    """Run one elas_u4 simulation with given seed; write files; return summary."""
    run_dir = RUNS_DIR / f"seed_{seed:03d}__{condition}"
    run_dir.mkdir(parents=True, exist_ok=True)

    n = len(df)
    print(f"  [{condition} seed={seed}] START  n_docs={n}  total_pos={int(df['included'].sum())}", flush=True)

    X = df[["title", "abstract"]].fillna("")
    labels = df["included"].values.astype(int)
    is_sent = df["is_sentinel"].astype(bool).values
    sids = df["sentinel_id"].fillna("").values
    total_pos = int(labels.sum())

    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    pos_non_sent = pos_idx[~is_sent[pos_idx]] if condition != "baseline" else pos_idx
    if len(pos_non_sent) == 0:
        raise RuntimeError(f"No non-sentinel positive to seed prior in {condition}.")
    prior_pos = int(rng.choice(pos_non_sent, 1, replace=False)[0])
    prior_neg = int(rng.choice(neg_idx, 1, replace=False)[0])
    print(f"  [{condition} seed={seed}]   priors: pos={prior_pos} neg={prior_neg}", flush=True)

    t0 = time.time()
    sim = Simulate(X=X, labels=labels, cycles=[build_cycle()], print_progress=False)
    print(f"  [{condition} seed={seed}]   Simulate() built; labeling priors...", flush=True)
    sim.label([prior_pos, prior_neg])
    print(f"  [{condition} seed={seed}]   priors labeled; starting review() (this is the long part, no output during)...", flush=True)
    sim.review()
    elapsed = time.time() - t0
    print(f"  [{condition} seed={seed}]   review() done in {elapsed:.0f}s; computing metrics + writing files...", flush=True)

    results = sim._results
    results.to_csv(run_dir / "results.csv", index=False)
    order = results["record_id"].astype(int).values
    labels_in_order = labels[order]

    found = 0
    curve_rows = []
    for step, doc in enumerate(order, 1):
        if labels[doc] == 1:
            found += 1
        curve_rows.append((step, found, found / total_pos))
    recall_curve = pd.DataFrame(curve_rows, columns=["step", "n_found", "recall"])
    recall_curve.to_csv(run_dir / "recall_curve.csv", index=False)

    sent_positions = []
    for step, doc in enumerate(order, 1):
        if is_sent[doc]:
            sent_positions.append({
                "sentinel_id": sids[doc], "step": step,
                "percent_screened": step / n * 100,
                "recall_at_step": float(recall_curve.iloc[step - 1]["recall"]),
            })

    summary = {
        "condition": condition, "seed": seed, "engine": ENGINE_LABEL,
        "n_docs": n, "n_positives": total_pos, "n_sentinels": int(is_sent.sum()),
        "prior_pos": prior_pos, "prior_neg": prior_neg,
        "wss_80": wss_at_recall(recall_curve, 0.80, n),
        "wss_90": wss_at_recall(recall_curve, 0.90, n),
        "wss_95": wss_at_recall(recall_curve, 0.95, n),
        "steps_to_80": int(recall_curve[recall_curve["recall"] >= 0.80].iloc[0]["step"]) if (recall_curve["recall"] >= 0.80).any() else None,
        "steps_to_95": int(recall_curve[recall_curve["recall"] >= 0.95].iloc[0]["step"]) if (recall_curve["recall"] >= 0.95).any() else None,
        "loss": loss(labels_in_order, n),
        "atd": atd(labels_in_order),
        "sentinel_positions": sent_positions,
        "elapsed_seconds": round(elapsed, 1),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  [{condition} seed={seed}] WSS@95={summary['wss_95']} loss={summary['loss']:.4f} ATD={summary['atd']:.1f} t={elapsed:.1f}s")
    for sp in sent_positions:
        print(f"      sentinel {sp['sentinel_id']:>20s}  step {sp['step']:>5}  ({sp['percent_screened']:.2f}%)")
    return summary


def load_dataset(condition):
    path = DATA_DIR / ("foras_xlsx_baseline.csv" if condition == "baseline"
                       else f"foras_xlsx_with_sentinels_{condition}.csv")
    df = pd.read_csv(path, low_memory=False)
    df["abstract"]    = df["abstract"].fillna("")
    df["included"]    = df["included"].astype(int)
    df["is_sentinel"] = df["is_sentinel"].astype(bool)
    df["sentinel_id"] = df["sentinel_id"].fillna("")
    return df


def main():
    print(f"ELAS u4 config: classifier={_ELAS_U4.classifier} params={_ELAS_U4.classifier_param}")
    print(f"               feature_extractor={_ELAS_U4.feature_extractor} params={_ELAS_U4.feature_extractor_param}")
    print(f"               n_query (config)={_ELAS_U4.n_query}, HARDCODED to 1 in build_cycle()\n")

    conditions = os.environ.get("CONDS", "baseline,raw,period,full").split(",")
    if "SEEDS_LIST" in os.environ:
        seeds = [int(s.strip()) for s in os.environ["SEEDS_LIST"].split(",")]
    else:
        n_seeds = int(os.environ.get("SEEDS", "10"))
        seeds = list(range(42, 42 + n_seeds))

    print(f"Conditions: {conditions}")
    print(f"Seeds:      {seeds}")
    print(f"Total runs: {len(conditions) * len(seeds)}\n")

    all_summaries = []
    if SUMMARY.exists():
        try:
            all_summaries = json.loads(SUMMARY.read_text()).get("runs", [])
        except Exception:
            pass
    done_keys = {(r["condition"], r["seed"]) for r in all_summaries}

    for cond in conditions:
        df = load_dataset(cond)
        print(f"\n=== {cond.upper()} === n={len(df)} pos={int(df['included'].sum())}")
        for seed in seeds:
            if (cond, seed) in done_keys:
                print(f"  [{cond} seed={seed}] skip (already done)")
                continue
            try:
                summary = simulate_once(df, cond, seed)
            except Exception as e:
                import traceback
                print(f"  [{cond} seed={seed}] ERROR — sim failed: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                print(f"  [{cond} seed={seed}] continuing to next seed...", flush=True)
                continue
            all_summaries.append(summary)
            # Atomic write: write to .tmp, then rename. Prevents OneDrive
            # null-byte corruption that occurs when mid-write sync happens.
            tmp_path = SUMMARY.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps({"runs": all_summaries, "engine": ENGINE_LABEL}, indent=2))
            os.replace(tmp_path, SUMMARY)  # atomic rename (os is already imported at top of file)

    print(f"\nDone. Wrote {len(all_summaries)} run-summaries to {SUMMARY.relative_to(REPORT_DIR)}")


if __name__ == "__main__":
    main()
