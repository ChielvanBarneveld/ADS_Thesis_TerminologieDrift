"""SQ1 sentinel-injection simulations with the ELAS h3 HEAVY model (+ embedding cache).

Companion to ../SQ1/run_simulations.py (ELAS u4). Same data, same conditions,
same seeds, same metrics. The ONLY model change is ELAS h3 (heavy): u4's TF-IDF
feature extractor is swapped for the transformer embedder `mxbai` (SVM + Balanced
+ Max are unchanged). Purpose: show whether a stronger representation recovers the
historical elusive papers that u4/TF-IDF discovers late (supervisor meeting 11 Jun 2026).

KEY EFFICIENCY TRICK - embedding cache
--------------------------------------
mxbai embedding of the ~10.6k-doc corpus is the expensive step, and within one
condition the text is identical across seeds. So we embed each condition's corpus
ONCE (GPU if available), cache the matrix to `<OUT>/emb_cache/<condition>.npy`, and
every (condition, seed) simulation reuses it via a tiny CachedFeatures extractor.
=> 2 seeds cost the same embedding work as 1; re-runs skip embedding entirely.
Only the cache-build step needs asreview-dory + a GPU; the AL loop is plain sklearn.

Designed to run BOTH locally and on Kaggle (free GPU). Paths are overridable:
    DATA_DIR  : folder with the condition CSVs  (default: ../../outputs/SQ1/simulations/datasets)
    OUT_DIR   : folder for results              (default: ../../outputs/SQ1_heavy/simulations)
On Kaggle: set DATA_DIR=/kaggle/input/<your-dataset>, OUT_DIR=/kaggle/working/sq1_heavy. See KAGGLE_README.md.

Usage (env vars):
  MODEL=elas_h3 CONDS=baseline,raw,period,full SEEDS=2 python run_simulations_heavy.py
  CONDS=raw SEEDS=1 python run_simulations_heavy.py            # quick smoke test
  EMBED_ONLY=1 python run_simulations_heavy.py                 # only build the caches, no sims
  SEEDS_LIST=42,43 python run_simulations_heavy.py             # specific seeds

Requirements: asreview>=3.0 (always) + asreview-dory (only to BUILD a cache; not
needed if every condition already has a cached .npy).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from asreview import ActiveLearningCycle, Simulate
from asreview.models.models import get_ai_config

# -- model + paths --------------------------------------------------
MODEL_NAME = os.environ.get("MODEL", "elas_h3")  # heavy preset

try:
    SCRIPT_DIR = Path(__file__).resolve().parent      # Report/code/SQ1_heavy
except NameError:                                     # exec'd in a notebook (Kaggle): no __file__
    SCRIPT_DIR = Path.cwd()
REPORT_DIR = SCRIPT_DIR.parent.parent                 # Report/ (only used if DATA_DIR/OUT_DIR unset)

DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") \
    else REPORT_DIR / "outputs" / "SQ1" / "simulations" / "datasets"
SIM_DIR = Path(os.environ["OUT_DIR"]) if os.environ.get("OUT_DIR") \
    else REPORT_DIR / "outputs" / "SQ1_heavy" / "simulations"
RUNS_DIR = SIM_DIR / "runs"
CACHE_DIR = SIM_DIR / "emb_cache"
SUMMARY = SIM_DIR / "summary.json"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CFG = get_ai_config(MODEL_NAME)["value"]
ENGINE_LABEL = (
    f"asreview_3.0_{MODEL_NAME}_"
    f"{_CFG.classifier}_{_CFG.feature_extractor}_{_CFG.balancer}_{_CFG.querier}_nq1_cached"
)


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown (torch not importable)"


# -- embedding cache ------------------------------------------------
class CachedFeatures:
    """Minimal feature extractor that returns a precomputed embedding matrix.

    Plugged into ActiveLearningCycle in place of the real mxbai extractor, so the
    active-learning loop never re-embeds. fit_transform ignores the text content of
    X and returns the cached matrix; the row order must match (it does: the cache is
    built from the SAME dataframe, in the same order).
    """
    name = "cached_mxbai"

    def __init__(self, matrix):
        self.matrix = matrix

    def fit_transform(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        if n != self.matrix.shape[0]:
            raise RuntimeError(
                f"CachedFeatures row mismatch: X has {n} rows, cache has "
                f"{self.matrix.shape[0]}. Cache is stale - delete it and re-embed."
            )
        return self.matrix

    def get_params(self, deep=False):
        return {}


def get_or_build_cache(condition, df):
    """Return the mxbai embedding matrix for a condition, building+saving it once.

    Building needs asreview-dory (the mxbai extractor) and benefits hugely from a GPU.
    Once cached, subsequent seeds/re-runs load the .npy and never touch dory/GPU.
    """
    cache_path = CACHE_DIR / f"{condition}.npy"
    if cache_path.exists():
        M = np.load(cache_path)
        if M.shape[0] == len(df):
            print(f"  [{condition}] embedding cache HIT  {cache_path.name}  shape={M.shape}", flush=True)
            return M
        print(f"  [{condition}] cache shape {M.shape} != n_docs {len(df)} - rebuilding", flush=True)

    print(f"  [{condition}] embedding cache MISS - building with mxbai on {_device()} "
          f"(this is the slow part; once per condition)...", flush=True)
    from asreview.extensions import load_extension
    fe_cls = load_extension("models.feature_extractors", _CFG.feature_extractor)
    fe = fe_cls(**(_CFG.feature_extractor_param or {}))
    t0 = time.time()
    M = fe.fit_transform(df[["title", "abstract"]].fillna(""))
    M = np.asarray(M, dtype=np.float32)
    np.save(cache_path, M)
    print(f"  [{condition}] embedded {M.shape[0]} docs -> {M.shape[1]}d in {time.time()-t0:.0f}s, "
          f"cached to {cache_path.name}", flush=True)
    return M


def build_cycle(cache_matrix) -> ActiveLearningCycle:
    """ELAS h3 cycle, but with the feature extractor replaced by the cache.

    from_meta(skip_feature_extraction=True) builds querier/classifier/balancer from
    the elas_h3 config WITHOUT loading mxbai (so no dory needed at sim time); we then
    attach the precomputed embeddings. n_query forced to 1 (paper-by-paper, like u4).
    """
    cfg = get_ai_config(MODEL_NAME)["value"]
    cfg.n_query = 1
    cycle = ActiveLearningCycle.from_meta(cfg, skip_feature_extraction=True)
    cycle.feature_extractor = CachedFeatures(cache_matrix)
    return cycle


# -- Metrics - IDENTICAL to ../SQ1/run_simulations.py (do not diverge) ---------
def wss_at_recall(recall_curve, target, n_docs):
    hits = recall_curve[recall_curve["recall"] >= target]
    if hits.empty:
        return None
    step_at_r = int(hits.iloc[0]["step"])
    return round(float(target - step_at_r / n_docs), 4)


def loss(labels_in_order, n_docs):
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
    pos = np.where(labels_in_order == 1)[0] + 1
    return float(pos.mean()) if len(pos) else None


def simulate_once(df, condition, seed, cache_matrix):
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
    sim = Simulate(X=X, labels=labels, cycles=[build_cycle(cache_matrix)], print_progress=False)
    sim.label([prior_pos, prior_neg])
    print(f"  [{condition} seed={seed}]   priors labeled; review() (uses cached embeddings)...", flush=True)
    sim.review()
    elapsed = time.time() - t0
    print(f"  [{condition} seed={seed}]   review() done in {elapsed:.0f}s; metrics + files...", flush=True)

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
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nSet DATA_DIR to the folder with the 4 condition CSVs."
        )
    df = pd.read_csv(path, low_memory=False)
    df["abstract"]    = df["abstract"].fillna("")
    df["included"]    = df["included"].astype(int)
    df["is_sentinel"] = df["is_sentinel"].astype(bool)
    df["sentinel_id"] = df["sentinel_id"].fillna("")
    return df


def main():
    print(f"MODEL: {MODEL_NAME}  (type={get_ai_config(MODEL_NAME).get('type')})  device={_device()}")
    print(f"  feature_extractor = {_CFG.feature_extractor} {_CFG.feature_extractor_param}  (cached per condition)")
    print(f"  classifier        = {_CFG.classifier} {_CFG.classifier_param}")
    print(f"  DATA_DIR = {DATA_DIR}")
    print(f"  OUT_DIR  = {SIM_DIR}")
    print(f"  ENGINE   = {ENGINE_LABEL}\n")

    conditions = os.environ.get("CONDS", "baseline,raw,period,full").split(",")
    embed_only = os.environ.get("EMBED_ONLY", "") not in ("", "0", "false", "False")
    if "SEEDS_LIST" in os.environ:
        seeds = [int(s.strip()) for s in os.environ["SEEDS_LIST"].split(",")]
    else:
        n_seeds = int(os.environ.get("SEEDS", "2"))
        seeds = list(range(42, 42 + n_seeds))

    print(f"Conditions: {conditions}")
    print(f"Seeds:      {seeds}  (embed_only={embed_only})")
    print(f"Total sims: {0 if embed_only else len(conditions) * len(seeds)}\n")

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
        cache_matrix = get_or_build_cache(cond, df)   # built once per condition
        if embed_only:
            continue
        for seed in seeds:
            if (cond, seed) in done_keys:
                print(f"  [{cond} seed={seed}] skip (already done)")
                continue
            try:
                summary = simulate_once(df, cond, seed, cache_matrix)
            except Exception as e:
                import traceback
                print(f"  [{cond} seed={seed}] ERROR - sim failed: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                continue
            all_summaries.append(summary)
            tmp_path = SUMMARY.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps({"runs": all_summaries, "engine": ENGINE_LABEL}, indent=2))
            os.replace(tmp_path, SUMMARY)

    if embed_only:
        print(f"\nEMBED_ONLY done. Caches in {CACHE_DIR}")
    else:
        print(f"\nDone. Wrote {len(all_summaries)} run-summaries to {SUMMARY}")


if __name__ == "__main__":
    main()
