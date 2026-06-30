"""SQ2 drift-grid simulations with the ELAS h3 HEAVY model (+ two-version embedding cache).

Heavy counterpart of ../SQ2/run_simulations_grid.py (which uses ELAS u4). Same grid,
same per-cell drift logic, same seeds, same metrics. The ONLY change is the model:
ELAS h3 (SVM + mxbai transformer embeddings) instead of u4 (SVM + TF-IDF).

KEY EFFICIENCY TRICK
--------------------
In SQ2 every abstract has exactly two text versions: original and rewritten. We embed
BOTH versions of the whole corpus ONCE with mxbai (two passes, GPU), cache them, and
then assemble each trial's feature matrix by picking per paper the original- or
rewritten vector according to that trial's drift mask. So no re-embedding per trial;
only the active-learning loop (sklearn SVM, CPU) costs time per trial.

Drift selection, prior seeding and seeds are IDENTICAL to the u4 grid runner, so the
two models are directly comparable cell-by-cell.

Runs locally and on Kaggle. Overridable via env vars:
    DATA_DIR   : folder with foras_regex_rewritten.parquet (default: ../../data/SQ2)
    OUT_DIR    : results folder (default: ../../outputs/SQ2_heavy)
    POS_GRID, NEG_GRID : comma lists (default 0,5,10,20,50,100)
    N_TRIALS   : trials per cell (default 3)
    PRIOR_TRIALS : path to a trials.jsonl from a previous commit to resume from

Feasibility: full 6x6 grid x 3 trials = 108 sims ~ 27h of AL loop. A Kaggle commit
caps at 12h, so run it in stages (e.g. N_TRIALS=1 first = 36 sims ~ 9h = full heatmap
in one commit), then add trials in later commits by passing PRIOR_TRIALS. Resumable:
finished (pp, nn, trial) combos are skipped.

Requirements: asreview>=3.0 + asreview-dory (mxbai) + pyarrow (parquet).
"""
from __future__ import annotations
import json, os, time, shutil
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from asreview import ActiveLearningCycle, Simulate
from asreview.models.models import get_ai_config

MODEL_NAME = os.environ.get("MODEL", "elas_h3")

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()
REPORT_DIR = SCRIPT_DIR.parent.parent

DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else REPORT_DIR / "data" / "SQ2"
OUT_DIR  = Path(os.environ["OUT_DIR"]) if os.environ.get("OUT_DIR") else REPORT_DIR / "outputs" / "SQ2_heavy"
GRID_DIR = OUT_DIR / "grid_runs"
CACHE_DIR = OUT_DIR / "emb_cache"
FIG_DIR = OUT_DIR / "figures"
for d in (GRID_DIR, CACHE_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
OUT_TRIALS = GRID_DIR / "trials.jsonl"
OUT_SUMMARY = OUT_DIR / "grid_summary.csv"

DEFAULT_POS = [0, 5, 10, 20, 50, 100]
DEFAULT_NEG = [0, 5, 10, 20, 50, 100]
DEFAULT_N_TRIALS = 3

_CFG = get_ai_config(MODEL_NAME)["value"]
ENGINE = f"asreview_3.0_{MODEL_NAME}_{_CFG.classifier}_{_CFG.feature_extractor}_{_CFG.balancer}_{_CFG.querier}_nq1_cached"


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def _find_parquet():
    p = DATA_DIR / "foras_regex_rewritten.parquet"
    if p.exists():
        return p
    hits = [x for x in glob(str(DATA_DIR / "**" / "*.parquet"), recursive=True)]
    if not hits:
        hits = [x for x in glob("/kaggle/input/**/foras_regex_rewritten.parquet", recursive=True)]
    if not hits:
        raise FileNotFoundError(f"foras_regex_rewritten.parquet not found under {DATA_DIR}")
    return Path(hits[0])


# ── feature cache: embed original + rewritten once ───────────────
class MaskedFeatures:
    """Returns a precomputed matrix assembled per trial (drifted -> rewritten vec)."""
    name = "cached_mxbai_masked"
    def __init__(self, matrix): self.matrix = matrix
    def fit_transform(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        if n != self.matrix.shape[0]:
            raise RuntimeError(f"row mismatch: X={n}, matrix={self.matrix.shape[0]}")
        return self.matrix
    def get_params(self, deep=False): return {}


def build_two_version_cache(df):
    """Return (E_orig, E_rew): mxbai embeddings of original and rewritten corpus."""
    p_orig = CACHE_DIR / "emb_original.npy"
    p_rew = CACHE_DIR / "emb_rewritten.npy"
    if p_orig.exists() and p_rew.exists():
        E_o, E_r = np.load(p_orig), np.load(p_rew)
        if E_o.shape[0] == len(df) and E_r.shape[0] == len(df):
            print(f"  embedding cache HIT ({E_o.shape})", flush=True)
            return E_o, E_r
    print(f"  embedding cache MISS - building both versions with mxbai on {_device()} (one-time)...", flush=True)
    from asreview.extensions import load_extension
    fe_cls = load_extension("models.feature_extractors", _CFG.feature_extractor)

    def embed(title_col, abs_col, tag):
        fe = fe_cls(**(_CFG.feature_extractor_param or {}))
        X = pd.DataFrame({"title": df[title_col].fillna("").astype(str),
                          "abstract": df[abs_col].fillna("").astype(str)})
        t0 = time.time()
        M = np.asarray(fe.fit_transform(X), dtype=np.float32)
        print(f"    embedded {tag}: {M.shape} in {time.time()-t0:.0f}s", flush=True)
        return M

    E_o = embed("original_title", "original_abstract", "original")
    np.save(p_orig, E_o)
    E_r = embed("rewritten_title", "rewritten_abstract", "rewritten")
    np.save(p_rew, E_r)
    return E_o, E_r


# ── metrics (identical to u4 grid) ───────────────────────────────
def loss(labels_in_order, n_docs):
    n_pos = int(labels_in_order.sum()); n_trunc = len(labels_in_order)
    if n_pos == 0 or n_pos == n_docs:
        return None
    cum_pos = np.cumsum(labels_in_order)
    actual = float(cum_pos.sum()) + n_pos * (n_docs - n_trunc)
    optimal = n_pos * n_docs - n_pos * (n_pos - 1) / 2
    worst = n_pos * (n_pos + 1) / 2
    return float((optimal - actual) / (optimal - worst))


def atd(labels_in_order):
    pos = np.where(labels_in_order == 1)[0] + 1
    return float(pos.mean()) if len(pos) else None


def build_cycle(matrix):
    cfg = get_ai_config(MODEL_NAME)["value"]
    cfg.n_query = 1
    cyc = ActiveLearningCycle.from_meta(cfg, skip_feature_extraction=True)
    cyc.feature_extractor = MaskedFeatures(matrix)
    return cyc


def run_trial(df, E_orig, E_rew, pp, nn, trial):
    # --- replicate u4 grid drift + prior selection EXACTLY (same rng order) ---
    seed = pp * 10000 + nn * 100 + trial
    rng = np.random.default_rng(seed)
    pos_idx = df.index[df["included"] == 1].to_numpy()
    neg_idx = df.index[df["included"] == 0].to_numpy()
    n_pd = int(round(len(pos_idx) * pp / 100))
    n_nd = int(round(len(neg_idx) * nn / 100))
    pos_drift = set(rng.choice(pos_idx, size=n_pd, replace=False)) if n_pd else set()
    neg_drift = set(rng.choice(neg_idx, size=n_nd, replace=False)) if n_nd else set()
    drifted = np.array(df.index.isin(pos_drift | neg_drift))

    labels = df["included"].values.astype(int)
    pos = np.where(labels == 1)[0]; neg = np.where(labels == 0)[0]
    pos_nd = pos[~drifted[pos]]; neg_nd = neg[~drifted[neg]]
    pp_pos = int(rng.choice(pos_nd if len(pos_nd) else pos, 1)[0])
    pp_neg = int(rng.choice(neg_nd if len(neg_nd) else neg, 1)[0])

    # assemble feature matrix from the two-version cache
    M = np.where(drifted[:, None], E_rew, E_orig).astype(np.float32)
    X_text = pd.DataFrame({"title": df["original_title"], "abstract": df["original_abstract"]}).fillna("")

    t0 = time.time()
    sim = Simulate(X=X_text, labels=labels, cycles=[build_cycle(M)], print_progress=False)
    sim.label([pp_pos, pp_neg]); sim.review()
    elapsed = time.time() - t0

    order = sim._results["record_id"].astype(int).values
    lio = labels[order]; n = len(df); total_pos = int(labels.sum())
    cum = np.cumsum(lio); recall = cum / total_pos
    s95 = next((i for i, r in enumerate(recall, 1) if r >= 0.95), None)
    wss = round(0.95 - s95 / n, 4) if s95 else None
    drift_pos = np.where((labels == 1) & drifted)[0]
    step_of = {int(r): i for i, r in enumerate(order, start=1)}
    dsteps = [step_of[int(i)] for i in drift_pos if int(i) in step_of]
    return {"pp": pp, "nn": nn, "trial": trial, "seed": seed, "engine": ENGINE,
            "n_docs": n, "n_positives": total_pos,
            "n_drifted_pos": int(((labels == 1) & drifted).sum()),
            "wss_95": wss, "loss": loss(lio, n), "atd": atd(lio),
            "drift_median_step": float(np.median(dsteps)) if dsteps else None,
            "elapsed_seconds": round(elapsed, 1)}


def make_quicklook_heatmaps(rows, pos_grid, neg_grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    if df.empty:
        return
    for metric, fname in [("wss_95", "heatmap_wss95_heavy.png"), ("loss", "heatmap_loss_heavy.png")]:
        piv = df.groupby(["pp", "nn"])[metric].mean().reset_index().pivot(index="pp", columns="nn", values=metric)
        piv = piv.reindex(index=sorted(pos_grid), columns=sorted(neg_grid))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(piv.values, origin="lower", cmap="RdBu_r" if metric == "wss_95" else "RdBu", aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_xlabel("% negatives drifted"); ax.set_ylabel("% positives drifted")
        ax.set_title(f"{metric} - ELAS h3 (heavy)")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label=metric)
        plt.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {FIG_DIR / fname}")


def main():
    pos_grid = [int(x) for x in os.environ.get("POS_GRID", ",".join(map(str, DEFAULT_POS))).split(",")]
    neg_grid = [int(x) for x in os.environ.get("NEG_GRID", ",".join(map(str, DEFAULT_NEG))).split(",")]
    n_trials = int(os.environ.get("N_TRIALS", DEFAULT_N_TRIALS))
    print(f"MODEL={MODEL_NAME} device={_device()}  POS={pos_grid} NEG={neg_grid} N_TRIALS={n_trials}")
    print(f"DATA_DIR={DATA_DIR}  OUT_DIR={OUT_DIR}")
    total = len(pos_grid) * len(neg_grid) * n_trials
    print(f"total trials = {total}\n")

    # resume from a prior commit's trials.jsonl if provided/found
    if not OUT_TRIALS.exists():
        prior = os.environ.get("PRIOR_TRIALS") or next(iter(glob("/kaggle/input/**/trials.jsonl", recursive=True)), None)
        if prior and Path(prior).exists():
            shutil.copy(prior, OUT_TRIALS)
            print(f"resuming from prior trials: {prior}")

    pq = _find_parquet()
    df = pd.read_parquet(pq).reset_index(drop=True)
    print(f"loaded {pq.name}: {len(df)} rows, {int(df['included'].sum())} positives")

    E_orig, E_rew = build_two_version_cache(df)

    done = set()
    if OUT_TRIALS.exists():
        for line in OUT_TRIALS.read_text().splitlines():
            try:
                r = json.loads(line); done.add((r["pp"], r["nn"], r["trial"]))
            except Exception:
                pass
        print(f"already done: {len(done)} trials")

    n_run = 0
    with OUT_TRIALS.open("a") as fh:
        for pp in pos_grid:
            for nn in neg_grid:
                for trial in range(n_trials):
                    if (pp, nn, trial) in done:
                        continue
                    try:
                        res = run_trial(df, E_orig, E_rew, pp, nn, trial)
                    except Exception as e:
                        import traceback
                        print(f"  ERR pp={pp} nn={nn} t={trial}: {type(e).__name__}: {e}", flush=True)
                        traceback.print_exc(); continue
                    fh.write(json.dumps(res) + "\n"); fh.flush(); n_run += 1
                    print(f"[{n_run}] pp={pp:>3} nn={nn:>3} t={trial} wss95={res['wss_95']} loss={res['loss']:.4f} ATD={res['atd']:.1f} t={res['elapsed_seconds']}s", flush=True)

    rows = [json.loads(l) for l in OUT_TRIALS.read_text().splitlines() if l.strip()]
    g = pd.DataFrame(rows)
    summ = g.groupby(["pp", "nn"]).agg(
        n_trials=("trial", "count"), wss_95_mean=("wss_95", "mean"), wss_95_std=("wss_95", "std"),
        loss_mean=("loss", "mean"), loss_std=("loss", "std"), atd_mean=("atd", "mean"),
        drift_median_step_mean=("drift_median_step", "mean")).reset_index()
    summ.to_csv(OUT_SUMMARY, index=False)
    print(f"\nwrote {OUT_SUMMARY}  ({len(g)} trials total)")
    make_quicklook_heatmaps(rows, pos_grid, neg_grid)


if __name__ == "__main__":
    main()
