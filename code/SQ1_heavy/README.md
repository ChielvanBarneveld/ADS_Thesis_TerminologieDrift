# SQ1 — Heavy-model validation (ELAS h3)

**Epic:** A13 · **Added:** 2026-06-16 · **Motivation:** supervisor questions on 11 Jun 2026.

This folder re-runs the SQ1 sentinel-injection experiment with the **heavy**
ASReview model (ELAS h3) instead of the light **ELAS u4** used in the main
thesis (`../SQ1/`). The point is a fair, like-for-like check: *if the standard
light pipeline discovers the historical elusive papers late, does a stronger
feature representation recover them in time?*

It is deliberately a **separate folder with its own code and its own outputs**,
so the main SQ1 results stay untouched and the two models can be compared
directly.

## What is identical to u4, and what changes

| | ELAS u4 (`../SQ1/`) | ELAS h3 (here) |
|---|---|---|
| Datasets | `outputs/SQ1/simulations/datasets/*.csv` | **same files, reused** |
| Conditions | baseline / raw / period / full | same |
| Seeds | 42–51 (10 per condition) | same |
| Priors, n_query=1 | yes | yes |
| Metrics (WSS@95, Loss, ATD, TD) | yes | **same definitions** |
| Classifier / balancer / querier | SVM / Balanced / Max | SVM / Balanced / Max |
| **Feature extractor** | **TF-IDF** | **mxbai (transformer embeddings)** |

The only substantive change is the feature extractor. The `elas_h3` config is
read straight from `asreview.models.models.get_ai_config("elas_h3")`; the cycle
is built with `ActiveLearningCycle.from_meta(...)`, which loads `mxbai` via the
`asreview-dory` extension — so nothing here hardcodes model parameters.

## Embedding cache (important)

The runner embeds each condition's corpus **once** with mxbai and caches it to
`outputs/SQ1_heavy/simulations/emb_cache/<condition>.npy`; every seed (and every
re-run) reuses that cache via a small `CachedFeatures` extractor. So 2 seeds cost
the same embedding work as 1, and only the cache-build step needs `asreview-dory`
+ a GPU — the active-learning loop is plain sklearn (CPU) and identical to u4.

## Recommended: run on Kaggle (free GPU)

mxbai on CPU is hours per condition; on a Kaggle T4/P100 it is seconds–minutes.
**See `KAGGLE_README.md`** for the full step-by-step (upload the 4 CSVs + this
script as a Kaggle Dataset, enable GPU, paste 5 cells, download the zipped results,
then run `make_compare.py` locally).

## Requirements (if running locally / on the run-machine)

```
pip install "asreview>=3.0" asreview-dory
```

`asreview-dory` provides the `mxbai` feature extractor. The first build downloads
the mixedbread embedding model (~670 MB). On CPU this is slow (build the caches
once with `EMBED_ONLY=1`, ideally overnight or on GPU). A GPU is strongly recommended.

## How to run (Windows PowerShell)

The ASReview environment lives **outside OneDrive** at
`C:\Users\Chiel van Barneveld\.venvs\thesis\` (venvs break when synced). This is
the same env the u4 SQ1 sims ran in.

```powershell
# 1) activate the thesis env (lives outside OneDrive)
& "$env:USERPROFILE\.venvs\thesis\Scripts\Activate.ps1"

# 2) one-time: install the heavy feature extractor (mxbai). No-op if already present.
pip install asreview-dory

# 3) run: 4 conditions x 2 seeds (42,43) = 8 runs. Resumable (re-running skips done runs).
$env:CONDS="baseline,raw,period,full"; $env:SEEDS="2"
python "C:\Users\Chiel van Barneveld\OneDrive - Universiteit Utrecht\ADS\Thesis\Report\code\SQ1_heavy\run_simulations_heavy.py"

# 4) when done: build the u4-vs-h3 comparison table
python "C:\Users\Chiel van Barneveld\OneDrive - Universiteit Utrecht\ADS\Thesis\Report\code\SQ1_heavy\make_compare.py"
```

(For a single quick check first: `$env:CONDS="raw"; $env:SEEDS="1"` before step 3.)

Outputs go to `Report/outputs/SQ1_heavy/simulations/`:
`summary.json` (all runs) + `runs/seed_<NNN>__<cond>/` (results, recall curve,
per-sentinel discovery, per-run summary). The writer is atomic + resumable, so
the run can be stopped and continued safely (handy given OneDrive sync).

## Comparing against u4

Once `summary.json` has runs, build the side-by-side table:

```
python make_compare.py
```

This writes `Report/outputs/SQ1_heavy/u4_vs_h3_comparison.{csv,md}` with, per
condition, the mean WSS@95 / Loss / ATD for both models and the mean
time-to-discovery of each of the three historical elusive papers (Solomon 1988,
Kardiner 1947, Grinker 1944) — the headline number for the supervisor question.

## Files

- `run_simulations_heavy.py` — the heavy-model runner (mirror of `../SQ1/run_simulations.py`).
- `make_compare.py` — u4 vs h3 comparison table (pure pandas; no asreview needed).
- `README.md` — this file.
