# `code/SQ2_heavy/` — SQ2 drift grid with ELAS h3 (GPU)

The SQ2 regex drift grid (§2.3 robustness check, §3.3) re-run with the **ELAS h3**
model (mxbai transformer embeddings) instead of TF-IDF, everything else identical.
This needs a GPU and was run on Kaggle; its outputs are committed under
`outputs/SQ2_heavy/` because a fresh CPU clone cannot regenerate them.

## Scripts

| Script | Inputs | Outputs | Purpose |
| ------ | ------ | ------- | ------- |
| `run_simulations_grid_heavy.py` | `data/SQ2/foras_regex_rewritten.parquet` | `outputs/SQ2_heavy/grid_summary.csv` + per-cell `trials.jsonl` (+ rough quicklook heatmaps) | Sweep the 6×6 drift grid once per cell (single trial) under ELAS h3, caching mxbai embeddings per condition. |
| `make_heatmaps_heavy.py` | `grid_summary.csv` | `outputs/SQ2_heavy/figures/heatmap_grid_heavy.png` | Figure 3.7 (WSS@95 + normalized loss, two panels) |

## Run order (GPU, e.g. Kaggle T4)

1. `python code/SQ2/build_regex_dataset.py` — build the parquet (CPU, ~30 s).
2. `python code/SQ2_heavy/run_simulations_grid_heavy.py` — 36 ELAS h3 simulations (GPU).
3. `python code/SQ2_heavy/make_heatmaps_heavy.py` — render Figure 3.7.

## Notes

- Single trial per cell (the light u4 grid uses five), so the per-cell values are single-trial estimates (thesis §3.3).
- Grid seeding matches the light runner (`seed = pp*10000 + nn*100 + trial`); embeddings are cached per condition (`emb_cache/`, gitignored, hundreds of MB).
- Config via `get_ai_config("elas_h3")`: classifier (SVM), balancer (Balanced) and querier (Max) are unchanged from u4; only the feature extractor is swapped to mxbai.
