# `code/SQ2/` — Regex drift sweep (ELAS u4)

Replace modern PTSD tokens with historical variants by regex, then sweep a 6×6
drift-prevalence grid (`pp` × `nn` ∈ {0, 5, 10, 20, 50, 100}, 5 trials per cell =
**180 ELAS u4 simulations**) and render the two heatmaps.

## Scripts

| Script | Inputs | Outputs | Purpose |
| ------ | ------ | ------- | ------- |
| `build_regex_dataset.py` | FORAS xlsx + `data/SQ1/candidates_with_terms.csv` | `data/SQ2/foras_regex_rewritten.parquet` (gitignored, ~30 s) + `data/SQ2/term_distribution.json` | Draw one historical term per paper (seeded, `default_rng(42)`) and regex-replace every PTSD token; build the frequency-weighted term distribution. |
| `make_term_distribution_figure.py` | `term_distribution.json` | `outputs/SQ2/figures/term_distribution.png` | Figure 3.3 |
| `run_simulations_grid.py` | the parquet | `outputs/SQ2/simulations/cell_pp{pp}_nn{nn}_seed{s}/summary.json` + `grid_summary.csv` | 180 ELAS u4 simulations over the grid (config via `get_ai_config("elas_u4")`, `n_query=1`; resumable via `trials.jsonl`). |
| `make_heatmaps.py` | `grid_summary.csv` | `outputs/SQ2/figures/heatmap_grid.png` | Figure 3.4 (WSS@95 + normalized loss, two panels) |

## Run order (fresh clone)

1. `python code/SQ2/build_regex_dataset.py` — regenerate the parquet + term distribution (deterministic).
2. `python code/SQ2/make_term_distribution_figure.py`
3. `python code/SQ2/run_simulations_grid.py` — 180 ELAS u4 simulations (CPU; needs `asreview`).
4. `python code/SQ2/make_heatmaps.py`

## Notes

- Portable paths (`Path(__file__)` chain); run from any CWD.
- The per-paper term draw is seeded (`numpy.random.default_rng(42)`), so the parquet is byte-reproducible. The parquet is gitignored and rebuilt by step 1.
- The drift selection and prior bootstrap in the grid runner are seeded per cell/trial (`seed = pp*10000 + nn*100 + trial`), so every cell is reproducible.
