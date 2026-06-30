# `code/SQ1/` — Sentinel injection (ELAS u4)

Inject three historical elusive papers (Solomon 1988, Kardiner 1947, Grinker 1944)
into FORAS under four conditions (`baseline`, `raw`, `period`, `full`), screen each
with the ELAS u4 pipeline over 10 seeds (42–51), and compute WSS@95, normalized loss
and time-to-discovery. **40 simulations** in total.

## Scripts

| Script | Inputs | Outputs | Purpose |
| ------ | ------ | ------- | ------- |
| `build_candidate_pool.py` | FORAS xlsx + live OpenAlex API | `data/SQ1/candidates_with_terms.csv` | (Provenance) query OpenAlex for 29 historical PTSD terms × 2 pools. Committed CSV is the 28 Apr 2026 snapshot. |
| `filter_v3_sentinels.py` | `data/SQ1/rewrites_data.json` | `data/SQ1/sentinels.json`, `data/SQ1/rewrites.json` | (Provenance) derive the 3 canonical v3 sentinels + their raw/period/full rewrites from the hand-authored superset. |
| `build_csvs_from_xlsx.py` | FORAS xlsx + `data/SQ1/rewrites.json` | `outputs/SQ1/simulations/datasets/foras_xlsx_{baseline,with_sentinels_raw,_period,_full}.csv` | Build the baseline CSV (FORAS only) and three sentinel-injected CSVs (one per condition). |
| `run_simulations_cli.py` (or `run_simulations.py`) | the four CSVs | `outputs/SQ1/simulations/runs/seed_NNN__<cond>/` (`recall_curve.csv`, `results.csv`, `summary.json`) | 40 ELAS u4 simulations (config via `get_ai_config("elas_u4")`, `n_query=1`, seeds 42–51). |
| `make_summary_cli.py` (or `make_summary.py`) | the run dirs | `outputs/SQ1/simulations/summary.json` + `summary_by_condition.csv` | Aggregate WSS@95 / loss / TD across seeds. |
| `make_recall_plot.py` | run dirs | `outputs/SQ1/figures/recall_curves.png` | Figure 3.1 |
| `make_sentinel_discovery_figure.py` | run dirs | `outputs/SQ1/figures/sq1_sentinel_discovery.png` | Figure 3.2 |
| `make_loss_figure.py` | summary | `outputs/SQ1/figures/sq1_loss_by_condition.png` | loss-by-condition |
| `make_drift_figure.py` | `data/SQ1/candidates_with_terms.csv` | `outputs/SQ1/figures/terminology_timeline.png` | Figure 1 |

## Run order (fresh clone)

The committed `data/SQ1/` lets you start at step 3; steps 1–2 are provenance only.

1. *(optional)* `python code/SQ1/build_candidate_pool.py` — re-query OpenAlex.
2. *(optional)* `python code/SQ1/filter_v3_sentinels.py` — regenerate the committed `sentinels.json` + `rewrites.json`.
3. `python code/SQ1/build_csvs_from_xlsx.py` — build the four simulation CSVs.
4. `python code/SQ1/run_simulations_cli.py` — 40 ELAS u4 simulations (CPU; needs `asreview`).
5. `python code/SQ1/make_summary_cli.py` — aggregate metrics.
6. Figures: `make_recall_plot.py`, `make_sentinel_discovery_figure.py`, `make_loss_figure.py`, `make_drift_figure.py`.

> Steps 4–5 must have produced `outputs/SQ1/simulations/runs/` before the recall and
> discovery figures (6) can be drawn; `outputs/SQ1/` is gitignored and regenerated here.

## Notes

- All scripts use portable paths (`Path(__file__).resolve()` chain), so they run from any CWD as long as the repo layout is intact.
- `build_candidate_pool.py` accepts `--limit`, `--dry-run`, `--skip-existing`, `--no-fetch`.
- The WSS@95 and normalized-loss computations pass the full corpus size as the denominator (AsReview truncates the recall curve at the last positive); see the metric definitions in the thesis §2.2.
