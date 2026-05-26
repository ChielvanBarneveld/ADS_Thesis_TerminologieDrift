# `code/SQ1/` — Sentinel injection

Scripts for SQ1 (sentinel injection into FORAS + ELAS u4 simulations).

## Scripts

| Script | Inputs | Outputs | Purpose |
| ------ | ------ | ------- | ------- |
| `build_candidate_pool.py`  | `data/foras/PTSS_Data_Foras_2025-02-05.xlsx` + live OpenAlex API | `data/SQ1/candidates_with_terms.csv` + `data/SQ1/raw/*.jsonl` | Query OpenAlex for 29 historical PTSD terms × 2 pools (pre-1980 full-text, post-1980 title-only); cross-check each candidate's references against FORAS to compute edge-density. |
| `build_csvs_from_xlsx.py`  | `data/foras/PTSS_Data_Foras_2025-02-05.xlsx` + `data/SQ1/rewrites.json` | `outputs/SQ1/simulations/datasets/foras_xlsx_{baseline,with_sentinels_raw,_period,_full}.csv` | Convert FORAS xlsx → simulation-friendly CSVs; produce one baseline (FORAS only) and three sentinel-injected CSVs (one per rewrite condition). Sets `is_sentinel` / `sentinel_id` columns for per-sentinel tracking. |

## Run order (after a fresh clone)

1. `python code/SQ1/build_candidate_pool.py` — (optional; the committed `candidates_with_terms.csv` is the canonical 28 Apr 2026 snapshot)
2. _Sentinel selection + rewrite generation are added in subsequent commits (sections 3.4 of the methodology)._
3. `python code/SQ1/build_csvs_from_xlsx.py` — produces the four CSVs used by the simulation engine.

## Notes

- Both scripts use portable paths (`Path(__file__).resolve()` chain), so they run from any CWD as long as the repo layout is intact.
- `build_candidate_pool.py` accepts `--limit`, `--dry-run`, `--skip-existing`, `--no-fetch` for testing and incremental runs.
