# `data/SQ2/` — Term distribution + regex parquet

| File                              | Origin                                                                    |
| --------------------------------- | ------------------------------------------------------------------------- |
| `term_distribution.json`          | Frequency-weighted historical-term distribution derived from `data/SQ1/candidates_with_terms.csv`'s `found_via_terms` column. 28 historical terms, 2,831 total hits. Top 9: shell shock 400, traumatic neurosis 273, war neuroses 248, effort syndrome 217, war neurosis 185, battle fatigue 180, soldier's heart 173, combat fatigue 128, combat stress reaction 101. |
| `foras_regex_rewritten.parquet`*  | **Not committed** (regen-able). Created by `code/SQ2/build_regex_dataset.py` in ~30 seconds from `data/foras/PTSS_Data_Foras_2025-02-05.xlsx` + `term_distribution.json`. Every row contains both the original and the regex-rewritten title+abstract. |

*Listed in `.gitignore`.

## Used by

- `code/SQ2/build_regex_dataset.py` — reads `term_distribution.json` + the FORAS xlsx, writes the parquet.
- `code/SQ2/run_simulations_grid.py` — reads the parquet, applies (pp × nn) drift cells, runs simulations.
- `code/SQ2/make_heatmaps.py` — reads `outputs/SQ2/` summaries and renders the two heatmaps.

## Re-generating

```bash
python code/SQ2/build_regex_dataset.py
```

Output ends up at `data/SQ2/foras_regex_rewritten.parquet`. Deterministic given the seed pinned inside the script.
