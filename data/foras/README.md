# `data/foras/` — FORAS-update raw corpus

| File                             | Source                                                      | Size  |
| -------------------------------- | ----------------------------------------------------------- | ----- |
| `PTSS_Data_Foras_2025-02-05.xlsx`| FORAS-update (van de Schoot et al., 2025), provided to the author Feb 2025 | ~9 MB |

## Schema

- **n = 10,594** records.
- Positive class used in this thesis: `label_included_FT` (full-text inclusion), matching the FORAS-paper §3.4 benchmark with **131 positives**.
- Used columns: `record_id`, `title`, `abstract`, `referenced_works`, `label_included_FT`, plus auxiliary metadata.

## Used by

- `code/SQ1/build_csvs_from_xlsx.py` — converts the xlsx to the simulation-friendly CSVs.
- `code/SQ2/build_regex_dataset.py` — reads the xlsx to generate the regex-rewritten parquet.
- `code/SQ3/build_citation_graph.py` *(scaffold)* — reads `referenced_works` to build the citation graph.

## Re-fetching

This file is not redistributable beyond the thesis project; contact the author for an updated snapshot.
