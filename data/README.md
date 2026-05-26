# `data/`

All input data needed by the scripts in `code/`. Subdivided into the same SQ folders so it is obvious which data each script consumes.

| Folder    | What                                                                                   |
| --------- | -------------------------------------------------------------------------------------- |
| `foras/`  | FORAS-update raw corpus (xlsx, n = 10,594)                                              |
| `SQ1/`    | Historical candidate pool + sentinel papers + hand-curated rewrites                     |
| `SQ2/`    | Term-frequency JSON used to weight regex replacements (parquet itself is regen-able)    |
| `SQ3/`    | (empty — citation-graph data is built at runtime by `code/SQ3/build_citation_graph.py`) |

## Provenance summary

Every file in this tree comes with a short note in the local `README.md`: where it came from, who generated it, and when. If a file is the *output* of a script in this repo, the README points to the script.
