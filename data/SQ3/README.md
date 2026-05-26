# `data/SQ3/` — Citation-graph data (scaffold)

Empty for now. The citation graph for SQ3 will be built at runtime by `code/SQ3/build_citation_graph.py` (planned), which reads:

- `data/foras/PTSS_Data_Foras_2025-02-05.xlsx` (column `referenced_works`)
- `data/SQ1/candidates_master.csv` (for one-hop external edges via OpenAlex)

Intermediate files (node-feature matrix, edge index, train/val/test masks) will be cached in this folder. Once those are produced, this README is updated with their schemas.

`.gitkeep` is committed to keep the directory in the repo.
