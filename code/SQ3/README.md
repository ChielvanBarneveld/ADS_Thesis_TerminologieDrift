# `code/SQ3/` — GNN on the FORAS citation network

Scaffold for SQ3 (Graph Neural Network experiments). No `.py` files yet — architecture choice (PPR / GraphSAGE / GAT / Node2Vec) is still open. See section 3.6 of `THESIS.md` for the planned design.

## Planned scripts

| Script | Purpose |
| ------ | ------- |
| `build_citation_graph.py`    | Build FORAS citation graph from `referenced_works` + OpenAlex one-hop edges |
| `train_gnn.py`               | Train chosen architecture on X% FORAS supervision |
| `evaluate_gnn.py`            | Compute WSS@95 / loss / ATD vs. ELAS u4 baseline |
