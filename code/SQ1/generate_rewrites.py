"""DEPRECATED — not part of the reproducibility chain.

The three sentinel rewrites (raw / period / full) are hand-authored and frozen in
`data/SQ1/rewrites_data.json` (the v2 superset). The canonical v3 sentinels and
their rewrites are derived from that file by `filter_v3_sentinels.py`, which writes
`data/SQ1/sentinels.json` and `data/SQ1/rewrites.json`.

This earlier generator script is retained only as a historical marker; it is not
run by any step of the pipeline. See `code/SQ1/README.md` for the actual run order.
"""

raise SystemExit(
    "generate_rewrites.py is deprecated. The rewrites are hand-authored in "
    "data/SQ1/rewrites_data.json; run filter_v3_sentinels.py instead "
    "(see code/SQ1/README.md)."
)
