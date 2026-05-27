"""Generate SQ1 sentinel rewrites in 3 conditions (raw/period/full) and build
simulation-ready CSVs.

Source-of-truth: Report/data/SQ1/rewrites_data.json (v2 superset, 6 sentinels x 3 conditions)
Filter v6->v3 happens in filter_v3_sentinels.py
(separated from this script so prose isn't subject to Python linter truncation).

Output:
  Report/data/SQ1/rewrites.json (v3 filtered, 3 sentinels x 3 conditions)   (validated + audit metadata)
  Report/outputs/SQ1/simulations/datasets/foras_xlsx_with_sentinels_{raw,period,full}.csv
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
# Portable paths: script self-discovers THESIS_ROOT and SQ1 folder.
# scripts/ -> SQ1/ -> (outputs OR Report) -> Thesis/
SCRIPT_DIR  = Path(__file__).resolve().parent
REPORT_DIR  = SCRIPT_DIR.parent.parent
DATA_SQ1    = REPORT_DIR / "data" / "SQ1"
DATA_IN     = DATA_SQ1 / "sentinels_v2_raw.json"   # output of select_sentinels.py
OUT_DATA    = DATA_SQ1 / "rewrites_data.json"      # rewrites for sentinels chosen by select_sentinels.py (v2 layer)


PTSD_BAN = re.compile(
    r"\b(post[\s\-]?traumatic|ptsd|posttraumatic stress|dsm[\-\s]?[iv]+)",
    re.IGNORECASE,
)

def banned(text):
    return PTSD_BAN.findall(text or "")

def wc(text):
    return len((text or "").split())

# Load data
data = json.loads(DATA_IN.read_text(encoding="utf-8"))
sentinels = data["sentinels"]
print("=" * 70)
print(f"SQ1 rewrite generator — {len(sentinels)} sentinels")
print("=" * 70)

# Validate + build records
out_records = []
for s in sentinels:
    print(f"\n[{s['id']}] {s['title'][:60]}... ({s['year']})")
    rec = {k: s[k] for k in ("id","openalex_id","title","year","bucket","era","n_edges_tiab","author_label","type","rewrite_notes")}
    rec["conditions"] = {}
    for cond in ("raw","period","full"):
        text = s[cond]
        hits = banned(text)
        # PTSD-tokens permitted ONLY in raw
        passed = (cond == "raw") or (len(hits) == 0)
        rec["conditions"][cond] = {
            "abstract":     text,
            "word_count":   wc(text),
            "banned_token_hits": hits,
            "passed":       passed,
        }
        print(f"  {cond:6} | wc={wc(text):4} | banned-hits={hits} | passed={passed}")
        if not passed:
            print(f"      *** VIOLATION: {cond} contains banned tokens ***")
    out_records.append(rec)

# Write master JSON
payload = {
    "generated_at":   datetime.now().isoformat(timespec="seconds"),
    "claim":          "claim_2_discovery_time_delta",
    "design":         data["_meta"]["design"],
    "amendment":      data["_meta"]["amendment_2026_05_12"],
    "n_sentinels":    len(out_records),
    "sentinels":      out_records,
}
MASTER.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nWrote {MASTER}")

# NOTE 12 May 2026: the CSV-merge step below was DEPRECATED. It used
# data/van_de_Schoot_2025.csv (combined FORAS+Synergy, n=14,764) which is
# NOT the FORAS-paper benchmark dataset. The simulation CSVs are now built
# by `build_csvs_from_xlsx.py` which reads PTSS_Data_Foras_2025-02-05.xlsx
# (n=10,594, 131 FT-positives).
#
# This script's purpose is now: validate rewrites + write rewrites_master.json
# only. To rebuild the simulation CSVs, run `build_csvs_from_xlsx.py` instead.
print("\nRewrite-validation done. To build simulation CSVs, run:")
print("    python3 outputs/SQ1/scripts/build_csvs_from_xlsx.py")
print("Done.")
