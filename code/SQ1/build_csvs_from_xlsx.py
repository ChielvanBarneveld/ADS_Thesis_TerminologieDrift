"""Build the simulation CSVs from FORAS-update xlsx + sentinel rewrites.

Reads:
  Report/data/foras/PTSS_Data_Foras_2025-02-05.xlsx   (n=10,594, 131 FT-positive)
  Report/data/SQ1/rewrites.json                       (3 sentinels x 3 conditions, v3)

Writes:
  Report/outputs/SQ1/simulations/datasets/
    foras_xlsx_baseline.csv             — FORAS-update only
    foras_xlsx_with_sentinels_raw.csv   — + 3 sentinels (raw abstracts, v3)
    foras_xlsx_with_sentinels_period.csv
    foras_xlsx_with_sentinels_full.csv

Positive label = label_included_FT (matches FORAS-paper §3.4 benchmark).
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import pandas as pd

# Portable paths: code/SQ1/this.py -> code/ -> Report/
SCRIPT_DIR  = Path(__file__).resolve().parent
REPORT_DIR  = SCRIPT_DIR.parent.parent
XLSX        = REPORT_DIR / "data" / "foras" / "PTSS_Data_Foras_2025-02-05.xlsx"
REWRITES    = REPORT_DIR / "data" / "SQ1" / "rewrites.json"
OUT         = REPORT_DIR / "outputs" / "SQ1" / "simulations" / "datasets"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_excel(XLSX)
    print(f"loaded xlsx: {len(df)} records")
    df["included"] = df["label_included_FT"].fillna(0).astype(int)
    df = df.dropna(subset=["title"]).copy()
    df["abstract"] = df["abstract"].fillna("")
    print(f"after dropna(title): {len(df)} records, positives (FT)={df['included'].sum()}")

    base = df[["title", "abstract", "included"]].copy()
    base["is_sentinel"] = False
    base["sentinel_id"] = ""

    out_baseline = OUT / "foras_xlsx_baseline.csv"
    base.to_csv(out_baseline, index=False, quoting=csv.QUOTE_ALL)
    print(f"wrote {out_baseline.name}: {len(base)} rows, pos={base['included'].sum()}")

    rewrites = json.loads(REWRITES.read_text())["sentinels"]
    for cond in ("raw", "period", "full"):
        sent_rows = []
        for s in rewrites:
            sent_rows.append({
                "title":        s["title"],
                "abstract":     s[cond] if isinstance(s.get(cond), str) else s["conditions"][cond]["abstract"],
                "included":     1,
                "is_sentinel":  True,
                "sentinel_id":  s["id"],
            })
        sdf = pd.DataFrame(sent_rows)
        out = pd.concat([base, sdf], ignore_index=True)
        out_path = OUT / f"foras_xlsx_with_sentinels_{cond}.csv"
        out.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"wrote {out_path.name}: {len(out)} rows, pos={out['included'].sum()} (+{len(sent_rows)} sentinels)")


if __name__ == "__main__":
    main()
