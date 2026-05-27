"""SQ1 pre-registered sentinel selection — claim 2 (discovery-time delta).

Pool: data/SQ1/candidates_with_terms.csv
Rules locked 2026-05-12. v3 cohort-diversity filter applied AFTERWARDS by filter_v3_sentinels.py.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd
# Portable paths: script self-discovers THESIS_ROOT and SQ1 folder.
# scripts/ -> SQ1/ -> (outputs OR Report) -> Thesis/
SCRIPT_DIR  = Path(__file__).resolve().parent
REPORT_DIR  = SCRIPT_DIR.parent.parent
DATA_SQ1    = REPORT_DIR / "data" / "SQ1"
CANDS       = DATA_SQ1 / "candidates_with_terms.csv"
OUT         = DATA_SQ1
OUT.mkdir(parents=True, exist_ok=True)

PTSD_TOKEN_RE = re.compile(
    r"\b(post[\s\-]?traumatic|ptsd|posttraumatic stress|dsm[\-\s]?[iv]+)",
    re.IGNORECASE,
)

YEAR_CUTOFF  = 2005      # Jones-canon cutoff
N_PER_BUCKET = 2  # v2 selection: produces 6 sentinels. v3 cohort-diversity filter in filter_v3_sentinels.py reduces to 3.
HIGH_MIN     = 3         # >=3 reverse-edges to FORAS-TIAB-included
                          # MEDIAN = 1..2, LOW = 0

# --- load
master = pd.read_csv(CANDS, low_memory=False)
print(f"Loaded master pool: {len(master)} papers")

# --- E2: not in FORAS
m = master[master["in_foras"] != True].copy()
print(f"After E2 (not in FORAS): {len(m)}")

# --- E3: title-only PTSD-token check (claim 2: discovery-time delta)
m["title_has_ptsd"] = m["title"].fillna("").apply(lambda s: bool(PTSD_TOKEN_RE.search(str(s))))
m = m[~m["title_has_ptsd"]].copy()
print(f"After E3 (no PTSD-token in TITLE): {len(m)}")

# --- E4: title + abstract>=10 words
m["has_abstract"] = m["abstract"].apply(lambda s: isinstance(s,str) and len(s.split())>=10)
m = m[m["title"].notna() & m["has_abstract"]].copy()
print(f"After E4 (title+abstract>=10w): {len(m)}")

# --- E5: year <= YEAR_CUTOFF
m["year"] = pd.to_numeric(m["year"], errors="coerce")
m = m[m["year"].notna() & (m["year"] <= YEAR_CUTOFF)].copy()
m["year"] = m["year"].astype(int)
print(f"After E5 (year<={YEAR_CUTOFF}): {len(m)}")

# E7 (term-search filter) removed: candidates_with_terms.csv is already
# pure term-search output, so the filter is a no-op.

# --- E6: dedup near-duplicates (year + first 6 words of title)
def dedupe_key(row):
    t = re.sub(r"[^\w\s]", "", str(row["title"]).lower())
    return (int(row["year"]), " ".join(t.split()[:6]))
m["_dk"] = m.apply(dedupe_key, axis=1)
m = (m.sort_values(["n_cited_by_foras_tiab"], ascending=[False])
       .drop_duplicates(subset="_dk", keep="first")
       .drop(columns=["_dk"]))
print(f"After E6 (dedup): {len(m)}")

# Buckets
def bucket(n):
    n = int(n) if pd.notna(n) else 0
    if n == 0:        return "LOW"
    if n < HIGH_MIN:  return "MEDIAN"
    return "HIGH"
m["bucket"] = m["n_cited_by_foras_tiab"].apply(bucket)

def era(y):
    if y < 1945: return "pre-1945"
    if y < 1980: return "1945-1979"
    return "1980+"
m["era"] = m["year"].apply(era)

print(f"\nBucket counts: {m['bucket'].value_counts().to_dict()}")
print(f"Era counts:    {m['era'].value_counts().to_dict()}")

# Tiebreak: edges desc, sources desc, year asc, openalex_id asc
sort_cols = ["n_cited_by_foras_tiab","year","openalex_id"]
sort_asc  = [False, True, True]

shortlist = (m.sort_values(sort_cols, ascending=sort_asc)
              .groupby("bucket", as_index=False).head(25))
shortlist.to_csv(OUT / "sentinels_shortlist.csv", index=False)
print(f"Wrote shortlist_top25.csv ({len(shortlist)} rows)")

# Initial picks: top-2 per bucket
picks = (m.sort_values(sort_cols, ascending=sort_asc)
          .groupby("bucket", as_index=False).head(N_PER_BUCKET))

# Era-spread enforcement
required_eras = {"pre-1945","1945-1979","1980+"}
missing = required_eras - set(picks["era"])
for me in list(missing):
    era_pool = m[(m["era"] == me) & (~m["openalex_id"].isin(picks["openalex_id"]))] \
                  .sort_values(sort_cols, ascending=sort_asc)
    if era_pool.empty:
        print(f"  era {me} unavailable, skipping")
        continue
    cand = era_pool.iloc[0]
    # swap out lowest-priority pick from a bucket with 2 picks
    for b in ["LOW","MEDIAN","HIGH"]:
        bp = picks[picks["bucket"] == b]
        if len(bp) >= 2:
            to_drop = bp.iloc[-1]
            picks = picks[picks["openalex_id"] != to_drop["openalex_id"]]
            picks = pd.concat([picks, cand.to_frame().T], ignore_index=True)
            print(f"  swap {to_drop['openalex_id']} ({to_drop['era']}) -> {cand['openalex_id']} ({me}) [bucket {b} -> {cand['bucket']}]")
            break

picks = picks.sort_values(["bucket","n_cited_by_foras_tiab"], ascending=[True,False])
print(f"\nFinal picks (N={len(picks)}):")
print(picks[["bucket","year","era","n_cited_by_foras_tiab","title","openalex_id"]].to_string(index=False))

# Write JSON
final = []
for _, r in picks.iterrows():
    final.append({
        "openalex_id":            r["openalex_id"],
        "doi":                    r.get("doi"),
        "title":                  r["title"],
        "year":                   int(r["year"]),
        "bucket":                 r["bucket"],
        "era":                    r["era"],
        "n_cited_by_foras_tiab":  int(r["n_cited_by_foras_tiab"] or 0),
        "n_cited_by_foras_ft":    int(r.get("n_cited_by_foras_ft") or 0),
        "n_cited_by_foras_any":   int(r.get("n_cited_by_foras_any") or 0),
        # provenance columns (in_jones, in_termsearch, in_snowball, n_sources)
        # were available in the original candidates_master.csv but are not in
        # candidates_with_terms.csv (pure term-search pool).
    })

payload = {
    "selection_date":  "2026-05-12",
    "claim":           "claim_2_discovery_time_delta",
    "e3_mode":         "title_only",
    "year_cutoff":     YEAR_CUTOFF,
    "high_min":        HIGH_MIN,
    "rules_doc":       "select_sentinels.py (v2 selection) + filter_v3_sentinels.py (v3 cohort-diversity)",
    "sentinels":       final,
}
(OUT / "sentinels_v2_raw.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"\nWrote sentinels_v2_raw.json with {len(final)} sentinels (v2 set; run filter_v3_sentinels.py for v3 canonical)")
