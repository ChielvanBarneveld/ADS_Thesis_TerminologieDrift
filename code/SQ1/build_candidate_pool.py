"""Build the historical-terminology candidate pool via OpenAlex.

Pipeline:
  1. For every term in HISTORICAL_TERMS, query OpenAlex twice:
     - Pool A (pre-1980):  term in title OR abstract  (search=...)
     - Pool B (post-1980): term in title only         (filter=title.search:...)
  2. Cursor-paginate through all results, save raw JSONL per (term, pool)
     under data/SQ1/raw/.
  3. Aggregate, dedup on OpenAlex ID.
  4. Cross-check each candidate's referenced_works against the FORAS-update
     paper IDs to compute n_edges_to_foras_FT_included and *_TIAB_included
     (FORWARD edges: candidate -> FORAS).
  5. Batch-fetch FORAS papers' own referenced_works from OpenAlex and
     compute n_cited_by_foras_{any,tiab,ft} per candidate
     (REVERSE edges: FORAS -> candidate). These are the columns that drive
     the LOW / MEDIAN / HIGH bucket stratification in select_sentinels.py.
  6. Write data/SQ1/candidates_with_terms.csv.

This is the single source of `data/SQ1/candidates_with_terms.csv`. Both the
SQ1 sentinel-selection (`select_sentinels.py`) and the SQ2 regex-distribution
(`code/SQ2/build_regex_dataset.py`) read this file - no other pool feeds the
pipeline post-pivot.

Run:
  python code/SQ1/build_candidate_pool.py
  python code/SQ1/build_candidate_pool.py --limit 5             (testing, caps per (term, pool))
  python code/SQ1/build_candidate_pool.py --dry-run             (print URLs, no requests)
  python code/SQ1/build_candidate_pool.py --skip-existing       (resume from cached raw/)
  python code/SQ1/build_candidate_pool.py --no-fetch            (aggregate from cache only)
  python code/SQ1/build_candidate_pool.py --reverse-edges-only  (skip term-search; just
                                                                 retrofit reverse-edges to
                                                                 the existing CSV)

History:
  - Original script: `find_historical_candidates.py` (queried OpenAlex on
    28 April 2026; produced `data/historical-terminology/candidates.csv`).
  - A later builder, `build_candidate_pool.py` (12 May 2026), tried to
    UNION the term-search pool with a Jones-canon list and a FORAS-snowball
    set into `candidates_master.csv`. After the 21 May scope-pivot the
    snowball and Jones components were dropped (they leaked methodology
    false-positives), and the active pipeline reverted to term-search-only.
    The committed `candidates_with_terms.csv` is the 28 April 2026 snapshot
    used by the thesis.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

# --- portable paths -----------------------------------------------------
# code/SQ1/this.py -> code/ -> <repo root>
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
SRC_FORAS  = REPO_ROOT / "data" / "foras" / "PTSS_Data_Foras_2025-02-05.xlsx"
OUT_DIR    = REPO_ROOT / "data" / "SQ1"
RAW_DIR    = OUT_DIR / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV    = OUT_DIR / "candidates_with_terms.csv"

OPENALEX_BASE = "https://api.openalex.org/works"
MAILTO        = "chiel.vbarneveld@gmail.com"   # polite-pool identifier

ID_RE = re.compile(r"w\d+", re.IGNORECASE)

# --- 29 historical terms (English, German, French) ----------------------
# Captures military, civilian-clinical, and foreign-language variants.
HISTORICAL_TERMS: list[str] = [
    # English military
    '"shell shock"',
    '"shell shocked"',
    '"shell concussion"',
    "\"soldier's heart\"",
    '"soldiers heart"',
    '"war neurosis"',
    '"war neuroses"',
    '"battle fatigue"',
    '"battle exhaustion"',
    '"combat stress reaction"',
    '"combat fatigue"',
    '"combat neurosis"',
    '"war hysteria"',
    '"operational fatigue"',
    '"operational exhaustion"',
    '"post-Vietnam syndrome"',
    # English broader / older clinical
    '"traumatic neurosis"',
    "\"Da Costa's syndrome\"",
    '"Da Costa syndrome"',
    '"effort syndrome"',
    '"railway spine"',
    '"gross stress reaction"',
    '"transient situational disturbance"',
    '"disordered action of the heart"',
    # Non-English (German, French)
    '"Granatschock"',
    '"Kriegsneurose"',
    '"obusite"',
    "\"nevrose de guerre\"",
    "\"nevrose traumatique\"",
]


# --- helpers ------------------------------------------------------------
def reconstruct_abstract(inverted):
    """OpenAlex returns abstracts as inverted indices; reassemble."""
    if not inverted:
        return ""
    pos = {}
    for word, positions in inverted.items():
        for p in positions:
            pos[p] = word
    if not pos:
        return ""
    n = max(pos) + 1
    return " ".join(pos.get(i, "") for i in range(n)).strip()


def short_id(openalex_url):
    """Extract bare OpenAlex W-id from a URL (or raw id)."""
    if not openalex_url:
        return None
    m = ID_RE.search(openalex_url)
    return m.group(0).upper() if m else None


def build_url(term, pool, cursor="*"):
    if pool == "pre1980":
        params = {
            "search":   term,
            "filter":   "publication_year:<1980",
            "per-page": 200,
            "cursor":   cursor,
            "select":   "id,doi,title,publication_year,abstract_inverted_index,referenced_works,language,type",
            "mailto":   MAILTO,
        }
    elif pool == "post1980_title":
        params = {
            "filter":   f"title.search:{term},publication_year:>1979",
            "per-page": 200,
            "cursor":   cursor,
            "select":   "id,doi,title,publication_year,abstract_inverted_index,referenced_works,language,type",
            "mailto":   MAILTO,
        }
    else:
        raise ValueError(f"Unknown pool: {pool}")
    safe = ':,*"'
    return f"{OPENALEX_BASE}?{urlencode(params, safe=safe)}"


def query_term_pool(term, pool, limit=None, dry_run=False, raw_path=None):
    out = []
    cursor = "*"
    page = 0
    while True:
        page += 1
        url = build_url(term, pool, cursor=cursor)
        if dry_run:
            print(f"[DRY] {url}")
            return []
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} for {term} {pool}: {resp.text[:200]}")
            break
        d = resp.json()
        results = d.get("results", [])
        out.extend(results)
        if raw_path is not None:
            with open(raw_path, "a", encoding="utf-8") as f:
                for w in results:
                    f.write(json.dumps(w, ensure_ascii=False) + "\n")
        meta = d.get("meta") or {}
        next_cursor = meta.get("next_cursor")
        total = meta.get("count")
        if page == 1:
            print(f"  [{term} | {pool}] total={total}, fetching...")
        if not results or not next_cursor:
            break
        cursor = next_cursor
        if limit is not None and len(out) >= limit:
            out = out[:limit]
            break
        time.sleep(0.1)
    return out


def load_foras_index():
    """Build a lookup from FORAS-update openalex_id -> {label_FT, label_TIAB}.

    The xlsx column `openalex_id` is the URL form `https://openalex.org/W...`.
    """
    df = pd.read_excel(SRC_FORAS)
    df["_pid"] = df["openalex_id"].fillna("").apply(short_id)
    df = df[df["_pid"].notna() & (df["_pid"] != "")]
    idx = {}
    for _, r in df.iterrows():
        idx[r["_pid"]] = {
            "label_included_FT":   r.get("label_included_FT"),
            "label_included_TIAB": r.get("label_included_TIAB"),
        }
    return set(idx.keys()), idx


def fetch_foras_referenced_works(foras_pids, batch_size=100, cache_path=None):
    """Fetch the referenced_works list for every FORAS paper via OpenAlex.

    Batches by |-joined openalex_id (OpenAlex caps at ~100 ids per filter).
    The FORAS-update xlsx does NOT carry referenced_works, so we re-derive
    them from the OpenAlex API. Result is cached to data/SQ1/foras_references_cache.jsonl
    so subsequent runs can skip the network round-trip.

    Returns: {foras_pid: [referenced_W_id, ...]}.
    """
    refs = {}

    # Cache hit?
    if cache_path is not None and Path(cache_path).exists():
        print(f"  Reading reference cache from {Path(cache_path).name}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    refs[rec["pid"]] = rec["referenced_works"]
                except (json.JSONDecodeError, KeyError):
                    continue
        if set(foras_pids).issubset(set(refs.keys())):
            print(f"  All {len(foras_pids)} FORAS-pids covered by cache.")
            return refs
        missing = set(foras_pids) - set(refs.keys())
        print(f"  Cache covers {len(refs)}/{len(foras_pids)}; fetching {len(missing)} more...")
    else:
        missing = set(foras_pids)
        print(f"  No cache; fetching all {len(missing)} FORAS-paper references...")

    pid_list = sorted(missing)
    t0 = time.time()
    n_batches = (len(pid_list) + batch_size - 1) // batch_size
    cache_handle = open(cache_path, "a", encoding="utf-8") if cache_path is not None else None
    try:
        for i in range(0, len(pid_list), batch_size):
            batch = pid_list[i:i + batch_size]
            fid = "|".join(batch)
            params = {
                "filter":   f"openalex_id:{fid}",
                "per-page": len(batch),
                "select":   "id,referenced_works",
                "mailto":   MAILTO,
            }
            url = f"{OPENALEX_BASE}?{urlencode(params, safe=':,*|')}"
            results = []
            for attempt in range(3):
                try:
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200:
                        results = r.json().get("results", [])
                        break
                    time.sleep(1.5 ** attempt)
                except requests.RequestException:
                    time.sleep(1.5 ** attempt)
            for w in results:
                pid = short_id(w.get("id"))
                if not pid:
                    continue
                rw = [short_id(x) for x in (w.get("referenced_works") or []) if short_id(x)]
                refs[pid] = rw
                if cache_handle is not None:
                    cache_handle.write(json.dumps({"pid": pid, "referenced_works": rw}, ensure_ascii=False) + "\n")
            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(pid_list):
                elapsed = time.time() - t0
                pct = 100 * (i + batch_size) / max(1, len(pid_list))
                print(f"    batch {i // batch_size + 1}/{n_batches} ({elapsed:.1f}s, {pct:.0f}%)")
            time.sleep(0.10)
    finally:
        if cache_handle is not None:
            cache_handle.close()
    return refs


def compute_reverse_edges(refs_by_foras_pid, foras_idx):
    """For each W-id referenced by any FORAS paper, count how many FORAS-{any,TIAB,FT}
    papers cite it. Returns (any, tiab, ft) dicts keyed by referenced W-id."""
    rev_any  = {}
    rev_tiab = {}
    rev_ft   = {}
    for foras_pid, rw in refs_by_foras_pid.items():
        meta = foras_idx.get(foras_pid, {})
        is_tiab = meta.get("label_included_TIAB") in (1, 1.0, "1")
        is_ft   = meta.get("label_included_FT")   in (1, 1.0, "1")
        for w in rw:
            rev_any[w] = rev_any.get(w, 0) + 1
            if is_tiab:
                rev_tiab[w] = rev_tiab.get(w, 0) + 1
            if is_ft:
                rev_ft[w] = rev_ft.get(w, 0) + 1
    return rev_any, rev_tiab, rev_ft


# --- main ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap results per (term, pool). For testing.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print URLs without making requests.")
    ap.add_argument("--terms", nargs="+", default=None,
                    help="Override default term list (each must be a phrase).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip (term, pool) combos where raw JSONL already exists and is non-empty.")
    ap.add_argument("--no-fetch", action="store_true",
                    help="Skip all OpenAlex term-search queries; aggregate from existing raw files only.")
    ap.add_argument("--reverse-edges-only", action="store_true",
                    help="Retrofit reverse-edge columns onto the existing candidates_with_terms.csv "
                         "without re-running the term-search step.")
    ap.add_argument("--skip-reverse-edges", action="store_true",
                    help="Skip the reverse-edge computation (saves ~1 min of OpenAlex calls). "
                         "n_cited_by_foras_* columns will be 0.")
    args = ap.parse_args()

    terms = args.terms or HISTORICAL_TERMS
    pools = ["pre1980", "post1980_title"]

    # --- short-circuit: retrofit reverse-edges onto existing CSV ---------
    if args.reverse_edges_only:
        if not OUT_CSV.exists():
            print(f"ERROR: --reverse-edges-only requires existing {OUT_CSV}; aborting.")
            return 1
        print(f"Retrofitting reverse-edge columns into {OUT_CSV.name}...")
        df = pd.read_csv(OUT_CSV, low_memory=False)
        foras_pids, foras_idx = load_foras_index()
        refs = fetch_foras_referenced_works(
            foras_pids,
            cache_path=OUT_DIR / "foras_references_cache.jsonl",
        )
        rev_any, rev_tiab, rev_ft = compute_reverse_edges(refs, foras_idx)
        df["n_cited_by_foras_any"]  = df["openalex_id"].map(rev_any).fillna(0).astype(int)
        df["n_cited_by_foras_tiab"] = df["openalex_id"].map(rev_tiab).fillna(0).astype(int)
        df["n_cited_by_foras_ft"]   = df["openalex_id"].map(rev_ft).fillna(0).astype(int)
        df.to_csv(OUT_CSV, index=False)
        print(f"  Reverse-edges added: any>=1 {(df['n_cited_by_foras_any']>0).sum()}, "
              f"tiab>=1 {(df['n_cited_by_foras_tiab']>0).sum()}, "
              f"ft>=1 {(df['n_cited_by_foras_ft']>0).sum()}")
        print(f"  Bucket counts: LOW={(df['n_cited_by_foras_tiab']==0).sum()} "
              f"MEDIAN={((df['n_cited_by_foras_tiab']>=1)&(df['n_cited_by_foras_tiab']<=2)).sum()} "
              f"HIGH={(df['n_cited_by_foras_tiab']>=3).sum()}")
        print(f"Output: {OUT_CSV}")
        return 0

    print(f"Term count : {len(terms)}")
    print(f"Pools      : {pools}")
    print(f"FORAS xlsx : {SRC_FORAS}")
    print(f"Output dir : {OUT_DIR}")
    print()

    aggregated = {}
    for term in terms:
        for pool in pools:
            term_safe = re.sub(r"[^\w]+", "_", term).strip("_").lower()[:60]
            raw_path = RAW_DIR / f"{term_safe}__{pool}.jsonl"

            cached = raw_path.exists() and raw_path.stat().st_size > 0
            should_fetch = not args.no_fetch and not (args.skip_existing and cached)

            if should_fetch and not args.dry_run:
                try:
                    with open(raw_path, "w", encoding="utf-8") as f:
                        f.truncate(0)
                except (PermissionError, OSError) as e:
                    print(f"  WARN: cannot truncate {raw_path.name} ({e}); append-mode")
                results = query_term_pool(
                    term, pool,
                    limit=args.limit,
                    dry_run=args.dry_run,
                    raw_path=raw_path,
                )
            elif cached:
                print(f"  [{term} | {pool}] cached, reading raw...")
                results = []
                with open(raw_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                results.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            else:
                results = []

            for w in results:
                pid = short_id(w.get("id"))
                if not pid:
                    continue
                if pid not in aggregated:
                    aggregated[pid] = {
                        "openalex_id": pid,
                        "doi":         w.get("doi"),
                        "title":       w.get("title"),
                        "year":        w.get("publication_year"),
                        "abstract":    reconstruct_abstract(w.get("abstract_inverted_index")),
                        "language":    w.get("language"),
                        "type":        w.get("type"),
                        "referenced_works": [
                            short_id(rw)
                            for rw in (w.get("referenced_works") or [])
                            if short_id(rw)
                        ],
                        "pools":           set(),
                        "found_via_terms": set(),
                    }
                aggregated[pid]["pools"].add(pool)
                aggregated[pid]["found_via_terms"].add(term.strip('"'))

    if args.dry_run:
        return 0

    foras_pids, foras_idx = load_foras_index()

    rev_any  = {}
    rev_tiab = {}
    rev_ft   = {}
    if not args.skip_reverse_edges:
        print("\n[Reverse-edges] Fetching FORAS papers' referenced_works from OpenAlex...")
        refs = fetch_foras_referenced_works(
            foras_pids,
            cache_path=OUT_DIR / "foras_references_cache.jsonl",
        )
        rev_any, rev_tiab, rev_ft = compute_reverse_edges(refs, foras_idx)
        print(f"  Computed reverse-edges for {len(rev_any)} unique referenced W-ids "
              f"(of which {sum(1 for v in rev_tiab.values() if v > 0)} have >=1 TIAB-edge).")
    else:
        print("\n[Reverse-edges] SKIPPED (--skip-reverse-edges set).")

    rows = []
    for pid, r in aggregated.items():
        in_foras = pid in foras_pids
        f = foras_idx.get(pid, {})
        edges_to_ft = 0
        edges_to_ab = 0
        for rwp in r["referenced_works"]:
            ref_meta = foras_idx.get(rwp)
            if not ref_meta:
                continue
            if ref_meta.get("label_included_FT") in (1, 1.0, "1"):
                edges_to_ft += 1
            if ref_meta.get("label_included_TIAB") in (1, 1.0, "1"):
                edges_to_ab += 1
        rows.append({
            "openalex_id":      r["openalex_id"],
            "doi":              r["doi"],
            "title":            r["title"],
            "year":             r["year"],
            "language":         r["language"],
            "type":             r["type"],
            "abstract":         r["abstract"],
            "pools":            ";".join(sorted(r["pools"])),
            "found_via_terms":  ";".join(sorted(r["found_via_terms"])),
            "n_terms":          len(r["found_via_terms"]),
            "in_foras":         in_foras,
            "foras_label_FT":   f.get("label_included_FT")   if in_foras else None,
            "foras_label_TIAB": f.get("label_included_TIAB") if in_foras else None,
            "referenced_works": ";".join(r["referenced_works"]),
            "n_referenced":     len(r["referenced_works"]),
            "n_edges_to_foras_FT_included":   edges_to_ft,
            "n_edges_to_foras_TIAB_included": edges_to_ab,
            "n_cited_by_foras_any":           rev_any.get(r["openalex_id"], 0),
            "n_cited_by_foras_tiab":          rev_tiab.get(r["openalex_id"], 0),
            "n_cited_by_foras_ft":            rev_ft.get(r["openalex_id"], 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n=== Summary ===")
    print(f"Unique candidates: {len(df)}")
    print(f"Already in FORAS : {df['in_foras'].sum()}")
    print(f"External         : {(~df['in_foras']).sum()}")
    print()
    if "year" in df.columns and len(df):
        df["era"] = pd.cut(
            df["year"].fillna(-1),
            bins=[-2, 1919, 1944, 1979, 1999, 2025],
            labels=["<1920", "1920-44", "1945-79", "1980-99", "2000+"],
        )
        print("By era:")
        print(df.groupby(["era", "in_foras"]).size().unstack(fill_value=0).to_string())
        print()
    if len(df):
        print("By pool:")
        print(df["pools"].value_counts().to_string())
        print()
        print("By term (top 15):")
        term_counts = {}
        for s in df["found_via_terms"]:
            for t in (s or "").split(";"):
                if t:
                    term_counts[t] = term_counts.get(t, 0) + 1
        for t, n in sorted(term_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"  {t}: {n}")
        print()
        print("Edge-density (candidate -> FORAS, FORWARD):")
        print(f"  Candidates with >=1 edge to FORAS-FT-included   : {(df['n_edges_to_foras_FT_included'] > 0).sum()}")
        print(f"  Candidates with >=1 edge to FORAS-TIAB-included : {(df['n_edges_to_foras_TIAB_included'] > 0).sum()}")
        if "n_cited_by_foras_tiab" in df.columns:
            print()
            print("Reverse-edges (FORAS -> candidate; drives LOW/MED/HIGH buckets):")
            print(f"  Cited by >=1 FORAS-any            : {(df['n_cited_by_foras_any']  > 0).sum()}")
            print(f"  Cited by >=1 FORAS-TIAB-included  : {(df['n_cited_by_foras_tiab'] > 0).sum()}")
            print(f"  Cited by >=1 FORAS-FT-included    : {(df['n_cited_by_foras_ft']   > 0).sum()}")
            print("  Bucket distribution (by n_cited_by_foras_tiab):")
            print(f"    LOW    (=0)   : {(df['n_cited_by_foras_tiab']==0).sum()}")
            print(f"    MEDIAN (1-2)  : {((df['n_cited_by_foras_tiab']>=1)&(df['n_cited_by_foras_tiab']<=2)).sum()}")
            print(f"    HIGH   (>=3)  : {(df['n_cited_by_foras_tiab']>=3).sum()}")
    print()
    print(f"Output: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
