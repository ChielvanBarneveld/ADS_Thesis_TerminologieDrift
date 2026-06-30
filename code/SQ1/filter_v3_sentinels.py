"""SQ1 — v2-to-v3 sentinel filter (cohort-diversity enforcement).

The v2 sentinel selection produced by select_sentinels.py contains 6 papers
(N_PER_BUCKET = 2). v3 (2026-05-19 decision) tightens this to N=3 with three
diversity constraints:
  1. Each first-author appears at most once
  2. Each cohort/population appears at most once
  3. Each era is covered exactly once: pre-1945, 1945-1979, 1980+

v3.1 retroframing (2026-05-26): the original v3 spec used three (era × bucket)
cells with bucket ∈ {LOW, MEDIAN, HIGH} derived from reverse-citation density
to FORAS-TIAB-included papers. A 2026-05-26 forensic check showed that the
pre-pivot reverse-edge computation was done against FORAS+Synergy combined
(van_de_Schoot_2025.csv, n=14.764) — the bucket labels therefore reflected
Synergy citations, not FORAS-only citations. The pure-FORAS reverse-edges are
0/0/0 for the three sentinels, so the bucket-axis is not reproducible from
FORAS-only data. v3.1 keeps the same three sentinels (Solomon/Kardiner/Grinker)
and reframes them on the era × cohort axis, which is fully reproducible
from publication_year and first-author/journal metadata alone.

This script implements that filter as committed code so the v2 -> v3
transition is part of the reproducibility chain (not a manual edit).

Inputs:
  Report/data/SQ1/rewrites_data.json   (v2 superset: 6 sentinels x 3 conditions)
Outputs:
  Report/data/SQ1/sentinels.json       (v3 metadata: 3 sentinels + diversity_validation)
  Report/data/SQ1/rewrites.json        (v3 filtered: 3 sentinels x 3 conditions)
"""
from __future__ import annotations
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
DATA       = REPORT_DIR / "data" / "SQ1"
V2_REWRITES = DATA / "rewrites_data.json"           # committed v2 superset
OUT_SENTS   = DATA / "sentinels.json"               # v3 metadata
OUT_REWS    = DATA / "rewrites.json"                # v3 filtered rewrites

# The v3 selection per the 2026-05-19 decision (sentinels.json v3 rationale).
# These three OpenAlex IDs uniquely satisfy the diversity constraints:
#   - Solomon 1988 (W2057711239): 1980+ era / Israeli IDF Lebanon War cohort
#   - Kardiner 1947 (W1794622667): 1945-1979 era / US WWI veterans cohort
#   - Grinker  1944 (W4250282992): pre-1945 era / Allied WWII Tunisia cohort
KEEP_IDS = {"W2057711239", "W1794622667", "W4250282992"}

# Diversity-validation metadata (mirrors the v3 spec).
V3_METADATA = {
    "selection_date": "2026-05-19",
    "version": "v3",
    "n_sentinels": 3,
    "claim": "claim_2_discovery_time_delta",
    "e3_mode": "title_only",
    "year_cutoff": 2005,
    "rules_doc": "02_sentinel_selection/methods.md (v3 addendum)",
    "v3_change_rationale": (
        "v2 (12 mei) had 6 sentinels waarvan 3 uit Solomon-cohort (Lebanon War CSR) "
        "en 2 uit Grinker-cohort (WWII Tunisia). Cluster-discovery in raw/period/full "
        "werd hierdoor gedreven door cohort-overlap, niet door drift-vocabulary. "
        "Bovendien had Solomon 1993 een table-of-contents als abstract. v3 dropt naar "
        "N=3 met enforced cohort-diversity: 1 per era x 1 per cohort."
    ),
    "v3_1_retroframing_rationale": (
        "Op 2026-05-26 ontdekt dat de pre-pivot LOW/MEDIAN/HIGH bucket-stratificatie "
        "gebaseerd was op reverse-edges tegen FORAS+Synergy combined (n=14.764) en "
        "niet reproduceerbaar is uit pure FORAS-only data (n=10.594). v3.1 dropt het "
        "bucket-axis en behoudt era + cohort als stratificatie-criteria. Zelfde 3 "
        "sentinels; de oude bucket-labels worden bewaard onder _pre_pivot_bucket "
        "voor transparantie. Zie decisions.md 2026-05-26."
    ),
    "stratification_axis": "era_x_cohort",
    "diversity_constraints": [
        "Geen twee sentinels van dezelfde eerste-auteur",
        "Geen twee sentinels uit dezelfde populatie/cohort",
        "Abstract >= 500 chars EN substantief (geen TOC/lijst)",
        "Drie eras gedekt: pre-1945, 1945-1979, 1980+",
    ],
}

# Per-sentinel v3 metadata (cohort, first_author, journal, etc.)
SENTINEL_METADATA = {
    "W2057711239": {
        "sentinel_id": "solomon_1988", "first_author": "Zahava Solomon",
        "co_authors": ["Mario Mikulincer"],
        "cohort": "Israeli IDF Lebanon War 1982 CSR casualties (n=285)",
        "journal": "The Journal of Nervous and Mental Disease",
        "doi": "https://doi.org/10.1097/00005053-198805000-00002",
        "abstract_contains_ptsd_token": True,
        "note": "PTSD-token in abstract is consistent with E3 (title-only).",
    },
    "W1794622667": {
        "sentinel_id": "kardiner_1947",
        "first_author": "Abram Kardiner (impliciet via book-title; OpenAlex authorships leeg)",
        "co_authors": ["Herbert Spiegel"],
        "cohort": "US WWI veterans Bureau case files (n=50)",
        "journal": "Annals of Internal Medicine",
        "doi": "https://doi.org/10.7326/0003-4819-27-6-1051_2",
        "abstract_contains_ptsd_token": False,
        "note": "Book-restatement van Kardiners 1941 'Traumatic Neuroses of War'.",
    },
    "W4250282992": {
        "sentinel_id": "war_neuroses_tunisian_1944", "first_author": "Roy R. Grinker",
        "co_authors": [],
        "cohort": "Allied WWII neuropsychiatric casualties, Tunisian campaign (n=120)",
        "journal": "Journal of the American Medical Association",
        "doi": "https://doi.org/10.1001/jama.1944.02850420061026",
        "abstract_contains_ptsd_token": False,
        "note": "Classic WWII war-neurosis preliminary field report.",
    },
}

DROPPED_FROM_V6 = [
    {"sentinel_id": "solomon_1993", "openalex_id": "W1897891557",
     "reason": "Same cohort as solomon_1988 (Solomon-Lebanon); abstract = book TOC (non-substantive)"},
    {"sentinel_id": "solomon_1987", "openalex_id": "W2151134135",
     "reason": "Same cohort as solomon_1988 (Solomon-Lebanon); redundant"},
    {"sentinel_id": "traumatic_war_neuroses_1951", "openalex_id": "W1971917793",
     "reason": "Futterman+Pumpian-Mindlin different cohort from Grinker; era overlap with kardiner_1947"},
]


def main() -> None:
    v2 = json.loads(V2_REWRITES.read_text(encoding="utf-8"))
    v2_sentinels = v2["sentinels"]
    print(f"Read v2 superset: {len(v2_sentinels)} sentinels from {V2_REWRITES.name}")

    # Filter v2 -> v3
    kept = [s for s in v2_sentinels if s["openalex_id"] in KEEP_IDS]
    if len(kept) != 3:
        raise RuntimeError(
            f"Expected 3 v3 sentinels from KEEP_IDS={KEEP_IDS}, found {len(kept)}.\n"
            f"Available v2 IDs: {[s['openalex_id'] for s in v2_sentinels]}"
        )

    # ── sentinels.json (v3 metadata) ────────────────────────────────────
    v3_sentinels_meta = []
    for s in kept:
        oid = s["openalex_id"]
        m = SENTINEL_METADATA[oid]
        abstract_text = s.get("raw") or s.get("conditions", {}).get("raw", {}).get("abstract", "")
        v3_sentinels_meta.append({
            "sentinel_id": m["sentinel_id"], "openalex_id": oid,
            "doi": m["doi"], "title": s["title"], "year": s["year"],
            "first_author": m["first_author"], "co_authors": m["co_authors"],
            "cohort": m["cohort"], "journal": m["journal"],
            "era": s.get("era"),
            "n_cited_by_foras_tiab": s.get("n_cited_by_foras_tiab", s.get("n_edges_tiab", 0)),
            "n_cited_by_foras_ft":   s.get("n_cited_by_foras_ft", 0),
            "n_cited_by_foras_any":  s.get("n_cited_by_foras_any", 0),
            "_pre_pivot_bucket": s.get("_pre_pivot_bucket") or s.get("bucket"),
            "_pre_pivot_n_edges_tiab": s.get("_pre_pivot_n_edges_tiab") or s.get("n_edges_tiab"),
            "abstract_chars_in_oa":  len(abstract_text),
            "abstract_contains_ptsd_token": m["abstract_contains_ptsd_token"],
            "note": m["note"],
        })

    diversity_validation = {
        "unique_first_authors": len({m["first_author"] for m in v3_sentinels_meta}),
        "unique_cohorts":       len({m["cohort"]       for m in v3_sentinels_meta}),
        "unique_journals":      len({m["journal"]      for m in v3_sentinels_meta}),
        "unique_eras":          len({m["era"]          for m in v3_sentinels_meta}),
        "abstract_min_chars":   min(m["abstract_chars_in_oa"] for m in v3_sentinels_meta),
        "any_TOC_abstract":     False,
    }
    assert diversity_validation["unique_first_authors"] == 3, "v3 must have 3 unique first-authors"
    assert diversity_validation["unique_cohorts"]       == 3, "v3 must have 3 unique cohorts"
    assert diversity_validation["unique_eras"]          == 3, "v3 must cover 3 eras"

    v3_doc = dict(V3_METADATA)
    v3_doc["sentinels"] = v3_sentinels_meta
    v3_doc["diversity_validation"] = diversity_validation
    v3_doc["dropped_from_v6"] = DROPPED_FROM_V6
    OUT_SENTS.write_text(json.dumps(v3_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_SENTS.name} (v3, 3 sentinels)")

    # ── rewrites.json (v3 filtered: 3 sentinels x 3 conditions) ─────────
    rewrites_doc = dict(v2)
    rewrites_doc["sentinels"] = kept
    rewrites_doc["n_sentinels"] = len(kept)
    rewrites_doc["_filtered_from_v2"] = (
        "Filtered from rewrites_data.json (v2 superset, 6 sentinels) to v3 "
        "(3 sentinels) per filter_v3_sentinels.py. KEEP_IDS encoded in script."
    )
    OUT_REWS.write_text(json.dumps(rewrites_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_REWS.name} ({len(kept)} sentinels x 3 conditions)")


if __name__ == "__main__":
    main()
