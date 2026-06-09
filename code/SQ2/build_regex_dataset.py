"""Build the SQ2 regex-rewritten FORAS dataset.

For each FORAS paper, produce a parallel `rewritten_*` version where every
PTSD-token in title + abstract is replaced by a SINGLE historical term sampled
from the candidate-pool distribution (one term per paper -> internally
consistent era, not a mix of terms within one abstract).

Reads:
  Report/data/foras/PTSS_Data_Foras_2025-02-05.xlsx
  Report/data/SQ1/candidates_with_terms.csv          (used to derive term frequencies)

Writes:
  Report/data/SQ2/term_distribution.json         (historical term -> weight)
  Report/data/SQ2/foras_regex_rewritten.parquet  (original + rewritten columns)

The output parquet has columns:
    title, abstract, included,
    original_title, original_abstract,
    rewritten_title, rewritten_abstract,
    is_rewriteable        (True iff abstract/title contained >=1 PTSD-token)

The grid simulation script (`run_simulations_grid.py`) then mixes original and
rewritten versions per (pp, nn) cell.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent.parent
XLSX       = REPORT_DIR / "data" / "foras" / "PTSS_Data_Foras_2025-02-05.xlsx"
CAND       = REPORT_DIR / "data" / "SQ1" / "candidates_with_terms.csv"
OUT_DIR    = REPORT_DIR / "data" / "SQ2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIST   = OUT_DIR / "term_distribution.json"
OUT_PARQ   = OUT_DIR / "foras_regex_rewritten.parquet"

# PTSD-token regex (case-insensitive). Covers the main modern variants.
PTSD_TOKEN_RE = re.compile(
    r"\b(?:"
    r"post[\s-]?traumatic\s+stress\s+disorder|"
    r"posttraumatic\s+stress\s+disorder|"
    r"posttraumatic\s+stress|"
    r"post[\s-]?traumatic\s+stress|"
    r"PTSD|"
    r"PTSS"
    r")\b",
    flags=re.IGNORECASE,
)


def load_term_distribution() -> dict[str, float]:
    """Derive historical-term weights from the candidate-pool.

    candidates_with_terms.csv has a row per (paper, term) hit; each unique paper
    can be matched by multiple terms. We use raw frequency over the pool to
    proxy "how common is this term in PTSD-history literature".
    """
    df = pd.read_csv(CAND, low_memory=False)
    # Per-term provenance lives in `found_via_terms` (semicolon-separated).
    candidates_col = "found_via_terms"
    if candidates_col not in df.columns:
        raise RuntimeError(
            f"Expected column '{candidates_col}' in {CAND}; available: {list(df.columns)}"
        )
    counts: Counter[str] = Counter()
    for v in df[candidates_col].dropna():
        for term in str(v).split(";"):
            term = term.strip()
            if term:
                counts[term] += 1
    total = sum(counts.values())
    dist = {t: c / total for t, c in counts.most_common()}
    OUT_DIST.write_text(json.dumps(dist, indent=2))
    print(f"wrote {OUT_DIST.name}: {len(dist)} terms, total hits={total}")
    print("  top 10:", list(dist.items())[:10])
    return dist


def draw_term(dist: dict[str, float], rng: np.random.Generator) -> str:
    """Draw ONE historical term from the candidate-pool distribution."""
    terms = list(dist.keys())
    weights = np.array(list(dist.values()), dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(terms, p=weights))


def rewrite_text(text: str, term: str) -> tuple[str, int]:
    """Replace every PTSD-token in `text` with the SAME `term`.

    One term per paper (see main): all occurrences in a paper's title and
    abstract are replaced by a single sampled historical term, so each rewritten
    paper is internally consistent in era/terminology (not a mix of e.g.
    "shell shock" and "battle fatigue" within one abstract).
    Returns (rewritten_text, n_replaced). Empty/NaN input returns ("", 0).
    """
    if not isinstance(text, str) or not text:
        return "", 0
    n_replaced = len(PTSD_TOKEN_RE.findall(text))
    rewritten = PTSD_TOKEN_RE.sub(lambda _m: term, text)
    return rewritten, n_replaced


def main() -> None:
    print(f"reading {XLSX.name}...")
    df = pd.read_excel(XLSX)
    df["included"] = df["label_included_FT"].fillna(0).astype(int)
    df = df.dropna(subset=["title"]).copy()
    df["abstract"] = df["abstract"].fillna("")
    print(f"loaded {len(df)} records, positives (FT)={df['included'].sum()}")

    dist = load_term_distribution()

    rng = np.random.default_rng(42)  # deterministic rewrite-seed
    rewritten_titles: list[str] = []
    rewritten_abstracts: list[str] = []
    rewriteable: list[bool] = []
    for _, row in df.iterrows():
        # ONE term per paper: title + abstract share the same historical term,
        # so a rewritten paper is consistent in era (not a mix of terms).
        term = draw_term(dist, rng)
        t_new, n_t = rewrite_text(row["title"], term)
        a_new, n_a = rewrite_text(row["abstract"], term)
        rewritten_titles.append(t_new)
        rewritten_abstracts.append(a_new)
        rewriteable.append((n_t + n_a) > 0)

    out = pd.DataFrame({
        "original_title":     df["title"].values,
        "original_abstract":  df["abstract"].values,
        "rewritten_title":    rewritten_titles,
        "rewritten_abstract": rewritten_abstracts,
        "included":           df["included"].values,
        "is_rewriteable":     rewriteable,
    })
    # title/abstract columns selectable later — start with originals
    out["title"]    = out["original_title"]
    out["abstract"] = out["original_abstract"]

    out.to_parquet(OUT_PARQ, index=False)
    n_rew = int(out["is_rewriteable"].sum())
    print(f"wrote {OUT_PARQ.name}: {len(out)} rows, rewriteable={n_rew} ({100*n_rew/len(out):.1f}%)")
    pos_rew = int(((out["included"] == 1) & out["is_rewriteable"]).sum())
    neg_rew = int(((out["included"] == 0) & out["is_rewriteable"]).sum())
    print(f"   positives rewriteable: {pos_rew}/{int((out['included']==1).sum())}")
    print(f"   negatives rewriteable: {neg_rew}/{int((out['included']==0).sum())}")


if __name__ == "__main__":
    main()
