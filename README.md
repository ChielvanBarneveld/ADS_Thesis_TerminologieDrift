# ADS Thesis — Terminology drift in PTSD literature

Master's thesis Applied Data Science, Utrecht University. Reproducibility bundle
for the experiments described in `THESIS.md` / `THESIS_full.pdf` (kept separately
in the author's working folder, not in this repo). Part of the FORAS project.

## Research question

> *Can a standard active-learning screening pipeline (ELAS u4) recover PTSD
> literature written under historical terminology (shell shock, soldier's heart,
> war neurosis, battle fatigue, combat stress reaction), and how does its
> performance scale with the degree of terminological drift?*

This is a **simulation / sensitivity study** with two complementary sub-questions
on the standard ELAS u4 pipeline (SVM classifier · Balanced balancer · TF-IDF
feature extractor · Max querier, read from `asreview` via
`get_ai_config("elas_u4")`).

| SQ  | Question                                                                                   | Code            |
| --- | ------------------------------------------------------------------------------------------ | --------------- |
| SQ1 | **Recovery.** Inject 3 historical elusive papers in FORAS in 3 rewrite conditions (raw / period / full); 10 seeds × 4 conditions = 40 simulations. Metrics: time-to-discovery, WSS@95, normalized loss. | `code/SQ1/`      |
| SQ2 | **Degradation.** Replace PTSD terminology in FORAS abstracts with historical variants over a grid of drift prevalence (% positives × % negatives drifted, each ∈ {0,5,10,20,50,100}, 5 trials/cell). Two heatmaps: WSS@95 and normalized loss. | `code/SQ2/`      |

### Heavy-model validation (ELAS h3)

To answer a supervisor question — *if the light pipeline discovers drifted papers
late, does a stronger feature representation recover them in time?* — SQ1 and SQ2
are **re-run with the heavy ELAS h3 model** (mxbai transformer embeddings instead
of TF-IDF; everything else identical). These runs need a GPU and were executed on
Kaggle.

| Folder              | What                                                            |
| ------------------- | -------------------------------------------------------------- |
| `code/SQ1_heavy/`   | h3 re-run of SQ1 + `make_compare.py` (u4-vs-h3 table)         |
| `code/SQ2_heavy/`   | h3 re-run of the SQ2 drift grid                               |

> **Note:** the Graph-Neural-Network direction (former SQ3) was dropped in the
> 30 May 2026 scope pivot; this repo is a pure simulation study (SQ1 + SQ2).

## Reproducibility contract

> Cloning this repo and running the scripts under `code/` regenerates every
> **light** output under `outputs/` exactly as referenced in the thesis. The
> **heavy** outputs need a GPU and are therefore committed as evidence.

- **Code** — committed.
- **Input data** — committed (FORAS xlsx, OpenAlex candidate-pool snapshot,
  hand-curated sentinels + rewrites, term-frequency JSON).
- **Light outputs** (`outputs/SQ1/`, `outputs/SQ2/`) — *not* committed; kept as
  `.gitkeep` placeholders and regenerated on CPU by re-running the relevant
  `code/SQx/` script.
- **Heavy outputs** (`outputs/SQ1_heavy/`, `outputs/SQ2_heavy/`) — **committed**
  (summaries, per-run results, figures, `u4_vs_h3_comparison`). They are produced
  on a GPU (see `code/SQ{1,2}_heavy/KAGGLE_README.md`) and cannot be regenerated
  on a fresh CPU clone. Only the large GPU embedding caches (`emb_cache/`, ~260 MB)
  are gitignored — they are rebuilt on a GPU and are not needed downstream.
- **One regenerable data file** — `data/SQ2/foras_regex_rewritten.parquet`.
  Skipped because `code/SQ2/build_regex_dataset.py` deterministically rebuilds it
  in ~30 s from the committed FORAS xlsx + term distribution.

## How to run

```bash
git clone https://github.com/ChielvanBarneveld/ADS_Thesis_TerminologieDrift.git
cd ADS_Thesis_TerminologieDrift

# 1. Python env
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt

# 2. SQ1 — sentinel injection (40 ELAS u4 simulations, CPU)
python code/SQ1/build_csvs_from_xlsx.py
python code/SQ1/run_simulations_cli.py
python code/SQ1/make_summary_cli.py

# 3. SQ2 — regex drift sweep (CPU)
python code/SQ2/build_regex_dataset.py
python code/SQ2/run_simulations_grid.py
python code/SQ2/make_heatmaps.py

# 4. Heavy validation (ELAS h3, GPU — see KAGGLE_README in each heavy folder)
#    After downloading the Kaggle results into outputs/SQ{1,2}_heavy/:
python code/SQ1_heavy/make_compare.py     # writes outputs/SQ1_heavy/u4_vs_h3_comparison.{csv,md}
```

Each `code/SQx/` folder has its own `README.md` documenting inputs, outputs and
the order of scripts.

## Folder layout

```
.
├── code/
│   ├── SQ1/         # sentinel injection (ELAS u4)
│   ├── SQ1_heavy/   # SQ1 re-run with ELAS h3 + u4-vs-h3 comparison
│   ├── SQ2/         # regex drift sweep (ELAS u4)
│   └── SQ2_heavy/   # SQ2 drift grid with ELAS h3
├── data/
│   ├── foras/       # FORAS-update xlsx (raw input)
│   ├── SQ1/         # candidate pool + sentinels + rewrites
│   └── SQ2/         # term distribution (parquet regen-able)
└── outputs/
    ├── SQ1/         # gitignored — regenerated on CPU by code/SQ1
    ├── SQ1_heavy/   # committed — GPU results (emb_cache gitignored)
    ├── SQ2/         # gitignored — regenerated on CPU by code/SQ2
    └── SQ2_heavy/   # committed — GPU results (emb_cache gitignored)
```

## Citing the data

The FORAS-update corpus is from van de Schoot et al. (2025). The OpenAlex
candidate-pool snapshot was queried on 28 April 2026 using 29 historical PTSD
terms (English, German, French); see `data/SQ1/README.md` for the exact query list.

## Author

Chiel van Barneveld — `chiel.vbarneveld@gmail.com`
First supervisor: dr. Rens van de Schoot · Second supervisor: Timo van der Kuil ·
Second examiner: dr. A. (Robert) Bagheri.
