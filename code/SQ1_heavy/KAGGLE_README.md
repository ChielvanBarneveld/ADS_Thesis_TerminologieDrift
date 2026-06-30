# Running the SQ1 heavy-model on Kaggle (free GPU) + embedding cache

The heavy model (`elas_h3`) embeds the corpus with **mxbai** (a 335M-param
transformer). On CPU that is hours per run; on a Kaggle **T4/P100 GPU** it is
seconds–minutes. Combined with the **embedding cache** (built into
`run_simulations_heavy.py`), each condition is embedded once and reused across all
seeds, so the GPU work is tiny and the rest (the active-learning loop) is plain CPU.

## What goes where

- **GPU (Kaggle):** building the per-condition embedding caches (`emb_cache/*.npy`).
- **CPU (Kaggle, same notebook):** the active-learning loop, reusing the caches.
- **Your laptop:** download the zipped results, drop into `Report/outputs/SQ1_heavy/`,
  then run `make_compare.py` for the u4-vs-h3 table.

## Step 1 — make a Kaggle Dataset with the inputs

Upload these 6 files into one new Kaggle Dataset (e.g. named `sq1-foras-heavy`):

- the 4 condition CSVs from `Report/outputs/SQ1/simulations/datasets/`:
  `foras_xlsx_baseline.csv`, `foras_xlsx_with_sentinels_raw.csv`,
  `foras_xlsx_with_sentinels_period.csv`, `foras_xlsx_with_sentinels_full.csv`
- `run_simulations_heavy.py`
- `make_compare.py`

(Tip: on your laptop, gather them first — see the one-liner at the bottom.)

## Step 2 — new Kaggle Notebook, settings

- **Accelerator: GPU T4 x2** (or P100).
- **Internet: ON** (needed for `pip install` and the one-time mxbai model download).
- Add your `sq1-foras-heavy` dataset to the notebook (right panel → Add Input).

It will mount at `/kaggle/input/sq1-foras-heavy/`.

## Step 3 — paste these cells

```python
# Cell 1 — install ASReview + the heavy (mxbai) extractor
!pip install -q "asreview>=3.0" asreview-dory
```

```python
# Cell 2 — confirm the GPU is visible
import torch; print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

```python
# Cell 3 — auto-find the uploaded files (works no matter how Kaggle nests the zip)
import os, glob
hits = glob.glob("/kaggle/input/**/foras_xlsx_baseline.csv", recursive=True)
assert hits, "CSVs not found under /kaggle/input — did you Add Input the dataset?"
DATA_DIR = os.path.dirname(hits[0])
RUNNER = os.path.join(DATA_DIR, "run_simulations_heavy.py")
os.environ["DATA_DIR"] = DATA_DIR
os.environ["OUT_DIR"]  = "/kaggle/working/sq1_heavy/simulations"
os.environ["CONDS"]    = "baseline,raw,period,full"
os.environ["SEEDS"]    = "2"                   # 4 conditions x 2 seeds = 8 sims
print("DATA_DIR =", DATA_DIR)
```

```python
# Cell 4 — run. The script reads DATA_DIR/OUT_DIR from the env above.
# (exec works because the runner guards __file__ for notebooks.)
exec(open(RUNNER).read())
```

```python
# Cell 5 — (optional) build the u4-vs-h3 table needs the u4 summary too; do this
# step on your laptop instead. Here we just zip the heavy results for download.
import shutil
shutil.make_archive("/kaggle/working/sq1_heavy_results", "zip", "/kaggle/working/sq1_heavy")
print("Download /kaggle/working/sq1_heavy_results.zip from the right panel (Output).")
```

What you'll see in Cell 4: per condition `embedding cache MISS — building ...` (the
GPU step, once), then per seed the `WSS@95 / loss / ATD` line and each historical
elusive paper's discovery step. Re-running is resumable and skips finished sims;
embedding caches are reused (`cache HIT`).

To **only build the caches first** (sanity-check the GPU path), set
`os.environ["EMBED_ONLY"]="1"` in Cell 3 before Cell 4.

## Step 4 — back on your laptop

1. Download `sq1_heavy_results.zip`, unzip into `Report/outputs/SQ1_heavy/`
   (so you get `Report/outputs/SQ1_heavy/simulations/summary.json` + `runs/`).
2. Compare against the u4 baseline:
   ```powershell
   & "$env:USERPROFILE\.venvs\thesis\Scripts\Activate.ps1"
   python "C:\Users\Chiel van Barneveld\OneDrive - Universiteit Utrecht\ADS\Thesis\Report\code\SQ1_heavy\make_compare.py"
   ```
   → writes `Report/outputs/SQ1_heavy/u4_vs_h3_comparison.{md,csv}`.

## Gather the upload files on your laptop (PowerShell)

```powershell
$dst = "$env:USERPROFILE\Desktop\sq1-foras-heavy"
New-Item -ItemType Directory -Force $dst | Out-Null
$base = "C:\Users\Chiel van Barneveld\OneDrive - Universiteit Utrecht\ADS\Thesis\Report"
Copy-Item "$base\outputs\SQ1\simulations\datasets\foras_xlsx_baseline.csv" $dst
Copy-Item "$base\outputs\SQ1\simulations\datasets\foras_xlsx_with_sentinels_raw.csv" $dst
Copy-Item "$base\outputs\SQ1\simulations\datasets\foras_xlsx_with_sentinels_period.csv" $dst
Copy-Item "$base\outputs\SQ1\simulations\datasets\foras_xlsx_with_sentinels_full.csv" $dst
Copy-Item "$base\code\SQ1_heavy\run_simulations_heavy.py" $dst
Copy-Item "$base\code\SQ1_heavy\make_compare.py" $dst
explorer $dst   # then upload this folder's contents as a new Kaggle Dataset
```

## Notes

- The embedding cache (`emb_cache/<condition>.npy`) is what makes 2 seeds cost the
  same GPU work as 1. It is also why a `run_simulations_heavy.py` re-run is fast.
- The AL loop is sklearn (CPU) and identical to the u4 protocol (n_query=1), so
  WSS@95 / Loss / ATD / time-to-discovery stay directly comparable to u4.
- If a Kaggle session times out (9h limit), just re-run Cell 4 — finished sims and
  caches are skipped, so it resumes.
```
