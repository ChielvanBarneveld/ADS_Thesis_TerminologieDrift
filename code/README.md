# `code/`

All experiment code. One subfolder per sub-question.

| Subfolder | SQ  | What                                                        |
| --------- | --- | ----------------------------------------------------------- |
| `SQ1/`       | SQ1 | Sentinel injection into FORAS + ELAS u4 simulations              |
| `SQ1_heavy/` | SQ1 | SQ1 re-run with the ELAS h3 (mxbai) model + u4-vs-h3 comparison  |
| `SQ2/`       | SQ2 | Regex drift sweep over the (pp × nn) intensity grid              |
| `SQ2_heavy/` | SQ2 | SQ2 drift grid re-run with the ELAS h3 (mxbai) model            |

Each subfolder has its own `README.md` documenting inputs, outputs and the order in which to run the scripts.

## Conventions

- Scripts at the top of each `SQx/` folder are runnable directly with `python path/to/script.py`.
- Scripts named `*_cli.py` are CLI wrappers around an importable module of the same stem (e.g. `make_summary.py` + `make_summary_cli.py`).
- All paths are resolved relative to the repo root, so scripts work from any CWD as long as the repo layout is intact.
