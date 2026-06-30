# ELAS u4 vs ELAS h3 (heavy) — SQ1 comparison

u4 runs: 40 · h3 runs: 40

## Per-condition means (WSS@95 / Loss / ATD)

| condition | model | n | WSS@95 | Loss | ATD |
|---|---|---|---|---|---|
| baseline | h3 | 10 | 0.8372 | 0.0220 | 296.4 |
| baseline | u4 | 10 | 0.8593 | 0.0211 | 286.5 |
| raw | h3 | 10 | 0.7648 | 0.0305 | 386.3 |
| raw | u4 | 10 | 0.8362 | 0.0318 | 400.1 |
| period | h3 | 10 | 0.7648 | 0.0294 | 375.3 |
| period | u4 | 10 | 0.8359 | 0.0271 | 350.9 |
| full | h3 | 10 | 0.8424 | 0.0218 | 295.8 |
| full | u4 | 10 | 0.8605 | 0.0208 | 285.1 |

## Historical elusive paper time-to-discovery (mean step)

| condition | sentinel | model | mean step | mean % screened | n |
|---|---|---|---|---|---|
| raw | kardiner_1947 | h3 | 5184 | 48.92% | 10 |
| raw | kardiner_1947 | u4 | 5523 | 52.12% | 10 |
| raw | solomon_1988 | h3 | 2704 | 25.52% | 10 |
| raw | solomon_1988 | u4 | 5032 | 47.48% | 10 |
| raw | war_neuroses_tunisian_1944 | h3 | 5183 | 48.91% | 10 |
| raw | war_neuroses_tunisian_1944 | u4 | 5526 | 52.15% | 10 |
| period | kardiner_1947 | h3 | 4449 | 41.99% | 10 |
| period | kardiner_1947 | u4 | 3154 | 29.77% | 10 |
| period | solomon_1988 | h3 | 2649 | 25.00% | 10 |
| period | solomon_1988 | u4 | 3242 | 30.60% | 10 |
| period | war_neuroses_tunisian_1944 | h3 | 4450 | 42.00% | 10 |
| period | war_neuroses_tunisian_1944 | u4 | 3244 | 30.62% | 10 |
| full | kardiner_1947 | h3 | 357 | 3.37% | 10 |
| full | kardiner_1947 | u4 | 338 | 3.19% | 10 |
| full | solomon_1988 | h3 | 255 | 2.40% | 10 |
| full | solomon_1988 | u4 | 245 | 2.31% | 10 |
| full | war_neuroses_tunisian_1944 | h3 | 450 | 4.24% | 10 |
| full | war_neuroses_tunisian_1944 | u4 | 339 | 3.20% | 10 |
