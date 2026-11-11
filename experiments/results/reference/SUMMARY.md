# Benchmark `reference`

Reproduces the original study exactly as its committed CSVs were produced: naive / seasonal-naive / SARIMA / GBM on 5 folds, a separate 3-fold SARIMA run, and the 3-seed LSTM on 3 folds. It exists to prove the refactor is faithful; quote the `aligned` benchmark instead, which puts every model on the same folds.

## Reproduction vs published results

Overall: **PASS** - 18 PASS, 9 KNOWN, 0 FAIL, 0 MISSING

| Reference | Model | Horizon | Published MASE | Reproduced MASE | Rel. diff | Max fold diff | Tolerance | Status |
|---|---|---|---|---|---|---|---|---|
| `baseline_results.csv (published, efed2b8)` | gbm | h=1 | 0.9939 | 0.5892 | 40.72% | 0.9290 | 2% | KNOWN |
| `baseline_results.csv (published, efed2b8)` | gbm | h=6 | 1.1031 | 0.8663 | 21.47% | 0.5123 | 2% | KNOWN |
| `baseline_results.csv (published, efed2b8)` | gbm | h=24 | 1.3606 | 1.2375 | 9.05% | 0.3037 | 2% | KNOWN |
| `baseline_results.csv (published, efed2b8)` | naive | h=1 | 0.4727 | 0.4727 | 0.00% | 0.0000 | 0% | PASS |
| `baseline_results.csv (published, efed2b8)` | naive | h=6 | 0.6442 | 0.6442 | 0.00% | 0.0000 | 0% | PASS |
| `baseline_results.csv (published, efed2b8)` | naive | h=24 | 0.8115 | 0.8115 | 0.00% | 0.0000 | 0% | PASS |
| `baseline_results.csv (published, efed2b8)` | sarima | h=1 | 0.6069 | 0.6965 | 14.77% | 0.2041 | 0% | KNOWN |
| `baseline_results.csv (published, efed2b8)` | sarima | h=6 | 0.7655 | 0.8866 | 15.82% | 0.3665 | 0% | KNOWN |
| `baseline_results.csv (published, efed2b8)` | sarima | h=24 | 0.8926 | 1.1155 | 24.96% | 0.4907 | 0% | KNOWN |
| `baseline_results.csv (published, efed2b8)` | seasonal_naive | h=1 | 0.7410 | 0.7410 | 0.00% | 0.0000 | 0% | PASS |
| `baseline_results.csv (published, efed2b8)` | seasonal_naive | h=6 | 0.7410 | 0.7410 | 0.00% | 0.0000 | 0% | PASS |
| `baseline_results.csv (published, efed2b8)` | seasonal_naive | h=24 | 0.7426 | 0.7426 | 0.00% | 0.0000 | 0% | PASS |
| `baseline_results.csv (first published, 7f12081)` | gbm | h=1 | 0.5892 | 0.5892 | 0.00% | 0.0000 | 2% | PASS |
| `baseline_results.csv (first published, 7f12081)` | gbm | h=6 | 0.8663 | 0.8663 | 0.00% | 0.0000 | 2% | PASS |
| `baseline_results.csv (first published, 7f12081)` | gbm | h=24 | 1.2375 | 1.2375 | 0.00% | 0.0000 | 2% | PASS |
| `baseline_results.csv (first published, 7f12081)` | naive | h=1 | 0.4727 | 0.4727 | 0.00% | 0.0000 | 0% | PASS |
| `baseline_results.csv (first published, 7f12081)` | naive | h=6 | 0.6442 | 0.6442 | 0.00% | 0.0000 | 0% | PASS |
| `baseline_results.csv (first published, 7f12081)` | naive | h=24 | 0.8115 | 0.8115 | 0.00% | 0.0000 | 0% | PASS |
| `baseline_results.csv (first published, 7f12081)` | seasonal_naive | h=1 | 0.7410 | 0.7410 | 0.00% | 0.0000 | 0% | PASS |
| `baseline_results.csv (first published, 7f12081)` | seasonal_naive | h=6 | 0.7410 | 0.7410 | 0.00% | 0.0000 | 0% | PASS |
| `baseline_results.csv (first published, 7f12081)` | seasonal_naive | h=24 | 0.7426 | 0.7426 | 0.00% | 0.0000 | 0% | PASS |
| `sarima_results.csv` | sarima | h=1 | 0.6924 | 0.6924 | 0.00% | 0.0000 | 0% | PASS |
| `sarima_results.csv` | sarima | h=6 | 0.9102 | 0.9102 | 0.00% | 0.0000 | 0% | PASS |
| `sarima_results.csv` | sarima | h=24 | 1.1236 | 1.1236 | 0.00% | 0.0000 | 0% | PASS |
| `lstm_results.csv` | lstm | h=1 | 0.6216 | 1.2037 | 93.64% | 1.6209 | 5% | KNOWN |
| `lstm_results.csv` | lstm | h=6 | 0.7315 | 1.2588 | 72.09% | 1.4933 | 5% | KNOWN |
| `lstm_results.csv` | lstm | h=24 | 0.8549 | 1.3508 | 58.00% | 1.4214 | 5% | KNOWN |

**Known differences** (published numbers the committed code cannot produce; evidence in RESULTS.md):

- `baseline_results.csv (published, efed2b8)` / gbm: efed2b8 replaced the GBM rows without changing gbm.py; the committed code reproduces the first published rows (7f12081) exactly, not these.
- `baseline_results.csv (published, efed2b8)` / sarima: 5-fold rows first appear in efed2b8; neither committed SARIMA version (30-day or 60-day training window) reproduces them.
- `lstm_results.csv` / lstm: seed 42 was committed (f4ec735) while run_lstm.py still used the large config (hidden 128 x 2 layers); seeds 123/456 were added later. Running the original code from history on identical data reproduces neither: the committed CPU config stops after epoch 1 on fold 0 for every seed.

## Run `baseline_results`

| Model | Horizon | MASE | MAE (z) | RMSE (z) | n |
|---|---|---|---|---|---|
| naive | h=1 | **0.4727** ± 0.2422 | 0.1833 | 0.4309 | 5 folds |
| gbm | h=1 | **0.5892** ± 0.4168 | 0.2291 | 0.4086 | 5 folds |
| sarima | h=1 | **0.6965** ± 0.4099 | 0.2677 | 0.5613 | 5 folds |
| seasonal_naive | h=1 | **0.7410** ± 0.4765 | 0.2750 | 0.6219 | 5 folds |
| naive | h=6 | **0.6442** ± 0.3025 | 0.2501 | 0.5465 | 5 folds |
| seasonal_naive | h=6 | **0.7410** ± 0.4764 | 0.2750 | 0.6220 | 5 folds |
| gbm | h=6 | **0.8663** ± 0.6111 | 0.3382 | 0.5355 | 5 folds |
| sarima | h=6 | **0.8866** ± 0.4757 | 0.3392 | 0.6909 | 5 folds |
| seasonal_naive | h=24 | **0.7426** ± 0.4775 | 0.2756 | 0.6229 | 5 folds |
| naive | h=24 | **0.8115** ± 0.4992 | 0.3066 | 0.6124 | 5 folds |
| sarima | h=24 | 1.1155 ± 0.7904 | 0.4130 | 0.8023 | 5 folds |
| gbm | h=24 | 1.2375 ± 0.7852 | 0.4798 | 0.6983 | 5 folds |

## Run `sarima_results`

| Model | Horizon | MASE | MAE (z) | RMSE (z) | n |
|---|---|---|---|---|---|
| sarima | h=1 | **0.6924** ± 0.3063 | 0.2721 | 0.5907 | 3 folds |
| sarima | h=6 | **0.9102** ± 0.4678 | 0.3566 | 0.7347 | 3 folds |
| sarima | h=24 | 1.1236 ± 0.6413 | 0.4200 | 0.8406 | 3 folds |

## Run `lstm_results`

| Model | Horizon | MASE | MAE (z) | RMSE (z) | n |
|---|---|---|---|---|---|
| lstm | h=1 | 1.2037 ± 0.1103 | 0.6140 | 0.7716 | 3 seeds |
| lstm | h=6 | 1.2588 ± 0.0793 | 0.6292 | 0.8067 | 3 seeds |
| lstm | h=24 | 1.3508 ± 0.0881 | 0.6586 | 0.8715 | 3 seeds |

MAE/RMSE are in standardised units (z-scores of the training fold), as in the original study; MASE is scale-free.
