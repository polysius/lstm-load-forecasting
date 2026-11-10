# Benchmark `aligned`

Every model on the same 5 rolling-origin folds (the LSTM over 3 seeds), so the numbers are directly comparable. The GBM appears twice: `gbm_legacy` with the original features and `gbm` with the train/serve calendar skew and lag misalignment fixed. This is the comparison to quote.

## Run `all_models`

| Model | Horizon | MASE | MAE (z) | RMSE (z) | n |
|---|---|---|---|---|---|
| naive | h=1 | **0.4727** ± 0.2422 | 0.1833 | 0.4309 | 5 folds |
| gbm | h=1 | **0.5611** ± 0.3785 | 0.2210 | 0.3929 | 5 folds |
| gbm_legacy | h=1 | **0.5892** ± 0.4168 | 0.2291 | 0.4086 | 5 folds |
| sarima | h=1 | **0.6965** ± 0.4099 | 0.2677 | 0.5613 | 5 folds |
| seasonal_naive | h=1 | **0.7410** ± 0.4765 | 0.2750 | 0.6219 | 5 folds |
| lstm | h=1 | 1.0282 ± 0.0761 | 0.4701 | 0.6304 | 3 seeds |
| naive | h=6 | **0.6442** ± 0.3025 | 0.2501 | 0.5465 | 5 folds |
| gbm | h=6 | **0.7316** ± 0.4660 | 0.2908 | 0.4791 | 5 folds |
| seasonal_naive | h=6 | **0.7410** ± 0.4764 | 0.2750 | 0.6220 | 5 folds |
| gbm_legacy | h=6 | **0.8663** ± 0.6111 | 0.3382 | 0.5355 | 5 folds |
| sarima | h=6 | **0.8866** ± 0.4757 | 0.3392 | 0.6909 | 5 folds |
| lstm | h=6 | 1.0857 ± 0.0766 | 0.4881 | 0.6677 | 3 seeds |
| seasonal_naive | h=24 | **0.7426** ± 0.4775 | 0.2756 | 0.6229 | 5 folds |
| naive | h=24 | **0.8115** ± 0.4992 | 0.3066 | 0.6124 | 5 folds |
| gbm | h=24 | **0.9787** ± 0.5988 | 0.3926 | 0.5937 | 5 folds |
| sarima | h=24 | 1.1155 ± 0.7904 | 0.4130 | 0.8023 | 5 folds |
| lstm | h=24 | 1.1899 ± 0.0845 | 0.5226 | 0.7307 | 3 seeds |
| gbm_legacy | h=24 | 1.2375 ± 0.7852 | 0.4798 | 0.6983 | 5 folds |

MAE/RMSE are in standardised units (z-scores of the training fold), as in the original study; MASE is scale-free.
