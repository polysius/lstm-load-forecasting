# SPEC — LSTM Time-Series Forecasting System

This file is a copy of the original spec for reference. See the original at:
`SPEC_lstm_forecasting.md`

**Status:** Implementation in progress.
**Resolved decisions (2026-09-09):**
- Branch B — fresh build, no prior code
- Domain: electricity load forecasting (UCI ElectricityLoadDiagrams20112014)
- Resampling: hourly, s=24 (daily), s=168 (weekly)
- Features: target + calendar only
- Compute: CPU locally, Colab/Kaggle for GPU training
