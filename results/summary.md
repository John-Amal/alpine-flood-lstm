# Results: camelsch_regional_lstm_medium

Test period 01/10/2008 to 30/09/2020 (never used in training). Run `camelsch_regional_lstm_medium_1311_125817`, model_epoch025.

| Metric (median across basins) | Gauged (training basins) | Ungauged (held-out basins) |
|---|---|---|
| Basins | 163 | 41 |
| NSE | 0.72 | 0.51 |
| KGE | 0.78 | 0.56 |
| High-flow volume bias FHV (%) | -14.6 | -11.5 |
| Peak timing error (days) | 0.44 | 0.50 |
| Peak magnitude error, MAPE (%) | 38.3 | 42.0 |
| Annual-maximum ratio sim/obs | 0.67 | 0.70 |
| Annual-maximum timing error (days) | 0.0 | 0.0 |
| 10-yr return level ratio sim/obs (GPD) | 0.85 | 0.88 |
| 10-yr return level: obs and sim 90% CIs overlap | 75% | 71% |

Gap = skill lost when moving to basins the model has never seen (regionalisation gap).
