# Results: camelsch_regional_lstm_fast

Test period 01/10/2008 to 30/09/2020 (never used in training). Run `camelsch_regional_lstm_fast_1311_124014`, model_epoch012.

| Metric (median across basins) | Gauged (training basins) | Ungauged (held-out basins) |
|---|---|---|
| Basins | 163 | 41 |
| NSE | 0.64 | 0.50 |
| KGE | 0.68 | 0.52 |
| High-flow volume bias FHV (%) | -22.5 | -21.3 |
| Peak timing error (days) | 0.48 | 0.50 |
| Peak magnitude error, MAPE (%) | 45.5 | 43.1 |
| Annual-maximum ratio sim/obs | 0.59 | 0.62 |
| Annual-maximum timing error (days) | 0.0 | 0.0 |
| 10-yr return level ratio sim/obs (GPD) | 0.67 | 0.67 |
| 10-yr return level: obs and sim 90% CIs overlap | 36% | 32% |

Gap = skill lost when moving to basins the model has never seen (regionalisation gap).
