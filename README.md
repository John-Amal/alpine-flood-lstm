# alpine-flood-lstm

**Regional LSTM flood modelling for Swiss catchments: skill in ungauged basins, flood peaks and tail behaviour.**

One LSTM is trained on many Swiss catchments at once (CAMELS-CH in Caravan format, ERA5-Land forcing),
then tested on a 12-year period it never saw, in two settings:

- **Gauged:** catchments used in training, new time period.
- **Ungauged:** 20% of catchments held out entirely, standing in for basins with no gauge.

The gap between the two is the regionalisation gap: how much skill is lost when the model is transferred to
a catchment it has never seen.

Most large-sample studies stop at NSE/KGE. This project also asks whether the model gets **floods** right:

1. **Peak timing and magnitude:** peak-timing error, peak MAPE, high-flow volume bias (FHV), and annual-maximum ratio and timing.
2. **Tail behaviour:** a Generalised Pareto distribution is fitted to the ~3 largest independent events per year,
   separately for observed and simulated flow. The 10-year return levels are compared with 90% bootstrap
   confidence intervals. A model can score a high NSE and still underestimate the floods that matter.

## Results

> Fill in from `results/summary.md` after running.

| Metric (median across basins) | Gauged | Ungauged (held-out) |
|---|---|---|
| NSE | | |
| KGE | | |
| Peak timing error (days) | | |
| Peak magnitude error, MAPE (%) | | |
| 10-yr return level ratio sim/obs | | |

| | |
|---|---|
| ![](results/fig1_nse_cdf.png) | ![](results/fig2_heldout_map.png) |
| ![](results/fig3_flood_hydrograph.png) | ![](results/fig4_return_levels.png) |

**Key findings:** _(2-3 sentences: size of the regionalisation gap, whether peaks are biased low, where the tail estimates disagree)_

## Design choices

| Choice | Why |
|---|---|
| Split by whole basins, plus a separate time period | Tests transfer in space and time together, which is the operational situation for a new site. A random time split would leak information. |
| One regional model, static catchment attributes as inputs | Lets the model learn how catchment properties (area, elevation, slope, snow fraction and others) shape the response, which is what makes transfer possible. |
| NSE loss, normalised per basin | Stops large, high-flow basins dominating training. |
| Validation on training basins only | Held-out basins stay completely unseen until the final test. |
| GPD on independent peaks (≥ 7 days apart), fixed event rate | Standard peaks-over-threshold approach. A fixed rate keeps the threshold comparable between observed and simulated series. |
| Fixed random seed, generated configs | The split and the variable selection can be reproduced. Variable names are matched to the data version, since Caravan renamed some PET fields in v1.5. |

## Run it

About 45-90 minutes in total on a laptop CPU with the `fast` config, most of it unattended training.

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Data: CAMELS-CH, Caravan extension (~550 MB) from Zenodo, record 15025258
mkdir -p data && cd data
curl -L -o Caravan_extension_CH.zip "https://zenodo.org/records/15025258/files/Caravan_extension_CH.zip?download=1"
unzip -q Caravan_extension_CH.zip && cd ..

# 3. Everything: split, train, evaluate, analyse
./run_all.sh fast cpu          # or: ./run_all.sh fast mps (Apple GPU), ./run_all.sh full cuda:0
```

Outputs go to `results/`: `summary.md`, `basin_metrics.csv` (every metric per basin) and four figures.

| Config | Training years | Model | Epochs | Approx. time |
|---|---|---|---|---|
| `fast` | 1995-2005 | LSTM, 64 hidden units, 270-day input sequence | 12 × 400 updates | ~20-40 min on CPU |
| `full` | 1985-2005 | LSTM, 256 hidden units, 365-day input sequence | 30, full epochs | GPU recommended |

Smoke test without the real data (fake catchments from a simple snow and bucket model):

```bash
python tests/make_synthetic.py --out data/synthetic
DATA=data/synthetic ./run_all.sh fast cpu
```

## Limitations

- Daily resolution only. Small, fast Alpine catchments can peak within hours, so daily data smooths peak magnitude.
- ERA5-Land forcing (~9 km) under-resolves convective precipitation in complex terrain. Gridded station data (e.g. MeteoSwiss) should do better.
- A 12-year test period limits tail estimates to roughly the 10-year level. Longer return periods would be extrapolation.
- Some CAMELS-CH catchments are regulated (hydropower, lakes). They are kept in, which may lower skill for those basins.

## Next steps

- **Forecast mode:** drive the trained model with weather forecasts (ECMWF IFS, AI models such as AIFS or GraphCast)
  instead of reanalysis, and measure skill against lead time. Caravan MultiMet provides IFS HRES and GraphCast forecasts in Caravan format.
- **Ensembles and uncertainty:** run ensemble forecasts through the model and check whether the predictive spread is calibrated for flood peaks.
- **Counterfactual storylines:** force the model with the same extreme event in pre-industrial, present and warmer climates
  (km-scale storyline simulations) and test how far an LSTM trained on data can be trusted outside the conditions it was trained on.
- **Sub-daily and fine-tuning:** hourly targets for small catchments, and fine-tuning the regional model on short local records.

## References

- Kratzert et al. (2023). Caravan: A global community dataset for large-sample hydrology. *Scientific Data* 10, 61.
- Höge et al. (2023). CAMELS-CH: hydro-meteorological time series and landscape attributes for 331 catchments in hydrologic Switzerland. *ESSD* 15, 5755-5784.
- Kratzert et al. (2019). Toward improved predictions in ungauged basins: Exploiting the power of machine learning. *Water Resources Research* 55.
- Kratzert et al. (2022). NeuralHydrology: A Python library for deep learning research in hydrology. *JOSS* 7(71), 4050.
- Shalev & Kratzert (2024). Caravan MultiMet: Extending Caravan with multiple weather nowcasts and forecasts. arXiv:2411.09459.
