"""Prepare basin splits and NeuralHydrology configs for the CAMELS-CH (Caravan format) experiment.

What it does
------------
1. Finds the Caravan-format root (folder holding `attributes/` and `timeseries/`).
2. Keeps basins with enough streamflow data in both the training and test periods.
3. Splits basins at random into a training set and a held-out set. Held-out basins
   are never seen in training, so they stand in for ungauged catchments.
4. Picks the forcing and static-attribute names that exist in your copy of the data
   (Caravan renamed some PET variables in v1.5, so names are matched, not hard-coded).
5. Writes `configs/fast.yml` (quick check), `configs/medium.yml` (laptop GPU, ~1 h) and `configs/full.yml`.

Usage
-----
    python scripts/prepare.py --data-dir data/Caravan_extension_CH --device cpu
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Periods (hydrological years, Oct-Sep). Test = 12 years, never used in training.
PERIODS = {
    "full": {"train": ("01/10/1985", "30/09/2005"), "validation": ("01/10/2005", "30/09/2008")},
    "fast": {"train": ("01/10/1995", "30/09/2005"), "validation": ("01/10/2005", "30/09/2008")},
}
TEST = ("01/10/2008", "30/09/2020")

DYNAMIC_CANDIDATES = [
    "total_precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "potential_evaporation_sum",
    "surface_net_solar_radiation_mean",
    "snow_depth_water_equivalent_mean",
    "surface_pressure_mean",
]
STATIC_CANDIDATES = [
    # climate indices (Caravan)
    "p_mean", "pet_mean", "aridity", "frac_snow", "seasonality",
    "high_prec_freq", "high_prec_dur", "low_prec_freq",
    # physiography (HydroATLAS)
    "area", "ele_mt_sav", "slp_dg_sav", "for_pc_sse", "cly_pc_sav", "snd_pc_sav",
    "gla_pc_sse", "kar_pc_sse", "lka_pc_sse",
]
# When a name exists in several versions, prefer these suffixes, in order.
PREFERRED_SUFFIXES = ["", "_FAO_PENMAN_MONTEITH", "_ERA5_LAND"]


def find_caravan_root(path: Path) -> Path:
    for cand in [path, *path.rglob("*")]:
        if cand.is_dir() and (cand / "attributes").is_dir() and (cand / "timeseries").is_dir():
            return cand
    raise FileNotFoundError(f"No folder with attributes/ and timeseries/ found under {path}")


def pick(candidates: list[str], available: list[str]) -> list[str]:
    chosen = []
    for name in candidates:
        for suffix in PREFERRED_SUFFIXES:
            if name + suffix in available:
                chosen.append(name + suffix)
                break
    return chosen


def to_ts(d: str) -> pd.Timestamp:
    return pd.to_datetime(d, format="%d/%m/%Y")


def coverage(series: pd.Series, start: str, end: str) -> float:
    s = series.loc[to_ts(start):to_ts(end)]
    return float(s.notna().mean()) if len(s) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--subdataset", default="camelsch")
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = find_caravan_root(args.data_dir.resolve())
    ts_dir = root / "timeseries" / "netcdf" / args.subdataset
    files = sorted(ts_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No netCDF files in {ts_dir}")
    print(f"Caravan root: {root}\nFound {len(files)} basins in {ts_dir}")

    with xr.open_dataset(files[0]) as ds:
        dyn_available = list(ds.data_vars)
    attr_cols: list[str] = []
    for csv in (root / "attributes" / args.subdataset).glob("*.csv"):
        attr_cols += [c for c in pd.read_csv(csv, nrows=1).columns if c != "gauge_id"]

    dynamic = pick(DYNAMIC_CANDIDATES, dyn_available)
    static = pick(STATIC_CANDIDATES, attr_cols)
    if "streamflow" not in dyn_available or "total_precipitation_sum" not in dynamic:
        raise ValueError("Expected 'streamflow' and 'total_precipitation_sum' in the timeseries files.")
    print(f"Dynamic inputs ({len(dynamic)}): {dynamic}\nStatic attributes ({len(static)}): {static}")

    keep, dropped = [], []
    tr0, tr1 = PERIODS["full"]["train"]
    for f in files:
        with xr.open_dataset(f) as ds:
            q = ds["streamflow"].to_series()
        ok = coverage(q, tr0, tr1) >= args.min_coverage and coverage(q, *TEST) >= args.min_coverage
        (keep if ok else dropped).append(f.stem)
    print(f"Basins with >= {args.min_coverage:.0%} streamflow in train and test periods: {len(keep)} "
          f"(dropped {len(dropped)})")

    rng = random.Random(args.seed)
    shuffled = keep[:]
    rng.shuffle(shuffled)
    n_held = max(1, round(len(shuffled) * args.heldout_frac))
    heldout, train = sorted(shuffled[:n_held]), sorted(shuffled[n_held:])

    bdir = ROOT / "basins"
    bdir.mkdir(exist_ok=True)
    (bdir / "train.txt").write_text("\n".join(train) + "\n")
    (bdir / "heldout.txt").write_text("\n".join(heldout) + "\n")
    (bdir / "all.txt").write_text("\n".join(sorted(train + heldout)) + "\n")
    print(f"Split: {len(train)} training basins, {len(heldout)} held-out (ungauged) basins")

    common = {
        "dataset": "caravan",
        "data_dir": str(root),
        "train_basin_file": str(bdir / "train.txt"),
        # validation during training uses training basins only, so held-out basins stay unseen
        "validation_basin_file": str(bdir / "train.txt"),
        # test on ALL basins; analyse.py separates gauged (train) from ungauged (held-out)
        "test_basin_file": str(bdir / "all.txt"),
        "test_start_date": TEST[0],
        "test_end_date": TEST[1],
        "dynamic_inputs": dynamic,
        "static_attributes": static,
        "target_variables": ["streamflow"],
        "clip_targets_to_zero": ["streamflow"],
        "model": "cudalstm",
        "head": "regression",
        "output_activation": "linear",
        "initial_forget_bias": 3,
        "output_dropout": 0.4,
        "loss": "NSE",
        "optimizer": "Adam",
        "predict_last_n": 1,
        "metrics": ["NSE", "KGE", "FHV", "Peak-Timing", "Peak-MAPE"],
        "validate_every": 5,
        "validate_n_random_basins": 20,
        "device": args.device,
        "num_workers": 0,
        "log_interval": 5,
        "log_tensorboard": False,
        "save_weights_every": 1,
        "seed": args.seed,
        "run_dir": str(ROOT / "runs"),
    }
    variants = {
        "fast": {"hidden_size": 64, "seq_length": 270, "epochs": 12, "batch_size": 256,
                 "max_updates_per_epoch": 400, "learning_rate": {0: 1e-3, 8: 5e-4}},
        "medium": {"hidden_size": 128, "seq_length": 365, "epochs": 25, "batch_size": 256,
                   "max_updates_per_epoch": 1500, "learning_rate": {0: 1e-3, 15: 5e-4, 22: 2e-4}},
        "full": {"hidden_size": 256, "seq_length": 365, "epochs": 30, "batch_size": 256,
                 "learning_rate": {0: 1e-3, 20: 5e-4, 25: 1e-4}},
    }
    (ROOT / "configs").mkdir(exist_ok=True)
    for name, extra in variants.items():
        cfg = {"experiment_name": f"camelsch_regional_lstm_{name}", **common, **extra}
        per = PERIODS.get(name, PERIODS["full"])
        cfg["train_start_date"], cfg["train_end_date"] = per["train"]
        cfg["validation_start_date"], cfg["validation_end_date"] = per["validation"]
        out = ROOT / "configs" / f"{name}.yml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
