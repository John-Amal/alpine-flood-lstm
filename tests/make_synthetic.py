"""Build a tiny fake dataset in Caravan format, so the whole pipeline can be smoke-tested
without downloading anything. Streamflow comes from a simple snow + bucket model, so the
LSTM has something learnable. Not a substitute for the real data.

    python tests/make_synthetic.py --out data/synthetic --n-basins 15
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def basin(rng, dates, area, elev):
    n = len(dates)
    doy = dates.dayofyear.values
    t = 8 - elev / 300 + 10 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 3, n)
    p = rng.gamma(0.5, 6, n) * (rng.random(n) < 0.45)
    pet = np.clip(0.3 * (t + 5), 0, None)
    snow, store, q = 0.0, 50.0, np.zeros(n)
    for i in range(n):
        if t[i] < 0:
            snow += p[i]; rain = 0
        else:
            melt = min(snow, 3 * t[i]); snow -= melt; rain = p[i] + melt
        store += rain
        store -= min(store, pet[i] * 0.5)
        q[i] = 0.05 * store + 0.4 * max(store - 120, 0) * (0.5 + area / 2000)
        store -= q[i]
    return pd.DataFrame({
        "total_precipitation_sum": p, "temperature_2m_mean": t, "temperature_2m_min": t - 4,
        "temperature_2m_max": t + 4, "potential_evaporation_sum_ERA5_LAND": pet,
        "snow_depth_water_equivalent_mean": np.full(n, snow), "streamflow": q,
    }, index=pd.Index(dates, name="date"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--n-basins", type=int, default=15)
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    dates = pd.date_range("1984-01-01", "2020-12-31", freq="D")
    ts = a.out / "timeseries" / "netcdf" / "camelsch"; ts.mkdir(parents=True, exist_ok=True)
    at = a.out / "attributes" / "camelsch"; at.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(a.n_basins):
        gid = f"camelsch_{2000 + i}"
        area, elev = rng.uniform(20, 1500), rng.uniform(400, 2500)
        df = basin(rng, dates, area, elev)
        xr.Dataset.from_dataframe(df).to_netcdf(ts / f"{gid}.nc")
        rows.append({"gauge_id": gid, "area": area, "ele_mt_sav": elev, "p_mean": df.total_precipitation_sum.mean(),
                     "pet_mean_ERA5_LAND": df.potential_evaporation_sum_ERA5_LAND.mean(),
                     "frac_snow": float((df.temperature_2m_mean < 0).mean()),
                     "gauge_lat": rng.uniform(45.9, 47.7), "gauge_lon": rng.uniform(6.0, 10.4)})
    pd.DataFrame(rows).to_csv(at / "attributes_caravan_camelsch.csv", index=False)
    print(f"Wrote {a.n_basins} synthetic basins to {a.out}")


if __name__ == "__main__":
    main()
