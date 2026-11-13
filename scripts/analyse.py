"""Evaluate a trained run: gauged vs ungauged skill, flood peaks, and extreme-value tails.

Reads the NeuralHydrology test results (all basins, test period 2008-10 to 2020-09) and writes
`results/summary.md`, `results/basin_metrics.csv` and four figures.

Three questions:
1. How much skill is lost when moving from gauged basins to basins never seen in training?
2. Are flood peaks right: timing, magnitude, high-flow volume?
3. Does the model get the tail right? A GPD is fitted to the ~3 largest independent events per year,
   separately for observed and simulated flow, and the 10-year return levels are compared,
   with bootstrap confidence intervals.

Usage
-----
    python scripts/analyse.py                      # latest run, last epoch
    python scripts/analyse.py --run-dir runs/<name> --epoch 12
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks
from scipy.stats import genpareto
from neuralhydrology.evaluation.metrics import calculate_metrics

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["NSE", "KGE", "FHV", "Peak-Timing", "Peak-MAPE"]
RETURN_PERIOD = 10          # years
EVENTS_PER_YEAR = 3         # POT: keep the largest ~3 independent events per year
DECLUSTER_DAYS = 7          # minimum separation between independent peaks
N_BOOT = 300


# ---------------------------------------------------------------- loading
def latest_run() -> Path:
    runs = sorted((ROOT / "runs").glob("*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError("No runs found in runs/. Train a model first.")
    return runs[-1]


def load_results(run_dir: Path, epoch: int | None) -> tuple[dict, Path]:
    test_dir = run_dir / "test"
    if epoch is None:
        cands = sorted(test_dir.glob("model_epoch*"))
        if not cands:
            raise FileNotFoundError(f"No test results in {test_dir}. Run: nh-run evaluate --run-dir {run_dir}")
        epoch_dir = cands[-1]
    else:
        epoch_dir = test_dir / f"model_epoch{epoch:03d}"
    with (epoch_dir / "test_results.p").open("rb") as fp:
        return pickle.load(fp), epoch_dir


def series(res: dict, basin: str) -> tuple[pd.Series, pd.Series]:
    ds = res[basin]["1D"]["xr"]
    obs = ds["streamflow_obs"].isel(time_step=-1).to_series()
    sim = ds["streamflow_sim"].isel(time_step=-1).to_series()
    return obs, sim


# ---------------------------------------------------------------- extremes
def return_level(peaks: np.ndarray, u: float, n_years: float, T: float) -> float:
    xi, _, sigma = genpareto.fit(peaks - u, floc=0)
    lam = len(peaks) / n_years
    m = lam * T
    return u + (sigma / xi * (m ** xi - 1) if abs(xi) > 1e-6 else sigma * np.log(m))


def gpd_rl(q: pd.Series, rng: np.random.Generator) -> tuple[float, float, float, int]:
    q = q.dropna()
    n_years = len(q) / 365.25
    idx, _ = find_peaks(q.to_numpy(), distance=DECLUSTER_DAYS)   # independent events, >= 7 days apart
    events = np.sort(q.to_numpy()[idx])[::-1]
    k = int(round(EVENTS_PER_YEAR * n_years))
    if len(events) <= k or k < 10:
        return np.nan, np.nan, np.nan, len(events)
    u = float(events[k])                      # threshold just below the k largest events
    peaks = events[:k]
    rl = return_level(peaks, u, n_years, RETURN_PERIOD)
    boots = []
    for _ in range(N_BOOT):
        s = rng.choice(peaks, size=len(peaks), replace=True)
        try:
            boots.append(return_level(s, u, n_years, RETURN_PERIOD))
        except Exception:
            pass
    lo, hi = np.nanpercentile(boots, [5, 95]) if boots else (np.nan, np.nan)
    return rl, lo, hi, len(peaks)


def annual_max_stats(obs: pd.Series, sim: pd.Series) -> tuple[float, float]:
    """Median ratio sim/obs of annual maxima and median timing error (days), per hydrological year."""
    df = pd.DataFrame({"obs": obs, "sim": sim}).dropna()
    hy = df.index.year + (df.index.month >= 10)
    ratios, lags = [], []
    for _, g in df.groupby(hy):
        if len(g) < 300:
            continue
        t_obs = g["obs"].idxmax()
        win = g.loc[t_obs - pd.Timedelta(days=3): t_obs + pd.Timedelta(days=3), "sim"]
        ratios.append(win.max() / g["obs"].max())
        lags.append(abs((win.idxmax() - t_obs).days))
    return float(np.median(ratios)), float(np.median(lags))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--epoch", type=int)
    args = ap.parse_args()

    run_dir = args.run_dir or latest_run()
    res, epoch_dir = load_results(run_dir, args.epoch)
    cfg = yaml.safe_load((run_dir / "config.yml").read_text())
    train = set((ROOT / "basins" / "train.txt").read_text().split())
    held = set((ROOT / "basins" / "heldout.txt").read_text().split())
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    rows = []
    for b in sorted(res):
        obs, sim = series(res, b)
        m = calculate_metrics(obs.to_xarray(), sim.to_xarray(), metrics=METRICS, resolution="1D")
        am_ratio, am_lag = annual_max_stats(obs, sim)
        rl_o, lo_o, hi_o, n_o = gpd_rl(obs, rng)
        rl_s, lo_s, hi_s, n_s = gpd_rl(sim, rng)
        rows.append({
            "basin": b, "group": "ungauged (held-out)" if b in held else "gauged (training)" if b in train else "other",
            **m, "annmax_ratio": am_ratio, "annmax_lag_days": am_lag,
            "rl10_obs": rl_o, "rl10_obs_lo": lo_o, "rl10_obs_hi": hi_o,
            "rl10_sim": rl_s, "rl10_sim_lo": lo_s, "rl10_sim_hi": hi_s, "n_peaks_obs": n_o,
        })
    df = pd.DataFrame(rows).set_index("basin")
    df["rl10_ratio"] = df["rl10_sim"] / df["rl10_obs"]
    df["rl10_ci_overlap"] = (df["rl10_sim_lo"] <= df["rl10_obs_hi"]) & (df["rl10_obs_lo"] <= df["rl10_sim_hi"])
    df.to_csv(out / "basin_metrics.csv")

    # ---- summary table
    groups = ["gauged (training)", "ungauged (held-out)"]
    def med(g, c): return df.loc[df.group == g, c].median()
    lines = [
        f"# Results: {cfg['experiment_name']}",
        "",
        f"Test period {cfg['test_start_date']} to {cfg['test_end_date']} (never used in training). "
        f"Run `{run_dir.name}`, {epoch_dir.name}.",
        "",
        "| Metric (median across basins) | Gauged (training basins) | Ungauged (held-out basins) |",
        "|---|---|---|",
        f"| Basins | {(df.group == groups[0]).sum()} | {(df.group == groups[1]).sum()} |",
    ]
    fmt = {
        "NSE": ("NSE", "{:.2f}"), "KGE": ("KGE", "{:.2f}"),
        "FHV": ("High-flow volume bias FHV (%)", "{:+.1f}"),
        "Peak-Timing": ("Peak timing error (days)", "{:.2f}"),
        "Peak-MAPE": ("Peak magnitude error, MAPE (%)", "{:.1f}"),
        "annmax_ratio": ("Annual-maximum ratio sim/obs", "{:.2f}"),
        "annmax_lag_days": ("Annual-maximum timing error (days)", "{:.1f}"),
        "rl10_ratio": (f"{RETURN_PERIOD}-yr return level ratio sim/obs (GPD)", "{:.2f}"),
    }
    for col, (label, f) in fmt.items():
        lines.append(f"| {label} | {f.format(med(groups[0], col))} | {f.format(med(groups[1], col))} |")
    cov = [df.loc[df.group == g, "rl10_ci_overlap"].mean() for g in groups]
    lines.append(f"| {RETURN_PERIOD}-yr return level: obs and sim 90% CIs overlap | {cov[0]:.0%} | {cov[1]:.0%} |")
    lines += ["", "Gap = skill lost when moving to basins the model has never seen (regionalisation gap).", ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # ---- figures
    colors = {groups[0]: "#3b6ea8", groups[1]: "#d1793a"}

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for g in groups:
        v = np.sort(df.loc[df.group == g, "NSE"].dropna().clip(lower=-1))
        ax.plot(v, np.linspace(0, 1, len(v)), label=f"{g} (median {np.median(v):.2f})", color=colors[g], lw=2)
    ax.set(xlabel="NSE (clipped at -1)", ylabel="Fraction of basins", xlim=(-1, 1),
           title="Daily streamflow skill, test period")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "fig1_nse_cdf.png", dpi=150); plt.close(fig)

    # map of held-out basins (if coordinates are available)
    try:
        from neuralhydrology.datasetzoo.caravan import load_caravan_attributes
        attrs = load_caravan_attributes(Path(cfg["data_dir"]), basins=list(df.index))
        if {"gauge_lat", "gauge_lon"} <= set(attrs.columns):
            fig, ax = plt.subplots(figsize=(6, 4))
            tr = attrs.loc[df.index[df.group == groups[0]]]
            ho = df[df.group == groups[1]].join(attrs[["gauge_lat", "gauge_lon"]])
            ax.scatter(tr.gauge_lon, tr.gauge_lat, s=8, c="lightgrey", label="training basins")
            sc = ax.scatter(ho.gauge_lon, ho.gauge_lat, c=ho.NSE.clip(0, 1), cmap="viridis", s=45,
                            edgecolor="k", lw=0.4, vmin=0, vmax=1, label="held-out basins")
            fig.colorbar(sc, ax=ax, label="NSE (held-out basins)")
            ax.set(xlabel="Longitude", ylabel="Latitude", title="Skill in basins never seen in training")
            ax.legend(frameon=False, loc="lower left", fontsize=8)
            fig.tight_layout(); fig.savefig(out / "fig2_heldout_map.png", dpi=150); plt.close(fig)
    except Exception as e:  # map is optional
        print(f"Map skipped: {e}")

    # hydrograph: median held-out basin, around its largest observed flood
    ho = df[df.group == groups[1]].dropna(subset=["NSE"])
    b = ho["NSE"].sub(ho["NSE"].median()).abs().idxmin()
    obs, sim = series(res, b)
    t = obs.idxmax()
    w = slice(t - pd.Timedelta(days=45), t + pd.Timedelta(days=45))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(obs.loc[w], color="k", lw=1.5, label="observed")
    ax.plot(sim.loc[w], color=colors[groups[1]], lw=1.5, label="LSTM (basin unseen in training)")
    ax.set(ylabel="Streamflow (mm/day)", title=f"Largest test-period flood, {b} (median held-out basin, "
           f"NSE {ho.loc[b, 'NSE']:.2f})")
    ax.legend(frameon=False); fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(out / "fig3_flood_hydrograph.png", dpi=150); plt.close(fig)

    # return levels obs vs sim, held-out basins
    r = ho.dropna(subset=["rl10_obs", "rl10_sim"])
    if r.empty:
        print("No held-out basin had enough peaks for a GPD fit; fig4 skipped.")
        print(f"\nWrote {out}/summary.md, basin_metrics.csv and figures.")
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.errorbar(r.rl10_obs, r.rl10_sim,
                xerr=[r.rl10_obs - r.rl10_obs_lo, r.rl10_obs_hi - r.rl10_obs],
                yerr=[r.rl10_sim - r.rl10_sim_lo, r.rl10_sim_hi - r.rl10_sim],
                fmt="o", ms=4, color=colors[groups[1]], ecolor="lightgrey", elinewidth=0.8)
    lim = [max(0.1, np.nanmin(r[["rl10_obs", "rl10_sim"]].values) * 0.8),
           np.nanmax(r[["rl10_obs", "rl10_sim"]].values) * 1.2]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Observed (mm/day)", ylabel="Simulated (mm/day)",
           title=f"{RETURN_PERIOD}-yr flood, held-out basins\nGPD on independent peaks, 90% bootstrap CI")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(matplotlib.ticker.ScalarFormatter()); a.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout(); fig.savefig(out / "fig4_return_levels.png", dpi=150); plt.close(fig)
    print(f"\nWrote {out}/summary.md, basin_metrics.csv and figures.")


if __name__ == "__main__":
    main()
