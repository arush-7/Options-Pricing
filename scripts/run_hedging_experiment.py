#!/usr/bin/env python3
"""Delta-hedging experiment: does discrete hedging behave like the theory says?

Three questions, one script:

1. If I sell 20% vol and the market realises 12%, 20% or 28%, what is my PnL?
   (Theory: ``1/2 integral Gamma S^2 (sigma_r^2 - sigma_i^2) dt``.)
2. How much does the rebalancing frequency matter?
3. Where does the PnL actually come from — theta, gamma or the residual?

Run ``python scripts/run_hedging_experiment.py``. Writes:
    results/hedging_summary.csv
    results/figures/hedge_pnl_hist.png
    results/figures/hedge_error_vs_rebalance.png
    results/figures/hedge_attribution.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optlab.hedging.attribution import decompose_pnl
from optlab.hedging.simulate import HedgeResult, simulate_delta_hedge
from optlab.pricing.monte_carlo import simulate_gbm_paths

FIG_DIR = Path("results/figures")
RESULTS_DIR = Path("results")

S0, K, T, R = 100.0, 100.0, 1.0, 0.04
IMPLIED_VOL = 0.20


def vol_mismatch_table(args) -> tuple[pd.DataFrame, dict[float, HedgeResult]]:
    """Sell a fixed 20% vol and sweep what the market actually realises."""
    rows, results = [], {}
    for realised in (0.12, 0.16, 0.20, 0.24, 0.28, 0.32):
        res = simulate_delta_hedge(
            S0, K, T, R, IMPLIED_VOL, realised,
            n_steps=args.rebalances, n_paths=args.paths, seed=args.seed,
        )
        results[realised] = res
        rows.append(
            {
                "realised_vol": realised,
                "implied_vol": IMPLIED_VOL,
                "mean_pnl": res.mean,
                "std_pnl": res.std,
                "ci95_halfwidth": 1.96 * res.stderr,
                "mean_trading_cost": res.mean_cost,
                "p05": res.quantiles()["p05"],
                "p95": res.quantiles()["p95"],
            }
        )
    return pd.DataFrame(rows), results


def rebalance_table(args) -> pd.DataFrame:
    """Same option, same path distribution, different hedging grids."""
    rows = []
    for n_steps in (2, 5, 12, 26, 52, 126, 252):
        res = simulate_delta_hedge(
            S0, K, T, R, IMPLIED_VOL, IMPLIED_VOL,
            n_steps=n_steps, n_paths=args.paths, seed=args.seed,
        )
        rows.append(
            {
                "rebalances": n_steps,
                "days_between": 365.0 / n_steps,
                "mean_pnl": res.mean,
                "std_pnl": res.std,
                "stderr": res.stderr,
            }
        )
    return pd.DataFrame(rows)


def plot_pnl_histograms(results: dict[float, HedgeResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for realised, style in ((0.14, "-"), (0.20, "-"), (0.28, "-")):
        if realised not in results:
            continue
        res = results[realised]
        ax.hist(res.pnl, bins=70, alpha=0.55, density=True, label=f"realised {realised:.0%}")
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("hedged PnL at expiry (per option, spot = 100)")
    ax.set_ylabel("density")
    ax.set_title("Delta-hedged PnL: sold 20% vol, daily rebalancing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rebalance_error(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(table["rebalances"], table["std_pnl"], "o-", label="std of hedge error")
    reference = table["std_pnl"].iloc[0] * np.sqrt(table["rebalances"].iloc[0] / table["rebalances"])
    ax.loglog(table["rebalances"], reference, "k--", linewidth=1, label=r"$\propto 1/\sqrt{n}$")
    ax.set_xlabel("number of rebalances over the year")
    ax.set_ylabel("standard deviation of PnL")
    ax.set_title("Discrete hedging error shrinks with rebalancing frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attribution(attribution, path: Path) -> None:
    summary = attribution.summary().set_index("component")["mean PnL"]
    parts = ["theta", "gamma", "vega", "financing", "(residual)"]
    values = summary.loc[parts].to_numpy(dtype=float)
    colors = ["#c44e52" if v < 0 else "#4c72b0" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(parts)), values, color=colors)
    ax.set_xticks(range(len(parts)))
    ax.set_xticklabels(parts, rotation=15, ha="right")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("mean PnL per option")
    ax.set_title(
        f"Where the hedged PnL comes from (realised 28% vs implied 20%)\n"
        f"sum = {summary.loc['= hedged PnL']:+.3f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", type=int, default=4000, help="simulated paths (more = tighter CI)")
    parser.add_argument("--rebalances", type=int, default=252, help="hedges per year")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mismatch, results = vol_mismatch_table(args)
    rebal = rebalance_table(args)

    print("=== vol sold vs vol realised (daily hedging, 1y ATM call) ===")
    print(mismatch.round(4).to_string(index=False))
    print("\n=== rebalancing frequency (vol sold = vol realised = 20%) ===")
    print(rebal.round(4).to_string(index=False))

    # Attribution: matched vol, and a genuine vol mismatch.
    for label, realised in (("matched (20% vs 20%)", 0.20), ("mismatch (28% vs 20%)", 0.28)):
        paths = simulate_gbm_paths(S0, R, realised, T, args.rebalances, args.paths, seed=args.seed)
        attribution = decompose_pnl(paths, K=K, T=T, r=R, sigma=IMPLIED_VOL)
        print(f"\n=== PnL attribution, {label} ===")
        print(attribution.summary().round(4).to_string(index=False))
        if realised == 0.28:
            plot_attribution(attribution, FIG_DIR / "hedge_attribution.png")

    plot_pnl_histograms(results, FIG_DIR / "hedge_pnl_hist.png")
    plot_rebalance_error(rebal, FIG_DIR / "hedge_error_vs_rebalance.png")

    mismatch.to_csv(RESULTS_DIR / "hedging_summary.csv", index=False)
    print(f"\nwrote {RESULTS_DIR / 'hedging_summary.csv'} and 3 figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
