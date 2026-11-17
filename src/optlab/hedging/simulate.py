"""Discrete delta-hedging: what actually happens between the textbooks.

The set-up
----------
You sell/buy one option and hedge it with ``delta`` shares, rebalancing on a
fixed grid of ``n_steps`` dates. Everything that is not the underlying accrues
interest at ``r``, and each trade costs ``transaction_cost_bps`` of its notional.

The key experiment is the mismatch between two volatilities:

* ``sigma_pricing`` — the vol baked into the option price you traded at
  (i.e. the implied vol);
* ``sigma_realized`` — the vol the underlying actually realised in the simulation.

In continuous time, with continuous rebalancing, this hedge is perfect and your
PnL is exactly the gamma rent::

    PnL = 1/2 * integral Gamma_t S_t^2 (sigma_realized^2 - sigma_pricing^2) dt

Three things break that in practice, and the simulation shows all three:

1. you can only rebalance at discrete times, so the PnL is *random* even when
   the two vols agree — the variance of the hedge error grows like ``1/n_steps``;
2. if realised vol exceeds the vol you sold, you systematically make money
   (long gamma), and lose it when you sold too rich;
3. transaction costs are a one-way drag proportional to the number of
   rebalances and the option's gamma.

Accounting
----------
Positions are: +1 option, ``q`` shares, and a cash account. At inception
``q = -delta`` and ``cash = -P - q*S``, so the initial portfolio value is zero
and the terminal value *is* the PnL. Everything (trade financing, cost
accounting) flows through the cash account, which is why the mean PnL is a clean
test of the whole engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from optlab.pricing.black_scholes import bs_greeks, bs_price, _validate_kind
from optlab.pricing.monte_carlo import simulate_gbm_paths


@dataclass
class HedgeResult:
    """Distribution of terminal PnL from a delta-hedging experiment."""

    pnl: np.ndarray
    n_paths: int
    n_steps: int
    sigma_pricing: float
    sigma_realized: float
    transaction_cost_bps: float
    rebalance_cost: np.ndarray = field(default_factory=lambda: np.array([]))

    # -- summary statistics -------------------------------------------------
    @property
    def mean(self) -> float:
        return float(np.mean(self.pnl))

    @property
    def std(self) -> float:
        return float(np.std(self.pnl, ddof=1)) if self.pnl.size > 1 else 0.0

    @property
    def stderr(self) -> float:
        n = max(self.pnl.size, 1)
        return self.std / np.sqrt(n)

    @property
    def mean_cost(self) -> float:
        return float(np.mean(self.rebalance_cost)) if self.rebalance_cost.size else 0.0

    def quantiles(self, probs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[str, float]:
        return {f"p{int(p * 100):02d}": float(np.quantile(self.pnl, p)) for p in probs}

    def summary(self) -> str:
        q = self.quantiles()
        lines = [
            f"delta-hedge PnL  ({self.n_steps} rebalances, {self.n_paths:,} paths, "
            f"{self.transaction_cost_bps:.1f} bps costs)",
            f"  vol sold {self.sigma_pricing:.2%}  vs realised {self.sigma_realized:.2%}",
            f"  mean {self.mean:+.4f}  (std err {self.stderr:.4f})",
            f"  std  {self.std:.4f}   mean round-trip cost {self.mean_cost:.4f}",
            f"  5%..95%: {q['p05']:+.3f} .. {q['p95']:+.3f}",
        ]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """One row per path — useful for histograms and for tests."""
        return pd.DataFrame({"pnl": self.pnl, "cost": self.rebalance_cost})


def simulate_delta_hedge(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_pricing: float,
    sigma_realized: float | None = None,
    n_steps: int = 252,
    n_paths: int = 2_000,
    q: float = 0.0,
    kind: str = "call",
    transaction_cost_bps: float = 0.0,
    seed: int | None = None,
) -> HedgeResult:
    """Simulate delta-hedging one short option position to expiry.

    The position is long one option and short ``delta`` shares (the usual way a
    dealer runs a book), rebalanced at every one of ``n_steps`` interior dates and
    unwound at expiry.

    Parameters
    ----------
    sigma_pricing : float
        Vol used to price the option and to compute the hedge ratio.
    sigma_realized : float, optional
        Vol used to simulate the underlying. Defaults to ``sigma_pricing``.
    n_steps : int
        Number of hedging dates. 252 = daily for a one-year option.
    transaction_cost_bps : float
        Cost per unit of traded notional, in basis points (1 bp = 1e-4).

    Returns
    -------
    HedgeResult

    Notes
    -----
    PnL is measured on the hedged portfolio, whose initial value is zero. With
    ``sigma_realized == sigma_pricing`` and no costs the mean should be
    statistically indistinguishable from zero — the test suite asserts exactly
    that, which is the cheapest way to catch a sign error in the hedging logic.
    """
    kind = _validate_kind(kind)
    if sigma_realized is None:
        sigma_realized = sigma_pricing
    if T <= 0:
        raise ValueError("T must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    dt = T / n_steps
    tc = transaction_cost_bps * 1e-4
    paths = simulate_gbm_paths(S0, r, sigma_realized, T, n_steps, n_paths, q, seed=seed)

    premium = float(bs_price(S0, K, T, r, sigma_pricing, q, kind))
    delta0 = float(bs_greeks(S0, K, T, r, sigma_pricing, q, kind).delta)

    shares = np.full(n_paths, -delta0)          # short the hedge
    cash = np.full(n_paths, -premium + delta0 * S0)  # buy option, sell shares
    inception_cost = np.full(n_paths, tc * abs(delta0) * S0)
    cash -= inception_cost
    rebalance_cost = inception_cost.copy()

    for i in range(1, n_steps):
        tau = T - i * dt
        spot = paths[:, i]
        greeks = bs_greeks(spot, K, tau, r, sigma_pricing, q, kind)
        target = -np.asarray(greeks.delta, dtype=float)
        trade = target - shares
        trade_cost = tc * np.abs(trade) * spot

        cash = cash * np.exp(r * dt) - trade * spot - trade_cost
        shares = target
        rebalance_cost += trade_cost

    # Final period, then liquidate the stock leg at expiry.
    expiry_spot = paths[:, -1]
    if kind == "call":
        payoff = np.maximum(expiry_spot - K, 0.0)
    else:
        payoff = np.maximum(K - expiry_spot, 0.0)

    cash = cash * np.exp(r * dt)
    unwind_cost = tc * np.abs(shares) * expiry_spot
    pnl = payoff + shares * expiry_spot + cash - unwind_cost
    rebalance_cost += unwind_cost

    return HedgeResult(
        pnl=pnl,
        n_paths=n_paths,
        n_steps=n_steps,
        sigma_pricing=float(sigma_pricing),
        sigma_realized=float(sigma_realized),
        transaction_cost_bps=float(transaction_cost_bps),
        rebalance_cost=rebalance_cost,
    )
