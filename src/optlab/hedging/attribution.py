"""Decomposing a delta-hedged PnL into theta, gamma and vega.

The Taylor expansion that everyone quotes
-----------------------------------------
For one step of calendar time ``dt``, with the underlying moving ``dS`` and vol
moving ``dsigma``::

    dV = Theta*dt + Delta*dS + 1/2 * Gamma*dS^2 + Vega*dsigma + (higher order)

A delta-hedged book has the ``Delta*dS`` term cancelled, but the hedge still has
to be financed: holding ``+1`` option and ``-Delta`` shares means borrowing
``V - Delta*S``, which costs ``r (V - Delta*S) dt``. So the book's PnL is::

    hedged PnL per step = Theta*dt + 1/2 * Gamma*dS^2 + Vega*dsigma
                          - r*(V - Delta*S)*dt + residual

* ``Theta*dt``        — the rent you pay for being long optionality (negative).
  By the Black-Scholes PDE this term already contains the option's own variance
  cost, ``-1/2 sigma^2 S^2 Gamma``, which is what the gamma term pays back.
* ``1/2 Gamma dS^2``  — what you earn from realised movement (the gamma rent).
  Since ``E[dS^2] = sigma_realised^2 S^2 dt``, the pair of terms collapses to the
  classic ``1/2 Gamma S^2 (sigma_r^2 - sigma_i^2) dt``.
* ``Vega*dsigma``     — mark-to-market of a vol move; zero if you hold to expiry
  with vol pinned, which is why a full delta-gamma-vega book is repriced, not held.
* ``financing``       — carry on the borrowed cash needed to hold the hedge.
  Leaving this out is the classic way to make theta and gamma appear not to add up.
  With ``q = 0`` the theta + gamma + financing expectation is exactly zero when
  realised vol equals implied vol.
* ``residual``        — higher-order terms plus the discrete-rebalancing error.

Reporting the residual explicitly is the honest part: if it is large relative to
the components, the decomposition is not telling you the story.

This module never trades anything — it just explains a PnL path you already have,
which is exactly how a trader uses a risk report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from optlab.pricing.black_scholes import bs_greeks, bs_price, _validate_kind

@dataclass
class PnlAttribution:
    """Per-path PnL attribution and its aggregate summary.

    ``components`` has one row per path with columns ``delta``, ``gamma``,
    ``theta``, ``vega``, ``finance``, ``residual``, ``hedged_total`` and
    ``unhedged_total``. ``summary()`` collapses it to means, which is what a risk
    report shows.
    """

    components: pd.DataFrame
    n_steps: int
    dt: float

    def summary(self) -> pd.DataFrame:
        mean = self.components.mean(numeric_only=True)
        backtested = mean["hedged_total"]
        explained = mean["theta"] + mean["gamma"] + mean["vega"] + mean["finance"]
        scale = sum(abs(mean[c]) for c in ("theta", "gamma", "vega", "finance"))
        # The share is meaningless when the total is ~0 (matched vol, no costs).
        share = explained / backtested if abs(backtested) > 1e-3 * scale else np.nan
        rows = [
            ("theta", mean["theta"]),
            ("gamma", mean["gamma"]),
            ("vega", mean["vega"]),
            ("financing", mean["finance"]),
            ("(residual)", mean["residual"]),
            ("= hedged PnL", backtested),
            ("(explained share)", share),
        ]
        return pd.DataFrame(rows, columns=["component", "mean PnL"])

    def to_frame(self) -> pd.DataFrame:
        return self.components

    def __repr__(self) -> str:
        return (
            f"PnlAttribution(paths={len(self.components)}, steps={self.n_steps}, "
            f"mean hedged PnL={self.components['hedged_total'].mean():+.4f})"
        )


def decompose_pnl(
    paths: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str = "call",
    q: float = 0.0,
    sigma_path: np.ndarray | None = None,
) -> PnlAttribution:
    """Attribute the PnL of a discretely delta-hedged option position.

    Parameters
    ----------
    paths : ndarray of shape (n_paths, n_steps + 1)
        Underlying price paths on the hedging grid, one path per row — exactly
        what :func:`optlab.pricing.monte_carlo.simulate_gbm_paths` returns. A
        single 1-D path of length ``n_steps + 1`` is also accepted.
    K, T, r, sigma, kind, q
        Contract and pricing inputs; ``sigma`` is the vol the option is marked at
        (the implied vol you traded).
    sigma_path : ndarray, optional
        Vol on the same grid, either ``(n_steps + 1,)`` or
        ``(n_paths, n_steps + 1)``. When given, ``vega * dsigma`` is included;
        otherwise the vega component is zero and any vol move shows up in the
        residual.

    Returns
    -------
    PnlAttribution

    Notes
    -----
    Signs are for a *long* option: theta is the decay you pay, gamma the squared
    move you collect, and financing is the carry on the borrowed hedge. The hedge
    is rebalanced at the start of each step, which is the convention that keeps
    the decomposition and the realised hedged PnL consistent — including the
    ``finance`` leg, without which theta and gamma would appear not to add up.
    """
    kind = _validate_kind(kind)
    # Layout matches simulate_gbm_paths: one path per row, time along columns.
    paths = np.asarray(paths, dtype=float)
    squeeze = paths.ndim == 1
    if squeeze:
        paths = paths[None, :]
    if paths.ndim != 2:
        raise ValueError("paths must be 1-D (a single path) or 2-D (n_paths, n_steps + 1)")
    if paths.shape[1] < 2:
        raise ValueError("need at least two time steps to decompose PnL")

    n_steps = paths.shape[1] - 1
    dt = T / n_steps
    spot = paths[:, :-1]            # (n_paths, steps)
    spot_next = paths[:, 1:]
    d_spot = spot_next - spot
    tau = (T - np.arange(n_steps) * dt)[None, :]            # (1, steps) -> broadcasts
    tau_next = (T - np.arange(1, n_steps + 1) * dt)[None, :]

    greeks = bs_greeks(spot, K, tau, r, sigma, q, kind)
    delta = np.asarray(greeks.delta, dtype=float)
    gamma = np.asarray(greeks.gamma, dtype=float)
    theta = np.asarray(greeks.theta, dtype=float)
    vega = np.asarray(greeks.vega, dtype=float)

    value = bs_price(spot, K, tau, r, sigma, q, kind)
    value_next = bs_price(spot_next, K, tau_next, r, sigma, q, kind)
    step_change = np.asarray(value_next - value, dtype=float)

    if sigma_path is None:
        d_sigma = np.zeros_like(step_change)
    else:
        sp = np.asarray(sigma_path, dtype=float)
        if sp.ndim == 1:
            if sp.size != n_steps + 1:
                raise ValueError("a 1-D sigma_path must have n_steps + 1 entries")
            sp = np.broadcast_to(sp[None, :], paths.shape)
        if sp.shape != paths.shape:
            raise ValueError(
                "sigma_path must have shape (n_steps + 1,) or (n_paths, n_steps + 1)"
            )
        d_sigma = sp[:, 1:] - sp[:, :-1]

    delta_pnl = delta * d_spot
    gamma_pnl = 0.5 * gamma * d_spot**2
    theta_pnl = theta * dt
    vega_pnl = vega * d_sigma
    # Long option, short delta shares => borrowing V - Delta*S at the risk-free rate.
    finance_pnl = -r * (value - delta * spot) * dt
    hedged = step_change - delta_pnl + finance_pnl
    residual = hedged - (theta_pnl + gamma_pnl + vega_pnl + finance_pnl)

    totals = {
        "delta": delta_pnl.sum(axis=1),
        "gamma": gamma_pnl.sum(axis=1),
        "theta": theta_pnl.sum(axis=1),
        "vega": vega_pnl.sum(axis=1),
        "finance": finance_pnl.sum(axis=1),
        "residual": residual.sum(axis=1),
        "hedged_total": hedged.sum(axis=1),
        "unhedged_total": step_change.sum(axis=1),
    }
    return PnlAttribution(
        components=pd.DataFrame({name: np.atleast_1d(values) for name, values in totals.items()}),
        n_steps=n_steps,
        dt=dt,
    )
