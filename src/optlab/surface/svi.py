"""SVI (Stochastic Volatility Inspired) parametrisation of one expiry slice.

Model
-----
Collar the implied total variance (``w = sigma_BS^2 * T``) as a function of
log-moneyness ``k = ln(K / F)``::

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma_svi^2) )

Five parameters per expiry:

* ``a``     overall level of variance,
* ``b``     wing slope (b >= 0),
* ``rho``   skew / asymmetry (-1 < rho < 1); equity skew means rho < 0,
* ``m``     horizontal shift of the minimum,
* ``sigma_svi``  smoothness of the vertex (note: *not* a volatility; the naming
  collision with ``sigma`` is unfortunate but standard in the literature).

Analytic derivatives (used for the arbitrage checks)::

    w'(k)  = b * (rho + (k-m)/d),                 d = sqrt((k-m)^2 + sigma_svi^2)
    w''(k) = b * sigma_svi^2 / d^3

Why SVI rather than a cubic spline through the quotes?
------------------------------------------------------
Five parameters per expiry means the fit is smooth and stable, it extrapolates
like a wing (linearly in total variance, which is what the market does), and
``w'' >= 0`` is guaranteed for the central region — which is the condition that
rules out negative risk-neutral densities (butterfly arbitrage). A spline
through noisy mids will happily produce a negative density.

Fitting is a bounded nonlinear least squares problem, solved with
``scipy.optimize.least_squares`` from several starting points; we keep the best
fit. This is not a convex problem, so the multi-start matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

# |rho| is bounded away from 1 so the wings stay finite.
_RHO_MAX = 0.999


@dataclass(frozen=True)
class SVIParams:
    """Parameters of one SVI slice, in total-variance space."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    # -- total variance -----------------------------------------------------
    def total_variance(self, k):
        """``w(k)``: implied variance times time to expiry."""
        k = np.asarray(k, dtype=float)
        u = k - self.m
        return self.a + self.b * (self.rho * u + np.sqrt(u * u + self.sigma**2))

    def dw_dk(self, k):
        """First derivative of total variance with respect to log-moneyness."""
        k = np.asarray(k, dtype=float)
        u = k - self.m
        d = np.sqrt(u * u + self.sigma**2)
        return self.b * (self.rho + u / d)

    def d2w_dk2(self, k):
        """Second derivative of total variance with respect to log-moneyness."""
        k = np.asarray(k, dtype=float)
        u = k - self.m
        d = np.sqrt(u * u + self.sigma**2)
        return self.b * self.sigma**2 / d**3

    # -- implied vol --------------------------------------------------------
    def iv(self, k, T):
        """Implied Black-Scholes volatility at log-moneyness ``k``, expiry ``T``."""
        w = np.maximum(self.total_variance(k), 1e-16)
        return np.sqrt(w / T)

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.a, self.b, self.rho, self.m, self.sigma)

    def as_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "rho": self.rho, "m": self.m, "sigma": self.sigma}


def _residuals(theta, k, w, weights, arb_grid, arb_penalty):
    a, b, rho, m, sigma = theta
    u = k - m
    d = np.sqrt(u * u + sigma * sigma)
    model = a + b * (rho * u + d)
    res = (model - w) * np.sqrt(weights)
    if arb_penalty > 0.0 and arb_grid is not None:
        # Penalise (softly) any point where Gatheral's g-function goes negative,
        # i.e. where the implied density would be negative.
        p = SVIParams(a, b, rho, m, sigma)
        g = gatheral_g(p, arb_grid)
        violation = np.minimum(g, 0.0)
        res = np.concatenate([res, arb_penalty * violation])
    return res


def gatheral_g(params: SVIParams, k) -> np.ndarray:
    """Gatheral's g-function; ``g(k) >= 0`` for all k means no butterfly arbitrage.

    .. math::

        g(k) = \\left(1 - \\frac{k w'}{2w}\\right)^2
               - \\frac{w'}{4}\\left(\\frac{1}{w} + \\frac{1}{4}\\right)
               + \\frac{w''}{2}

    A negative value means the risk-neutral density implied by the slice is
    negative somewhere — a static arbitrage.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(params.total_variance(k), dtype=float)
    wp = np.asarray(params.dw_dk(k), dtype=float)
    wpp = np.asarray(params.d2w_dk2(k), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        first = (1.0 - k * wp / (2.0 * w)) ** 2
        second = (wp / 4.0) * (1.0 / w + 0.25)
    return first - second + wpp / 2.0


def svi_rmse(params: SVIParams, k, w) -> float:
    """Root-mean-square error of the fit in total variance units."""
    k = np.asarray(k, dtype=float)
    return float(np.sqrt(np.mean((params.total_variance(k) - np.asarray(w, dtype=float)) ** 2)))


def iv_rmse(params: SVIParams, k, iv, T) -> float:
    """RMSE expressed in *volatility points* (the number a trader cares about)."""
    k = np.asarray(k, dtype=float)
    return float(np.sqrt(np.mean((params.iv(k, T) - np.asarray(iv, dtype=float)) ** 2)))


def fit_svi(
    k,
    w,
    weights=None,
    *,
    n_starts: int = 12,
    arb_penalty: float = 1e-2,
    seed: int | None = 0,
    max_nfev: int = 2000,
) -> SVIParams:
    """Fit SVI parameters to total variance points ``w = sigma^2 T`` at ``k``.

    Parameters
    ----------
    k : array
        Log-moneyness ``ln(K/F)``. At least 5 distinct points are required.
    w : array
        Total implied variance ``sigma_BS^2 * T``.
    weights : array, optional
        Relative weights (e.g. inverse bid-ask variance). Defaults to equal.
    n_starts : int
        Number of random starts; the best fit by sum of squares is returned.
        SVI fitting is non-convex, so this is what keeps results stable.
    arb_penalty : float
        Soft penalty weight pushing Gatheral's g-function to stay >= 0. Set to
        0 to see what an unconstrained fit does (the notebook compares them).

    Returns
    -------
    SVIParams
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if k.shape != w.shape:
        raise ValueError("k and w must have the same shape")
    if k.size < 5:
        raise ValueError(f"SVI needs at least 5 points to pin down 5 parameters, got {k.size}")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(w)):
        raise ValueError("k and w must be finite")
    if np.any(w <= 0):
        raise ValueError("total variance w must be strictly positive")

    order = np.argsort(k)
    k, w = k[order], w[order]
    weights = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)[order]
    weights = np.maximum(weights, 1e-12)

    k_min, k_max = float(k.min()), float(k.max())
    span = max(k_max - k_min, 1e-3)
    # Arbitrage penalty grid: a bit wider than the quoted range, since a fit can
    # be arbitrageable just outside the data.
    arb_grid = np.linspace(k_min - 0.5 * span, k_max + 0.5 * span, 80) if arb_penalty > 0 else None

    lower = np.array([-1.0, 0.0, -_RHO_MAX, k_min - 2.0 * span, 1e-3])
    upper = np.array([1.0, 2.0, _RHO_MAX, k_max + 2.0 * span, 3.0])

    rng = np.random.default_rng(seed)
    atm_var = float(np.interp(0.0, k, w))
    base = np.array([max(atm_var * 0.5, 1e-6), 0.1, -0.3, float(k[np.argmin(w)]), 0.1])

    starts = [base]
    for _ in range(max(n_starts - 1, 0)):
        perturbed = base * (1.0 + 0.6 * rng.standard_normal(base.shape))
        perturbed[0] = abs(perturbed[0]) + 1e-6
        perturbed[1] = abs(perturbed[1]) + 1e-6
        perturbed[2] = np.clip(perturbed[2], -0.9, 0.4)
        perturbed[3] = base[3] + 0.3 * span * rng.standard_normal()
        perturbed[4] = abs(perturbed[4]) + 1e-3
        starts.append(np.clip(perturbed, lower + 1e-9, upper - 1e-9))

    best: SVIParams | None = None
    best_cost = np.inf
    for x0 in starts:
        x0 = np.clip(x0, lower + 1e-9, upper - 1e-9)
        try:
            sol = least_squares(
                _residuals,
                x0,
                args=(k, w, weights, arb_grid, arb_penalty),
                bounds=(lower, upper),
                max_nfev=max_nfev,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
            )
        except (ValueError, FloatingPointError):
            continue
        if sol.cost < best_cost:
            best_cost = float(sol.cost)
            best = SVIParams(*[float(v) for v in sol.x])

    if best is None:
        raise RuntimeError("SVI fit failed from every starting point")
    return best


def fit_svi_slice(
    strikes,
    ivs,
    T: float,
    forward: float | None = None,
    spot: float | None = None,
    r: float = 0.0,
    q: float = 0.0,
    weights=None,
    **kwargs,
) -> SVIParams:
    """Fit one expiry slice from quoted strikes and implied vols.

    ``forward`` is the natural moneyness anchor; if omitted it is built from
    ``spot * exp((r - q) T)``. Fitting in log-moneyness relative to the forward
    is what makes the skew parameters comparable across expiries.
    """
    strikes = np.asarray(strikes, dtype=float)
    ivs = np.asarray(ivs, dtype=float)
    if T <= 0:
        raise ValueError("T must be positive")
    if forward is None:
        if spot is None:
            raise ValueError("provide either forward or spot")
        forward = float(spot) * float(np.exp((r - q) * T))
    if forward <= 0:
        raise ValueError("forward must be positive")

    k = np.log(strikes / float(forward))
    w = ivs**2 * T
    return fit_svi(k, w, weights=weights, **kwargs)
