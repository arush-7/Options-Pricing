"""Implied volatility: invert the price back into a volatility.

Problem
-------
Given a market price ``P`` and the model inputs ``S, K, T, r, q``, find the
``sigma`` such that ``bs_price(..., sigma) == P``. The Black-Scholes price is
strictly increasing in ``sigma`` for a vanilla, so there is at most one root,
but there is no closed form: it must be solved numerically.

Method
------
Newton-Raphson uses the derivative we already have (vega)::

    sigma_{n+1} = sigma_n - (price(sigma_n) - P) / vega(sigma_n)

It converges quadratically where vega is healthy, i.e. near the money. Deep
out-of-the-money, vega collapses toward zero and Newton either crawls or jumps
outside the admissible region. So the solver clamps each step to
``[1e-8, 5.0]`` (0% to 500% vol) and falls back to Brent's method, which is
slower but bracketed and cannot diverge. This "fast path with a safe fallback"
is the standard production pattern.

Edge cases, all of which raise :class:`ImpliedVolError` rather than returning
a bogus number:

* price below the discounted intrinsic value — no positive vol can produce it;
* price at or above the no-arbitrage cap (``S e^{-qT}`` for a call);
* a zero or negative price;
* no bracket, meaning even 500% volatility is not expensive enough.

Prices very close to the intrinsic value are legitimately near-zero vol, which
is numerically hostile: the answer is sensitive to quote noise. That is a data
problem, and :mod:`optlab.data.cleaning` drops those rows instead of pretending
the inversion is reliable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from optlab.pricing.black_scholes import _validate_kind, bs_price, bs_vega

SIGMA_FLOOR = 1e-8
SIGMA_CAP = 5.0


class ImpliedVolError(ValueError):
    """Raised when a price cannot be inverted into a volatility."""


@dataclass(frozen=True)
class IVResult:
    """Outcome of an implied volatility inversion."""

    value: float
    iterations: int
    method: str
    converged: bool
    price: float
    lower_bound: float
    upper_bound: float

    def __float__(self) -> float:
        return self.value


def no_arbitrage_bounds(
    S: float, K: float, T: float, r: float, q: float = 0.0, kind: str = "call"
) -> tuple[float, float]:
    """Return ``(lower, upper)`` admissible prices for a European option.

    The lower bound is the discounted intrinsic value (a European option is
    worth at least its forward intrinsic; no early exercise to lean on). The
    upper bound is the discounted spot for a call and the discounted strike for
    a put.
    """
    kind = _validate_kind(kind)
    disc_r = float(np.exp(-r * T))
    disc_q = float(np.exp(-q * T))
    if kind == "call":
        lower = max(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:
        lower = max(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r
    return lower, upper


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    kind: str = "call",
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
    method: str = "auto",
    return_details: bool = False,
) -> float | IVResult:
    """Invert a European option price into an implied volatility.

    Parameters
    ----------
    price : float
        Observed (mid) price.
    S, K, T, r, q, kind
        Contract and market inputs, as in :func:`optlab.pricing.black_scholes.bs_price`.
    tol : float
        Convergence tolerance in price units.
    method : {"auto", "newton", "brent"}
        ``"auto"`` (default) tries Newton and falls back to Brent.
    return_details : bool
        If True, return the full :class:`IVResult` instead of a float.

    Raises
    ------
    ImpliedVolError
        If the price violates the no-arbitrage bounds or no bracket can be found.
    """
    kind = _validate_kind(kind)
    if T <= 0:
        raise ImpliedVolError("cannot imply a volatility from an expired option (T <= 0)")
    if price <= 0:
        raise ImpliedVolError(f"price must be positive, got {price!r}")

    price = float(price)
    lower, upper = no_arbitrage_bounds(S, K, T, r, q, kind)
    slack = 1e-12 * max(1.0, S)
    if price < lower - slack:
        raise ImpliedVolError(
            f"price {price:.6f} is below the discounted intrinsic value {lower:.6f}; "
            "no non-negative volatility reproduces it (stale or crossed quote?)"
        )
    if price > upper + slack:
        raise ImpliedVolError(
            f"price {price:.6f} exceeds the no-arbitrage cap {upper:.6f}"
        )
    if price <= lower + slack:
        raise ImpliedVolError(
            f"price {price:.6f} sits on the discounted intrinsic value {lower:.6f}; "
            "implied volatility is zero/undefined at this precision"
        )

    q_ = float(q)
    r_ = float(r)

    def residual(sigma: float) -> float:
        return bs_price(S, K, T, r_, sigma, q_, kind) - price

    # Brenner-Subrahmanyam style start: good near the money, and harmless
    # elsewhere because Newton is clamped and Brent backs it up.
    with np.errstate(divide="ignore", invalid="ignore"):
        guess = float(np.sqrt(2.0 * np.pi / T) * price / S) if S > 0 else 0.2
    if not np.isfinite(guess) or guess <= 0.0:
        guess = 0.2
    guess = float(np.clip(guess, 1e-3, 2.0))

    details: dict[str, object] = {}
    if method in ("auto", "newton"):
        sigma, iterations, converged = _newton(
            residual, S, K, T, r_, q_, kind, guess, tol * max(1.0, abs(price)), max_iter
        )
        if converged:
            details = dict(value=sigma, iterations=iterations, method="newton", converged=True)
    if not details:
        if method == "newton":
            raise ImpliedVolError(
                f"Newton did not converge in {max_iter} iterations and method='newton' "
                "forbids the Brent fallback"
            )
        value, iterations = _brent(residual, price, lower, upper)
        details = dict(value=value, iterations=iterations, method="brent", converged=True)

    result = IVResult(
        value=float(details["value"]),
        iterations=int(details["iterations"]),
        method=str(details["method"]),
        converged=bool(details["converged"]),
        price=price,
        lower_bound=lower,
        upper_bound=upper,
    )
    return result if return_details else result.value


def _newton(
    residual,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    kind: str,
    sigma0: float,
    price_tol: float,
    max_iter: int,
) -> tuple[float, int, bool]:
    """Clamped Newton iteration. Returns ``(sigma, iterations, converged)``."""
    sigma = sigma0
    for i in range(1, max_iter + 1):
        diff = residual(sigma)
        if abs(diff) < price_tol:
            return sigma, i, True
        vega = float(bs_vega(S, K, T, r, sigma, q))
        if not np.isfinite(vega) or vega < 1e-10:
            return sigma, i, False  # flat payoff region: let Brent handle it
        step = diff / vega
        sigma_next = sigma - step
        if not np.isfinite(sigma_next):
            return sigma, i, False
        sigma = float(np.clip(sigma_next, SIGMA_FLOOR, SIGMA_CAP))
    return sigma, max_iter, False


def _brent(residual, price: float, lower: float, upper: float) -> tuple[float, int]:
    """Bracketed solve on [SIGMA_FLOOR, SIGMA_CAP]; raises if not bracketed."""
    lo, hi = SIGMA_FLOOR, SIGMA_CAP
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo * f_hi > 0.0:
        raise ImpliedVolError(
            f"no volatility in [{lo}, {hi}] reproduces the price {price:.6f} "
            f"(residuals {f_lo:.3e} and {f_hi:.3e}); the quote is likely outside "
            "the no-arbitrage range for these inputs"
        )
    value = brentq(residual, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=200)
    return float(value), -1
