"""Bump-and-reprice Greeks: differentiate *any* pricer numerically.

Why not always use the analytic formulas?
-----------------------------------------
Because most things you actually price have no closed form: American options
on a tree, barrier options, anything Monte Carlo. If a pricer is a function
``price(**params) -> float``, you can get its sensitivities by nudging one
parameter and re-pricing. That is what this module does, with central
differences wherever the bump is legal.

The price of generality is accuracy: finite differences suffer from
subtractive cancellation (too small a bump) and from truncation error (too
large a bump). The defaults below are a compromise that reproduces the
analytic Black-Scholes Greeks to roughly 1e-6 relative error, which the test
suite checks. For a tree pricer, the *bump* must be large enough to dominate
the tree's own discretisation noise, so use ``rel_bump`` around 1e-3 there.

Definitions follow :class:`optlab.pricing.black_scholes.Greeks`: vega per unit
vol, theta per year, rho per unit rate, all with respect to a long position.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from optlab.pricing.black_scholes import Greeks, bs_price

Pricer = Callable[..., Any]


def bump_and_reprice(
    pricer: Pricer,
    *,
    parameter: str,
    bump: float,
    **params: Any,
) -> float:
    """Price the option with one parameter moved by ``bump``.

    ``params`` must contain the pricer's keyword arguments; ``parameter`` names
    the one to shift. Raises ``KeyError`` if the parameter is not supplied, so
    typos surface immediately instead of silently pricing the unbumped case.
    """
    if parameter not in params:
        raise KeyError(f"{parameter!r} is not among the pricing parameters {sorted(params)}")
    shifted = dict(params)
    shifted[parameter] = shifted[parameter] + bump
    return float(pricer(**shifted))


def finite_difference_greeks(
    pricer: Pricer = bs_price,
    *,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
    rel_bump: float = 1e-4,
    pricer_kwargs: dict[str, Any] | None = None,
) -> Greeks:
    """Compute delta, gamma, vega, theta and rho by finite differences.

    Parameters
    ----------
    pricer : callable
        Anything that accepts ``S, K, T, r, sigma, q, kind`` as keyword
        arguments and returns a price. ``bs_price`` and ``binomial_price`` both
        satisfy this, so ``pricer=binomial_price, pricer_kwargs={"steps": 500}``
        works out of the box.
    rel_bump : float
        Bump size as a fraction of the parameter's value (for ``r`` the bump is
        absolute, since rates can be negative or near zero).
    pricer_kwargs : dict, optional
        Extra keyword arguments forwarded to the pricer (e.g. ``steps``).

    Returns
    -------
    Greeks

    Notes
    -----
    * delta, gamma: central differences in ``S`` (gamma needs the second
      difference, so it is the noisiest of the set).
    * vega: central difference in ``sigma``, falling back to a one-sided
      difference if ``sigma - h <= 0``.
    * theta: central difference in ``T`` when ``T - h > 0``, otherwise a
      one-sided difference toward expiry. Sign convention is ``-dV/dT`` so it
      matches the analytic theta (decay as calendar time passes).
    * rho: central difference in ``r`` with an absolute bump.
    """
    extra = dict(pricer_kwargs or {})
    base = dict(S=float(S), K=float(K), T=float(T), r=float(r), sigma=float(sigma), q=float(q), kind=kind)

    def price(**overrides: Any) -> float:
        call = {**base, **overrides}
        return float(pricer(**call, **extra))

    h = float(rel_bump)
    h_S = h * base["S"]
    if h_S == 0.0:
        raise ValueError("S must be non-zero to bump spot")

    # Spot: central differences, gamma from the same three prices.
    p_up = price(S=base["S"] + h_S)
    p_down = price(S=base["S"] - h_S)
    p_mid = price()
    delta = (p_up - p_down) / (2.0 * h_S)
    gamma = (p_up - 2.0 * p_mid + p_down) / (h_S**2)

    # Volatility.
    h_sigma = max(h * base["sigma"], 1e-8)
    if base["sigma"] - h_sigma > 0.0:
        vega = (price(sigma=base["sigma"] + h_sigma) - price(sigma=base["sigma"] - h_sigma)) / (
            2.0 * h_sigma
        )
    else:
        vega = (price(sigma=base["sigma"] + h_sigma) - p_mid) / h_sigma

    # Time to expiry. Bump forward is always legal; bump backward is not if it
    # would cross today, so use a one-sided difference there.
    h_T = max(h * base["T"], 1e-8)
    if base["T"] - h_T > 0.0:
        theta = -(price(T=base["T"] + h_T) - price(T=base["T"] - h_T)) / (2.0 * h_T)
    else:
        theta = -(price(T=base["T"] + h_T) - p_mid) / h_T

    # Rate: absolute bump, since a relative bump makes no sense at r = 0.
    h_r = max(h, 1e-8)
    rho = (price(r=base["r"] + h_r) - price(r=base["r"] - h_r)) / (2.0 * h_r)

    return Greeks(delta=float(delta), gamma=float(gamma), vega=float(vega), theta=float(theta), rho=float(rho))


def greeks_error_table(analytic: Greeks, numerical: Greeks) -> dict[str, float]:
    """Absolute error per Greek, for validation notebooks and tests."""
    return {
        name: abs(getattr(numerical, name) - getattr(analytic, name))
        for name in ("delta", "gamma", "vega", "theta", "rho")
    }
