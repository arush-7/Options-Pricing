"""Black-Scholes-Merton closed-form price and analytic Greeks.

Model assumptions
-----------------
The underlying follows geometric Brownian motion under the risk-neutral
measure with constant drift ``r - q`` and constant volatility ``sigma``::

    dS_t = (r - q) S_t dt + sigma S_t dW_t

which gives lognormal S_T. The price of a European option is the discounted
risk-neutral expectation of its payoff, which for a call is

    C = S e^{-qT} N(d1) - K e^{-rT} N(d2)

    d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

All functions are vectorised: pass scalars and you get floats, pass numpy
arrays and you get arrays (numpy broadcasting rules apply), which is what the
Monte Carlo and surface code relies on.

Conventions (important when comparing with other libraries)
-----------------------------------------------------------
Vega is per *unit* volatility, i.e. the change in price for sigma going from
0.20 to 0.21 is ``vega * 0.01``. Theta and rho are per *year* and per *unit*
rate respectively. Delta is per unit of the underlying (not per contract).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_VALID_KINDS = ("call", "put")


@dataclass(frozen=True)
class Greeks:
    """First- and second-order sensitivities of a European option.

    Attributes
    ----------
    delta : float or ndarray
        dV/dS.
    gamma : float or ndarray
        d2V/dS2.
    vega : float or ndarray
        dV/dsigma, per unit of volatility (so 1% vol move = ``vega * 0.01``).
    theta : float or ndarray
        dV/dt (calendar time decay), per year. Usually negative for long options.
    rho : float or ndarray
        dV/dr, per unit of rate.
    """

    delta: Any
    gamma: Any
    vega: Any
    theta: Any
    rho: Any

    def as_dict(self) -> dict[str, Any]:
        """Return the Greeks as a plain dict (handy for DataFrames)."""
        return asdict(self)


def _validate_kind(kind: str) -> str:
    kind = kind.lower()
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
    return kind


def _as_arrays(*values: Any) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(v, dtype=float) for v in values)


def d1_d2(
    S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Black-Scholes ``d1`` and ``d2`` terms.

    ``T`` or ``sigma`` of zero make ``d1``/``d2`` undefined (division by zero);
    in that case the valid-region guard substitutes ``T=1, sigma=1`` so the
    returned numbers are finite. Callers should treat those points as
    degenerate via :func:`bs_price` / :func:`bs_greeks`, which handle them
    explicitly. ``d1`` and ``d2`` are only meaningful where ``T > 0`` and
    ``sigma > 0``.
    """
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    ok = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    T_safe = np.where(ok, T, 1.0)
    sigma_safe = np.where(ok, sigma, 1.0)
    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(np.where(ok, S, 1.0) / np.where(ok, K, 1.0)) + (r - q + 0.5 * sigma_safe**2) * T_safe) / (
        sigma_safe * sqrt_T
    )
    d2 = d1 - sigma_safe * sqrt_T
    return d1, d2


def bs_price(
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    sigma: Any,
    q: Any = 0.0,
    kind: str = "call",
) -> Any:
    """European Black-Scholes-Merton price.

    Parameters
    ----------
    S, K, T : float or array
        Spot, strike and time to expiry in years.
    r, q : float or array
        Continuously compounded risk-free rate and dividend/carry yield.
    sigma : float or array
        Volatility (0.20 means 20%).
    kind : {"call", "put"}

    Returns
    -------
    float or ndarray
        A Python float when every input is scalar, otherwise an array.

    Notes
    -----
    Degenerate inputs are handled rather than raising: ``T <= 0`` returns the
    payoff ``max(S-K, 0)`` (``max(K-S, 0)`` for a put) and ``sigma <= 0``
    returns the discounted forward intrinsic value, which is the correct
    deterministic limit.
    """
    kind = _validate_kind(kind)
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    scalar_input = all(np.ndim(v) == 0 for v in (S, K, T, r, sigma, q))

    live = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)

    T_safe = np.where(live, T, 1.0)
    r_safe = np.where(live, r, 0.0)
    q_safe = np.where(live, q, 0.0)
    S_safe = np.where(live, S, 1.0)
    K_safe = np.where(live, K, 1.0)

    disc_r = np.exp(-r_safe * T_safe)
    disc_q = np.exp(-q_safe * T_safe)

    if kind == "call":
        price = S_safe * disc_q * norm.cdf(d1) - K_safe * disc_r * norm.cdf(d2)
        payoff = np.maximum(S - K, 0.0)
        forward_intrinsic = np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    else:
        price = K_safe * disc_r * norm.cdf(-d2) - S_safe * disc_q * norm.cdf(-d1)
        payoff = np.maximum(K - S, 0.0)
        forward_intrinsic = np.maximum(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    # sigma == 0 and T > 0: deterministic forward, still discounted.
    price = np.where(sigma > 0, price, forward_intrinsic)
    # Expired option: undiscounted payoff.
    price = np.where(T > 0, price, payoff)

    return float(price) if scalar_input else price


def bs_vega(
    S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any = 0.0
) -> Any:
    """Vega of a European option (identical for calls and puts).

    Kept separate because the implied vol solver needs it on every Newton
    step, and because it is the derivative that makes deep OTM inversion hard.
    """
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    live = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    vega = np.where(
        live,
        S * np.exp(-q * np.where(T > 0, T, 0.0)) * norm.pdf(d1) * np.sqrt(np.where(T > 0, T, 1.0)),
        0.0,
    )
    return float(vega) if np.ndim(vega) == 0 else vega


def bs_greeks(
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    sigma: Any,
    q: Any = 0.0,
    kind: str = "call",
) -> Greeks:
    """Analytic Black-Scholes Greeks.

    Formulas (``N`` is the standard normal CDF, ``phi`` the PDF):

        delta_call = e^{-qT} N(d1)
        delta_put  = -e^{-qT} N(-d1)
        gamma      = e^{-qT} phi(d1) / (S sigma sqrt(T))
        vega       = S e^{-qT} phi(d1) sqrt(T)
        theta_call = -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) - r K e^{-rT} N(d2)
                     + q S e^{-qT} N(d1)
        theta_put  = -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) + r K e^{-rT} N(-d2)
                     - q S e^{-qT} N(-d1)
        rho_call   =  K T e^{-rT} N(d2)
        rho_put    = -K T e^{-rT} N(-d2)
    """
    kind = _validate_kind(kind)
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    live = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)

    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    T_safe = np.where(live, T, 1.0)
    S_safe = np.where(live, S, 1.0)
    sqrt_T = np.sqrt(T_safe)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * np.where(live, T, 0.0))
    disc_r = np.exp(-r * np.where(live, T, 0.0))
    carry = np.where(live, sigma, 1.0)

    gamma_live = disc_q * pdf_d1 / (S_safe * carry * sqrt_T)
    vega_live = S_safe * disc_q * pdf_d1 * sqrt_T
    decay = -S_safe * disc_q * pdf_d1 * carry / (2.0 * sqrt_T)

    if kind == "call":
        delta_live = disc_q * norm.cdf(d1)
        theta_live = decay - r * K * disc_r * norm.cdf(d2) + q * S_safe * disc_q * norm.cdf(d1)
        rho_live = K * disc_r * sqrt_T * sqrt_T * norm.cdf(d2)
    else:
        delta_live = -disc_q * norm.cdf(-d1)
        theta_live = decay + r * K * disc_r * norm.cdf(-d2) - q * S_safe * disc_q * norm.cdf(-d1)
        rho_live = -K * disc_r * sqrt_T * sqrt_T * norm.cdf(-d2)

    # Limits as T -> 0: the option becomes its payoff, delta a step function.
    itm = S > K
    atm = S == K
    if kind == "call":
        delta_expiry = np.where(itm, 1.0, np.where(atm, 0.5, 0.0))
    else:
        delta_expiry = np.where(itm, 0.0, np.where(atm, -0.5, -1.0))

    zero = np.zeros_like(gamma_live)
    zeros = (zero,) * 4
    delta = np.where(live, delta_live, delta_expiry)
    gamma = np.where(live, gamma_live, zero)
    vega = np.where(live, vega_live, zero)
    theta = np.where(live, theta_live, zeros[0])
    rho = np.where(live, rho_live, zeros[1])

    def _out(x: np.ndarray) -> Any:
        return float(x) if np.ndim(x) == 0 else x

    return Greeks(delta=_out(delta), gamma=_out(gamma), vega=_out(vega), theta=_out(theta), rho=_out(rho))


def put_call_parity_gap(
    call_price: Any, put_price: Any, S: Any, K: Any, T: Any, r: Any, q: Any = 0.0
) -> Any:
    """Return ``C - P - (S e^{-qT} - K e^{-rT})``.

    Zero for a consistent set of quotes. Used in the tests to validate both the
    formula and its implementation, and in the data cleaning to spot broken
    quotes before they reach the surface fit.
    """
    gap = (
        np.asarray(call_price, dtype=float)
        - np.asarray(put_price, dtype=float)
        - (np.asarray(S, dtype=float) * np.exp(-np.asarray(q, float) * np.asarray(T, float))
           - np.asarray(K, dtype=float) * np.exp(-np.asarray(r, float) * np.asarray(T, float)))
    )
    return float(gap) if np.ndim(gap) == 0 else gap
