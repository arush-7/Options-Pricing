"""Monte Carlo pricing under geometric Brownian motion.

Under GBM the terminal value has a closed form, so there is no need to step:
S_T is exactly lognormal.

    S_T = S_0 exp((r - q - sigma^2/2) T + sigma sqrt(T) Z),  Z ~ N(0, 1)

We still simulate *paths* (via :func:`simulate_gbm_paths`) because the hedging
module needs intermediate prices, but the European price only uses the terminal
value, which makes the estimator unbiased with no discretisation error.

Two variance reduction tricks are implemented, because they are what turns a
"Monte Carlo is too noisy" toy into something usable:

*Antithetic variates*
    Draw ``Z`` and ``-Z``. The pair averages out the symmetric part of the
    noise. The estimator stays unbiased and error is measured on pair averages,
    which is the correct unit of independent information.

*Control variate*
    For each sample we also track a quantity with a known expectation — here
    the discounted terminal stock ``e^{-rT} S_T``, whose expectation is
    ``S_0 e^{-qT}``. Regressing the payoff on it (coefficient ``beta``) and
    subtracting the known-mean part removes the part of the payoff's noise that
    is explained by the stock level. For a vanilla this is a modest gain; the
    technique matters far more for exotic payoffs, which is why it is worth
    having the machinery.

Standard errors are reported on the independent unit (pair when using
antithetics), so confidence intervals are honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optlab.pricing.black_scholes import _validate_kind


@dataclass(frozen=True)
class MCResult:
    """Outcome of a Monte Carlo price.

    ``price`` is the estimate, ``stderr`` its standard error and
    ``ci_low``/``ci_high`` a normal-approximation 95% interval. ``beta`` is the
    fitted control-variate coefficient (0.0 when no control variate is used).
    """

    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int
    n_units: int
    antithetic: bool
    control_variate: bool
    beta: float = 0.0

    def summary(self) -> str:
        red = []
        if self.antithetic:
            red.append("antithetic")
        if self.control_variate:
            red.append(f"control variate (beta={self.beta:.3f})")
        method = " + ".join(red) if red else "plain"
        return (
            f"MC price {self.price:.6f} +/- {1.96 * self.stderr:.6f} (95%) "
            f"[{method}, {self.n_paths:,} paths]"
        )


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    steps: int,
    n_paths: int,
    q: float = 0.0,
    antithetic: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate exact GBM paths on a fixed time grid.

    Returns
    -------
    ndarray of shape (n_paths, steps + 1)
        Column 0 is ``S0``; column ``steps`` is the terminal price. Using exact
        lognormal increments (rather than a first-order Euler scheme) means the
        marginal distribution at every grid point is exactly correct, so there
        is no discretisation bias to trade off against the step count.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if T <= 0:
        raise ValueError("T must be positive to simulate paths")
    if sigma <= 0:
        raise ValueError("sigma must be positive to simulate paths")

    rng = np.random.default_rng(seed)
    dt = T / steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    if antithetic:
        half = (n_paths + 1) // 2
        z_half = rng.standard_normal((half, steps))
        z = np.concatenate([z_half, -z_half], axis=0)[:n_paths]
    else:
        z = rng.standard_normal((n_paths, steps))

    log_increments = drift + diffusion * z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_paths, steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


def _terminal_prices(
    S: float,
    r: float,
    sigma: float,
    T: float,
    q: float,
    n_draws: int,
    antithetic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Terminal prices from ``n_draws`` normal draws (lognormal, exact)."""
    z = rng.standard_normal(n_draws)
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)
    if antithetic:
        z = np.concatenate([z, -z])
    return S * np.exp(drift + diffusion * z)


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100_000,
    q: float = 0.0,
    kind: str = "call",
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = None,
) -> MCResult:
    """European option price by Monte Carlo.

    Parameters
    ----------
    n_paths : int
        Total number of simulated terminal prices.
    antithetic, control_variate : bool
        Variance reduction switches; both default to on.
    seed : int, optional
        Seed the generator for reproducible output. Tests always pass one.

    Returns
    -------
    MCResult

    Examples
    --------
    >>> res = mc_price(100, 100, 1.0, 0.05, 0.2, n_paths=200_000, seed=7)
    >>> abs(res.price - bs_price(100, 100, 1.0, 0.05, 0.2)) < 0.02
    True
    """
    kind = _validate_kind(kind)
    if T <= 0:
        raise ValueError("T must be positive; an expired option has no simulated value")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")

    rng = np.random.default_rng(seed)
    n_draws = n_paths // 2 if antithetic else n_paths
    n_draws = max(n_draws, 1)

    terminal = _terminal_prices(S, r, sigma, T, q, n_draws, antithetic, rng)
    if kind == "call":
        payoff = np.maximum(terminal - K, 0.0)
    else:
        payoff = np.maximum(K - terminal, 0.0)
    discounted = np.exp(-r * T) * payoff
    discounted_spot = np.exp(-r * T) * terminal

    if antithetic:
        # Pair i and its mirror are one independent unit of information.
        x = 0.5 * (discounted[:n_draws] + discounted[n_draws:])
        y = 0.5 * (discounted_spot[:n_draws] + discounted_spot[n_draws:])
    else:
        x, y = discounted, discounted_spot

    n_units = x.size
    beta = 0.0
    if control_variate:
        var_y = float(np.var(y, ddof=1))
        if var_y > 0.0:
            beta = float(np.cov(x, y, ddof=1)[0, 1] / var_y)
            known_mean = S * np.exp(-q * T)  # E[e^{-rT} S_T]
            x = x - beta * (y - known_mean)

    price = float(np.mean(x))
    stderr = float(np.std(x, ddof=1) / np.sqrt(n_units)) if n_units > 1 else float("nan")
    return MCResult(
        price=price,
        stderr=stderr,
        ci_low=price - 1.96 * stderr,
        ci_high=price + 1.96 * stderr,
        n_paths=2 * n_units if antithetic else n_units,
        n_units=n_units,
        antithetic=antithetic,
        control_variate=control_variate,
        beta=beta,
    )
