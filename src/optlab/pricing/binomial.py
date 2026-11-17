"""Cox-Ross-Rubinstein binomial tree: European and American options.

The tree
--------
Split ``T`` into ``steps`` intervals of length ``dt = T / steps`` and let the
stock move up by ``u`` or down by ``d = 1/u`` per step::

    u = exp(sigma sqrt(dt)),  d = 1/u
    p = (exp((r - q) dt) - d) / (u - d)      risk-neutral up probability

With these choices the tree reproduces the first two moments of GBM, so the
European price converges to Black-Scholes at O(1/steps) (with the usual
saw-tooth because strikes do not land on nodes).

Why keep a tree at all when we have a closed form?
--------------------------------------------------
Because it handles **early exercise**. At every node the holder compares
"hold" (the discounted expected continuation value) with "exercise now"
(intrinsic value), and for an American option we take the maximum. That single
``np.maximum`` is the whole early-exercise feature, and it is exactly where
Black-Scholes has no answer.

Complexity is O(steps^2) time and O(steps) memory because we only ever keep
the current layer of the tree.
"""

from __future__ import annotations

import numpy as np

from optlab.pricing.black_scholes import _validate_kind


def _tree_parameters(r: float, q: float, sigma: float, dt: float) -> tuple[float, float, float]:
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            "CRR risk-neutral probability is outside (0, 1); the parameters admit "
            f"arbitrage (p={p:.4f}). Lower sigma, raise the step count, or check "
            "that r and q are per-year continuously compounded rates."
        )
    return u, d, p


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int = 200,
    q: float = 0.0,
    kind: str = "call",
    american: bool = False,
) -> float:
    """Price a European or American option on a CRR binomial tree.

    Parameters
    ----------
    S, K, T : float
        Spot, strike, time to expiry in years.
    r, sigma : float
        Risk-free rate and volatility.
    steps : int
        Number of time steps. Convergence to the Black-Scholes European price
        is roughly O(1/steps); 200 steps is accurate to a few basis points for
        a vanilla at-the-money option.
    q : float
        Continuous carry/dividend yield.
    kind : {"call", "put"}
    american : bool
        If True, apply the early-exercise condition at every node. Defaults to
        False so the tree can be checked against Black-Scholes.

    Returns
    -------
    float
        The option value at time 0.
    """
    kind = _validate_kind(kind)
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if T <= 0:
        return float(max(S - K, 0.0) if kind == "call" else max(K - S, 0.0))
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if sigma <= 0:
        raise ValueError(
            "sigma must be positive for the binomial tree; use bs_price for the "
            "deterministic sigma=0 limit"
        )

    dt = T / steps
    u, d, p = _tree_parameters(r, q, sigma, dt)
    disc = float(np.exp(-r * dt))
    one_minus_p = 1.0 - p

    # Terminal layer: node j (0..steps) has had j up-moves.
    j = np.arange(steps + 1)
    # exp(log S + j*log u + (steps-j)*log d) keeps this stable for large steps.
    log_step = sigma * np.sqrt(dt) * (2.0 * j - steps)
    terminal_spot = S * np.exp(log_step)

    if kind == "call":
        values = np.maximum(terminal_spot - K, 0.0)
        intrinsic = lambda spot: np.maximum(spot - K, 0.0)  # noqa: E731
    else:
        values = np.maximum(K - terminal_spot, 0.0)
        intrinsic = lambda spot: np.maximum(K - spot, 0.0)  # noqa: E731

    # Backward induction.
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + one_minus_p * values[:-1])
        if american:
            node_index = np.arange(i + 1)
            spots = S * np.exp(sigma * np.sqrt(dt) * (2.0 * node_index - i))
            values = np.maximum(values, intrinsic(spots))

    return float(values[0])


def early_exercise_premium(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int = 500,
    q: float = 0.0,
    kind: str = "put",
) -> float:
    """American minus European value on the same tree.

    Always ``>= 0``, and it should be ~0 for a European-style call on a
    non-dividend payer: with ``q = 0`` and ``r > 0`` you never want to exercise
    a call early, which is a nice sanity check of the tree and of your intution.
    """
    european = binomial_price(S, K, T, r, sigma, steps, q, kind, american=False)
    american = binomial_price(S, K, T, r, sigma, steps, q, kind, american=True)
    return american - european
