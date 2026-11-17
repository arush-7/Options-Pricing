"""Finite-difference Greeks must agree with the analytic ones."""

import pytest

from optlab.greeks.finite_diff import (
    bump_and_reprice,
    finite_difference_greeks,
    greeks_error_table,
)
from optlab.pricing.binomial import binomial_price
from optlab.pricing.black_scholes import bs_greeks, bs_price

_CONTRACTS = [
    # (S, K, T, r, q, sigma, kind)
    (100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "call"),
    (100.0, 80.0, 0.5, 0.03, 0.0, 0.25, "call"),
    (100.0, 120.0, 0.25, 0.02, 0.01, 0.30, "put"),
    (50.0, 50.0, 2.0, 0.04, 0.02, 0.15, "put"),
]


@pytest.mark.parametrize("S,K,T,r,q,sigma,kind", _CONTRACTS)
def test_finite_difference_matches_analytic(S, K, T, r, q, sigma, kind):
    analytic = bs_greeks(S, K, T, r, sigma, q, kind)
    numeric = finite_difference_greeks(
        bs_price, S=S, K=K, T=T, r=r, sigma=sigma, q=q, kind=kind
    )
    for name, error in greeks_error_table(analytic, numeric).items():
        tolerance = 1e-3 * max(1.0, abs(getattr(analytic, name)))
        assert error < tolerance, f"{name} off by {error:.3e}"


def test_finite_difference_works_on_a_tree_pricer():
    """The whole point of bump-and-reprice: no closed form required.

    The tree has its own discretisation noise, so the bump has to be larger than
    the price error; this is why rel_bump is a knob.
    """
    analytic = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call")
    numeric = finite_difference_greeks(
        binomial_price,
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        rel_bump=1e-2,
        pricer_kwargs={"steps": 2000},
    )
    assert numeric.delta == pytest.approx(analytic.delta, abs=5e-3)
    # Gamma is a second difference, so the tree's own O(1/steps) price error is
    # divided by the bump squared: expect a few 1e-3 of noise, not 1e-6.
    assert numeric.gamma == pytest.approx(analytic.gamma, abs=5e-3)
    assert numeric.vega == pytest.approx(analytic.vega, abs=0.1)


def test_american_tree_greeks_are_reported():
    numeric = finite_difference_greeks(
        binomial_price,
        S=100.0,
        K=110.0,
        T=1.0,
        r=0.06,
        sigma=0.25,
        kind="put",
        rel_bump=1e-3,
        pricer_kwargs={"steps": 400, "american": True},
    )
    assert -1.0 < numeric.delta < 0.0  # put delta is negative
    assert numeric.gamma > 0


def test_bump_and_reprice_reports_unknown_parameter():
    with pytest.raises(KeyError, match="vol"):
        bump_and_reprice(bs_price, parameter="vol", bump=0.01, S=100, K=100, T=1, r=0.05, sigma=0.2)


def test_bump_and_reprice_shifts_the_right_parameter():
    base = dict(S=100, K=100, T=1, r=0.05, sigma=0.2)
    bumped = bump_and_reprice(bs_price, parameter="sigma", bump=0.01, **base)
    assert bumped == pytest.approx(bs_price(**{**base, "sigma": 0.21}), rel=1e-14)
