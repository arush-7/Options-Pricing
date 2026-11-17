"""Price -> implied vol -> price round trips, and the refusal cases."""

import numpy as np
import pytest

from optlab.implied_vol.solver import (
    ImpliedVolError,
    implied_vol,
    no_arbitrage_bounds,
)
from optlab.pricing.black_scholes import bs_price

S, T, R, Q = 100.0, 0.5, 0.03, 0.0


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.50, 1.00])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_round_trip_recovers_the_volatility(K, sigma, kind):
    price = bs_price(S, K, T, R, sigma, Q, kind)
    recovered = implied_vol(price, S, K, T, R, Q, kind)
    assert recovered == pytest.approx(sigma, rel=1e-7)


def test_deep_out_of_the_money_still_inverts():
    """Vega is tiny here, which is exactly where naive Newton gives up."""
    price = bs_price(S, 140.0, 0.25, R, 0.30)
    assert price < 0.5
    assert implied_vol(price, S, 140.0, 0.25, R, 0.0, "call") == pytest.approx(0.30, rel=1e-5)


def test_newton_is_used_near_the_money():
    price = bs_price(S, 100.0, T, R, 0.2)
    details = implied_vol(price, S, 100.0, T, R, 0.0, "call", return_details=True)
    assert details.method == "newton"
    assert details.converged
    assert 0 < details.iterations < 20


def test_brent_fallback_produces_the_same_answer():
    price = bs_price(S, 100.0, T, R, 0.2)
    newton = implied_vol(price, S, 100.0, T, R, 0.0, "call")
    brent = implied_vol(price, S, 100.0, T, R, 0.0, "call", method="brent")
    assert brent == pytest.approx(newton, rel=1e-9)


def test_price_below_intrinsic_is_rejected():
    """A 50-strike call one year out is worth at least S - K e^{-rT}."""
    lower, _ = no_arbitrage_bounds(S, 50.0, 1.0, 0.05, 0.0, "call")
    with pytest.raises(ImpliedVolError, match="intrinsic"):
        implied_vol(lower - 1.0, S, 50.0, 1.0, 0.05, 0.0, "call")


def test_price_above_the_no_arbitrage_cap_is_rejected():
    _, upper = no_arbitrage_bounds(S, 100.0, T, R, 0.0, "call")
    with pytest.raises(ImpliedVolError, match="cap"):
        implied_vol(upper * 1.05, S, 100.0, T, R, 0.0, "call")


def test_price_on_the_intrinsic_bound_has_no_vol():
    lower, _ = no_arbitrage_bounds(S, 60.0, 1.0, 0.05, 0.0, "call")
    with pytest.raises(ImpliedVolError):
        implied_vol(lower, S, 60.0, 1.0, 0.05, 0.0, "call")


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_non_positive_prices_are_rejected(price):
    with pytest.raises(ImpliedVolError, match="positive"):
        implied_vol(price, S, 100.0, T, R)


def test_expired_option_is_rejected():
    with pytest.raises(ImpliedVolError, match="expired"):
        implied_vol(5.0, S, 100.0, 0.0, R)


def test_vector_inputs_are_rejected_clearly():
    """The solver is scalar on purpose; silent array coercion would confuse."""
    with pytest.raises((ImpliedVolError, TypeError, ValueError)):
        implied_vol(np.array([5.0, 6.0]), S, 100.0, T, R)
