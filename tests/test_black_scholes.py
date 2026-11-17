"""Black-Scholes values, parity and analytic Greeks."""

import numpy as np
import pytest

from optlab.pricing.black_scholes import (
    bs_greeks,
    bs_price,
    bs_vega,
    put_call_parity_gap,
)


def test_known_atm_call_and_put():
    """Textbook ATM values: S=K=100, T=1, r=5%, sigma=20%."""
    assert bs_price(100, 100, 1.0, 0.05, 0.2) == pytest.approx(10.450583572185565, rel=1e-11)
    assert bs_price(100, 100, 1.0, 0.05, 0.2, kind="put") == pytest.approx(
        5.573526022256971, rel=1e-11
    )


@pytest.mark.parametrize("S", [70.0, 100.0, 140.0])
@pytest.mark.parametrize("T", [0.05, 0.5, 2.0])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(S, T, q):
    """C - P = S e^{-qT} - K e^{-rT} must hold to machine precision."""
    K, r, sigma = 100.0, 0.04, 0.25
    call = bs_price(S, K, T, r, sigma, q, "call")
    put = bs_price(S, K, T, r, sigma, q, "put")
    assert put_call_parity_gap(call, put, S, K, T, r, q) == pytest.approx(0.0, abs=1e-12)


def test_delta_parity_and_greek_symmetries():
    S, K, T, r, q, sigma = 105.0, 100.0, 0.75, 0.03, 0.01, 0.22
    call = bs_greeks(S, K, T, r, sigma, q, "call")
    put = bs_greeks(S, K, T, r, sigma, q, "put")

    assert call.delta - put.delta == pytest.approx(np.exp(-q * T), rel=1e-12)
    assert call.gamma == pytest.approx(put.gamma, rel=1e-12)
    assert call.vega == pytest.approx(put.vega, rel=1e-12)
    assert call.vega > 0 and call.gamma > 0
    assert call.rho > 0 > put.rho
    # A long at-the-money-ish call decays as calendar time passes.
    assert bs_greeks(100, 100, 1.0, 0.03, 0.2, 0.0, "call").theta < 0
    # Vega from the Greek and the standalone helper must agree.
    assert call.vega == pytest.approx(bs_vega(S, K, T, r, sigma, q), rel=1e-12)


def test_expired_option_returns_payoff():
    assert bs_price(120, 100, 0.0, 0.05, 0.2, kind="call") == pytest.approx(20.0)
    assert bs_price(80, 100, 0.0, 0.05, 0.2, kind="call") == pytest.approx(0.0)
    assert bs_price(80, 100, 0.0, 0.05, 0.2, kind="put") == pytest.approx(20.0)

    expiring = bs_greeks(120, 100, 0.0, 0.05, 0.2, 0.0, "call")
    assert expiring.delta == pytest.approx(1.0)
    assert expiring.gamma == 0.0
    assert expiring.vega == 0.0


def test_zero_vol_is_the_discounted_forward_intrinsic():
    """With sigma=0 the option is worth max(S e^{-qT} - K e^{-rT}, 0)."""
    S, K, T, r, q = 100.0, 95.0, 1.0, 0.05, 0.02
    expected = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert bs_price(S, K, T, r, 0.0, q, "call") == pytest.approx(expected, rel=1e-12)
    assert bs_price(S, K, T, r, 0.0, q, "put") == pytest.approx(0.0, abs=1e-15)


def test_vectorised_inputs_match_scalar_calls():
    spots = np.array([80.0, 100.0, 120.0])
    prices = bs_price(spots, 100.0, 0.5, 0.04, 0.2)
    assert prices.shape == spots.shape
    for i, s in enumerate(spots):
        assert prices[i] == pytest.approx(bs_price(float(s), 100.0, 0.5, 0.04, 0.2), rel=1e-14)

    greeks = bs_greeks(spots, 100.0, 0.5, 0.04, 0.2)
    assert greeks.delta.shape == spots.shape
    assert np.all(np.diff(greeks.delta) > 0)  # delta increases with spot


def test_scalar_inputs_return_floats():
    assert isinstance(bs_price(100, 100, 1.0, 0.05, 0.2), float)
    assert isinstance(bs_price(100, 100, 1.0, 0.05, 0.2, kind="put"), float)


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        bs_price(100, 100, 1.0, 0.05, 0.2, kind="straddle")
