"""Tree and Monte Carlo engines must converge to the closed form."""

import numpy as np
import pytest

from optlab.pricing.binomial import binomial_price, early_exercise_premium
from optlab.pricing.black_scholes import bs_price
from optlab.pricing.monte_carlo import mc_price, simulate_gbm_paths

S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


def test_binomial_converges_to_black_scholes():
    """CRR is O(1/steps); 1600 steps should be within half a cent ATM."""
    analytic = bs_price(S, K, T, R, SIGMA)
    errors = {
        steps: abs(binomial_price(S, K, T, R, SIGMA, steps=steps) - analytic)
        for steps in (25, 100, 400, 1600)
    }
    assert errors[1600] < 5e-3
    assert errors[1600] < errors[100] < errors[25]


def test_binomial_puts_match_black_scholes():
    analytic = bs_price(S, K, T, R, SIGMA, kind="put")
    tree = binomial_price(S, K, T, R, SIGMA, steps=800, kind="put")
    assert tree == pytest.approx(analytic, abs=5e-3)


def test_american_put_worth_more_than_european():
    """Positive rates make early exercise of a put valuable."""
    premium = early_exercise_premium(S, 110.0, T, R, SIGMA, steps=400, kind="put")
    assert premium > 0.05


def test_american_call_on_non_dividend_payer_equals_european():
    """No carry => never optimal to exercise a call early."""
    european = binomial_price(S, K, T, R, SIGMA, steps=200, american=False)
    american = binomial_price(S, K, T, R, SIGMA, steps=200, american=True)
    assert american == pytest.approx(european, abs=1e-10)


def test_binomial_rejects_arbitrageable_parameters():
    with pytest.raises(ValueError, match="probability"):
        binomial_price(S, K, T, r=3.0, sigma=0.01, steps=10)


def test_monte_carlo_lands_in_its_confidence_interval():
    result = mc_price(S, K, T, R, SIGMA, n_paths=200_000, seed=11)
    analytic = bs_price(S, K, T, R, SIGMA)
    assert result.ci_low < analytic < result.ci_high
    assert abs(result.price - analytic) < 0.02


def test_mc_puts_also_converge():
    result = mc_price(S, K, T, R, SIGMA, n_paths=200_000, kind="put", seed=5)
    assert abs(result.price - bs_price(S, K, T, R, SIGMA, kind="put")) < 0.02


def test_variance_reduction_lowers_the_standard_error():
    plain = mc_price(S, K, T, R, SIGMA, n_paths=100_000, antithetic=False,
                     control_variate=False, seed=1)
    smart = mc_price(S, K, T, R, SIGMA, n_paths=100_000, antithetic=True,
                     control_variate=True, seed=1)
    assert smart.stderr < 0.9 * plain.stderr
    assert smart.beta > 0  # call payoff is positively correlated with the stock


def test_simulated_paths_have_the_right_shape_and_moments():
    paths = simulate_gbm_paths(100.0, 0.04, 0.25, 1.0, steps=12, n_paths=20_000, seed=2)
    assert paths.shape == (20_000, 13)
    assert np.allclose(paths[:, 0], 100.0)
    assert paths[:, -1].mean() == pytest.approx(100.0 * np.exp(0.04), rel=0.01)

    log_returns = np.diff(np.log(paths), axis=1)
    assert log_returns.std() == pytest.approx(0.25 * np.sqrt(1 / 12), rel=0.02)


def test_mc_validates_inputs():
    with pytest.raises(ValueError, match="sigma"):
        mc_price(S, K, T, R, 0.0)
    with pytest.raises(ValueError, match="T"):
        mc_price(S, K, 0.0, R, SIGMA)
