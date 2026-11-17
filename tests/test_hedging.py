"""The hedging engine: does the hedge actually neutralise delta?

These are the tests that catch sign errors, which are the classic failure mode in
hedging code. With realised vol equal to the vol you sold, a self-financing
delta hedge must have zero expected PnL.
"""

import numpy as np
import pytest

from optlab.hedging.attribution import decompose_pnl
from optlab.hedging.simulate import simulate_delta_hedge
from optlab.pricing.monte_carlo import simulate_gbm_paths

S0, K, T, R = 100.0, 100.0, 1.0, 0.03


def test_hedge_has_zero_expected_pnl_at_the_pricing_vol():
    result = simulate_delta_hedge(S0, K, T, R, 0.2, 0.2, n_steps=52, n_paths=4000, seed=3)
    assert abs(result.mean) < 4.0 * result.stderr + 0.02


def test_short_vol_when_realised_is_lower():
    """Sell 20% vol, realise 14%: the hedge grinds out a loss."""
    result = simulate_delta_hedge(S0, K, T, R, 0.2, 0.14, n_steps=52, n_paths=4000, seed=4)
    assert result.mean < -0.05


def test_long_gamma_when_realised_is_higher():
    result = simulate_delta_hedge(S0, K, T, R, 0.2, 0.30, n_steps=52, n_paths=4000, seed=5)
    assert result.mean > 0.05


def test_transaction_costs_reduce_pnl_by_about_their_size():
    free = simulate_delta_hedge(S0, K, T, R, 0.2, 0.2, n_steps=52, n_paths=2000, seed=6,
                                transaction_cost_bps=0.0)
    costly = simulate_delta_hedge(S0, K, T, R, 0.2, 0.2, n_steps=52, n_paths=2000, seed=6,
                                  transaction_cost_bps=20.0)
    drag = free.mean - costly.mean
    assert drag > 0
    assert drag == pytest.approx(costly.mean_cost, rel=0.5)


def test_more_rebalancing_shrinks_the_hedging_error():
    """The distribution tightens as the rebalancing grid gets finer."""
    coarse = simulate_delta_hedge(S0, K, T, R, 0.25, 0.25, n_steps=4, n_paths=2000, seed=7)
    fine = simulate_delta_hedge(S0, K, T, R, 0.25, 0.25, n_steps=252, n_paths=2000, seed=7)
    assert fine.std < coarse.std


def test_put_sides_work_too():
    result = simulate_delta_hedge(S0, 90.0, T, R, 0.25, 0.25, n_steps=52, n_paths=2000,
                                  seed=8, kind="put")
    assert abs(result.mean) < 4.0 * result.stderr + 0.02


def test_pnl_attribution_adds_up_and_explains_most_of_the_pnl():
    paths = simulate_gbm_paths(S0, R, 0.30, T, steps=52, n_paths=500, seed=9)
    attribution = decompose_pnl(paths, K=K, T=T, r=R, sigma=0.2)

    components = attribution.components
    total = (
        components["theta"] + components["gamma"] + components["vega"]
        + components["finance"] + components["residual"]
    )
    assert np.allclose(total, components["hedged_total"], atol=1e-10)

    mean = components.mean(numeric_only=True)
    # Realised 30% vs implied 20%: gamma rent beats theta decay on average.
    assert mean["gamma"] > 0
    assert mean["theta"] < 0
    assert mean["hedged_total"] > 0
    # Residual should be a small fraction of the story at this rebalancing frequency.
    assert abs(mean["residual"]) < 0.05 * (abs(mean["theta"]) + abs(mean["gamma"]))


def test_attribution_matches_the_hedge_engine_when_vol_matches():
    """The financing leg is what makes theta + gamma average out to zero.

    If it is dropped, the decomposed PnL drifts by roughly ``r (V - Delta*S)`` per
    year, which is exactly the sign error this test is here to catch.
    """
    paths = simulate_gbm_paths(S0, R, 0.20, T, steps=252, n_paths=2000, seed=12)
    attribution = decompose_pnl(paths, K=K, T=T, r=R, sigma=0.20)
    hedged = attribution.components["hedged_total"]
    assert abs(hedged.mean()) < 4.0 * hedged.std(ddof=1) / np.sqrt(len(hedged)) + 0.05


def test_attribution_handles_a_single_path():
    paths = simulate_gbm_paths(S0, R, 0.2, T, steps=12, n_paths=1, seed=10)[0]
    attribution = decompose_pnl(paths, K=K, T=T, r=R, sigma=0.2)
    assert len(attribution.components) == 1
    assert attribution.n_steps == 12


def test_vega_component_appears_when_a_vol_path_is_supplied():
    paths = simulate_gbm_paths(S0, R, 0.2, T, steps=12, n_paths=3, seed=11)
    sigma_path = np.linspace(0.2, 0.25, paths.shape[1])  # one vol per grid date
    attribution = decompose_pnl(paths, K=K, T=T, r=R, sigma=0.2, sigma_path=sigma_path)
    assert np.all(attribution.components["vega"] > 0)  # long option, vol up
