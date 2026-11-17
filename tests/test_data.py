"""Chain loading, parity-implied forwards and the cleaning filters."""

import numpy as np
import pandas as pd
import pytest

from optlab.data.cleaning import clean_chain, implied_forward, normalize_kind
from optlab.data.loaders import load_csv_chain
from optlab.pricing.black_scholes import bs_price

SAMPLE = "data/sample/option_chain_sample.csv"


def test_kind_normalisation():
    assert normalize_kind("C") == "call"
    assert normalize_kind(" Put ") == "put"
    with pytest.raises(ValueError, match="kind"):
        normalize_kind("straddle")


def test_implied_forward_recovers_a_known_forward():
    """Put-call parity should recover F to well under a cent from noisy mids."""
    rng = np.random.default_rng(0)
    strike = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0])
    T, r, sigma, forward = 0.5, 0.03, 0.22, 101.25
    calls = np.array([bs_price(forward * np.exp(-r * T), k, T, r, sigma) for k in strike])
    puts = np.array([bs_price(forward * np.exp(-r * T), k, T, r, sigma, kind="put") for k in strike])
    calls *= 1.0 + rng.normal(0, 0.002, calls.shape)  # quote noise

    recovered_forward, discount = implied_forward(calls, puts, strike, T, r)
    assert recovered_forward == pytest.approx(forward, rel=2e-3)
    assert discount == pytest.approx(np.exp(-r * T), rel=5e-3)


def test_implied_forward_rejects_too_few_pairs():
    with pytest.raises(ValueError, match="two"):
        implied_forward([5.0], [4.0], [100.0], 0.5, 0.03)


def test_sample_chain_loads_with_spot():
    frame = load_csv_chain(SAMPLE)
    assert frame.attrs["spot"] == pytest.approx(100.0)
    assert {"expiry", "strike", "kind", "bid", "ask"} <= set(frame.columns)
    assert frame["kind"].isin(["call", "put"]).all()


def test_cleaning_drops_the_junk_and_attaches_the_fit_columns():
    raw = load_csv_chain(SAMPLE)
    cleaned, report = clean_chain(
        raw, spot=raw.attrs["spot"], r=0.04, q=0.015, return_report=True
    )

    assert 0 < len(cleaned) < len(raw)  # junk quotes really were removed
    assert report.n_output == len(cleaned)
    assert {"mid", "rel_spread", "T", "forward", "k", "iv", "abs_delta"} <= set(cleaned.columns)
    assert (cleaned["ask"] >= cleaned["bid"]).all()
    assert (cleaned["iv"] > 0).all()
    assert np.allclose(np.log(cleaned["strike"] / cleaned["forward"]), cleaned["k"])
    # The deliberately bad rows must not survive.
    assert (cleaned["bid"] > 0).all()
    assert (cleaned["rel_spread"] <= 0.5).all()
    assert not cleaned.duplicated(["expiry", "strike", "kind"]).any()


def test_cleaning_keeps_only_the_delta_band():
    raw = load_csv_chain(SAMPLE)
    cleaned = clean_chain(raw, spot=raw.attrs["spot"], r=0.04, q=0.015)
    assert cleaned["abs_delta"].between(0.03, 0.97).all()


def test_cleaning_rejects_a_chain_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        clean_chain(pd.DataFrame({"strike": [100.0]}), spot=100.0, r=0.03)
