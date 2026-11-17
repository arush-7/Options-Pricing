"""SVI fitting and the static arbitrage diagnostics."""

import numpy as np
import pytest

from optlab.surface.arbitrage import (
    check_surface,
    is_arbitrage_free,
    repair_butterfly,
    scan_butterfly,
    scan_calendar,
)
from optlab.surface.svi import SVIParams, fit_svi, fit_svi_slice, gatheral_g, iv_rmse

TRUTH = SVIParams(a=0.0020, b=0.0100, rho=-0.55, m=-0.02, sigma=0.06)
T_SHORT = 0.25


def test_known_slice_is_butterfly_free():
    assert scan_butterfly(TRUTH, -0.5, 0.5).is_arbitrage_free


def test_fit_recovers_a_synthetic_slice():
    """Exact synthetic data: the fit should essentially recover the truth."""
    k = np.linspace(-0.35, 0.25, 25)
    w = TRUTH.total_variance(k)
    fitted = fit_svi(k, w, n_starts=6, seed=1)

    assert iv_rmse(fitted, k, TRUTH.iv(k, T_SHORT), T_SHORT) < 1e-4
    assert scan_butterfly(fitted, -0.5, 0.5).is_arbitrage_free
    # Wing slope and skew must have the right sign even if the parametrisation
    # is not unique.
    assert fitted.b > 0
    assert -1 < fitted.rho < 0


def test_fit_svi_slice_accepts_strikes_and_vols():
    strikes = np.linspace(70, 130, 21)
    forward = 100.5
    k = np.log(strikes / forward)
    ivs = TRUTH.iv(k, T_SHORT)
    fitted = fit_svi_slice(strikes, ivs, T_SHORT, forward=forward, n_starts=6, seed=2)
    assert iv_rmse(fitted, k, ivs, T_SHORT) < 1e-4


def test_fit_requires_enough_points():
    with pytest.raises(ValueError, match="at least 5"):
        fit_svi(np.array([0.0, 0.1, 0.2, 0.3]), np.array([0.01, 0.011, 0.012, 0.013]))


def test_negative_density_is_flagged():
    """Steep positive skew with tiny variance makes g(k) negative."""
    bad = SVIParams(a=0.0005, b=0.2, rho=0.99, m=0.0, sigma=0.002)
    report = scan_butterfly(bad, -0.5, 0.5)
    assert not report.is_arbitrage_free
    assert report.worst_value < 0
    assert np.min(gatheral_g(bad, np.linspace(-0.5, 0.5, 51))) < 0


def test_negative_total_variance_is_flagged():
    bad = SVIParams(a=-0.05, b=0.01, rho=-0.5, m=0.0, sigma=0.05)
    assert not scan_butterfly(bad, -0.5, 0.5).is_arbitrage_free


def test_calendar_order_matters():
    """Total variance must rise with maturity; the reverse is arbitrage."""
    late = SVIParams(a=0.0040, b=0.0150, rho=-0.50, m=-0.02, sigma=0.08)
    assert scan_calendar(TRUTH, late, -0.4, 0.4).is_arbitrage_free
    assert not scan_calendar(late, TRUTH, -0.4, 0.4).is_arbitrage_free


def test_check_surface_covers_every_slice_and_neighbour_pair():
    slices = [
        SVIParams(0.0010, 0.0080, -0.6, -0.01, 0.05),
        SVIParams(0.0025, 0.0110, -0.6, -0.01, 0.06),
        SVIParams(0.0045, 0.0140, -0.5, -0.02, 0.08),
    ]
    reports = check_surface(slices, [0.25, 0.5, 1.0], -0.4, 0.4)
    assert len(reports) == 5  # 3 butterflies + 2 calendars
    assert all(r.kind == "butterfly" for r in reports[:3])
    assert all(r.kind == "calendar" for r in reports[3:])
    assert is_arbitrage_free(reports)


def test_repair_fixes_a_violating_slice_and_leaves_a_good_one_alone():
    bad = SVIParams(a=0.0005, b=0.2, rho=0.99, m=0.0, sigma=0.002)
    repaired = repair_butterfly(bad)
    assert repaired.b < bad.b
    assert scan_butterfly(repaired, -0.5, 0.5).worst_value > scan_butterfly(bad, -0.5, 0.5).worst_value
    assert repair_butterfly(TRUTH) == TRUTH
