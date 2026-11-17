"""optlab — a small, readable options pricing and volatility research lab.

The package is organised the way a desk would think about the problem:

    pricing      what is this option worth?           (BS, binomial tree, Monte Carlo)
    greeks       how does the value move?            (analytic + bump-and-reprice)
    implied_vol  what vol does the market imply?     (Newton with Brent fallback)
    surface      how does implied vol vary by K, T?  (SVI slices + no-arbitrage checks)
    hedging      what happens if I actually hedge?   (delta-hedge PnL + attribution)
    data         where do quotes come from?          (loaders + cleaning)

Every module is importable on its own; nothing here needs a network connection
except :mod:`optlab.data.loaders`.
"""

from optlab.pricing.black_scholes import (
    Greeks,
    bs_greeks,
    bs_price,
    bs_vega,
    d1_d2,
)
from optlab.pricing.binomial import binomial_price
from optlab.pricing.monte_carlo import MCResult, mc_price, simulate_gbm_paths
from optlab.greeks.finite_diff import finite_difference_greeks
from optlab.implied_vol.solver import ImpliedVolError, IVResult, implied_vol
from optlab.surface.svi import SVIParams, fit_svi, fit_svi_slice
from optlab.surface.interpolate import VolSurface

__version__ = "0.1.0"

__all__ = [
    "Greeks",
    "bs_price",
    "bs_greeks",
    "bs_vega",
    "d1_d2",
    "binomial_price",
    "MCResult",
    "mc_price",
    "simulate_gbm_paths",
    "finite_difference_greeks",
    "implied_vol",
    "IVResult",
    "ImpliedVolError",
    "SVIParams",
    "fit_svi",
    "fit_svi_slice",
    "VolSurface",
]
