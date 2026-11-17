"""Interpolating an implied volatility surface across strikes and expiries.

The right space to interpolate in
---------------------------------
Not in vol, and not in (K, T). We interpolate **total variance**
``w(k, T) = sigma^2 T`` at fixed log-moneyness ``k = ln(K/F(T))``:

* across strikes: each expiry is an SVI slice, so the smile is smooth and its
  derivatives are analytic;
* across expiries: linear in ``T`` at fixed ``k``, which is the standard "flat
  forward variance" assumption. Because it interpolates total variance (not
  vol), it automatically preserves calendar-arbitrage-freeness whenever the
  fitted slices are themselves ordered.

Extrapolation is deliberately boring: flat vol outside the fitted range. A fancy
extrapolation from five quoted strikes is a good way to lose money.

Usage
-----
>>> surface = VolSurface.from_quotes(strikes, ivs, expiries, forwards)   # doctest: +SKIP
>>> surface.iv(K=105.0, T=0.3)                                          # doctest: +SKIP
0.2231...
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optlab.surface.svi import SVIParams, fit_svi_slice, iv_rmse


@dataclass
class VolSurface:
    """A set of SVI slices plus the machinery to read off any (K, T).

    Attributes
    ----------
    maturities : ndarray
        Expiries in years, sorted ascending.
    forwards : ndarray
        Forward price per expiry (the moneyness anchor).
    params : list[SVIParams]
        One SVI slice per expiry, same order as ``maturities``.
    fit_quality : dict
        Diagnostics per expiry: number of quotes and vol RMSE in vol points.
    """

    maturities: np.ndarray
    forwards: np.ndarray
    params: list[SVIParams]
    fit_quality: dict[float, dict[str, float]] = field(default_factory=dict)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_quotes(
        cls,
        strikes,
        ivs,
        expiries,
        forwards=None,
        *,
        spot: float | None = None,
        r: float = 0.0,
        q: float = 0.0,
        min_quotes: int = 5,
        **svi_kwargs,
    ) -> "VolSurface":
        """Build a surface from flat arrays of quotes.

        Parameters
        ----------
        strikes, ivs, expiries : array-like
            One entry per quote. ``expiries`` in years.
        forwards : array-like, optional
            Forward price per quote. If omitted, ``spot * exp((r-q)T)`` is used.
        min_quotes : int
            Expiries with fewer quotes than this are skipped (SVI needs 5 points).
        **svi_kwargs
            Passed to :func:`optlab.surface.svi.fit_svi`.

        Returns
        -------
        VolSurface
        """
        strikes = np.asarray(strikes, dtype=float)
        ivs = np.asarray(ivs, dtype=float)
        expiries = np.asarray(expiries, dtype=float)
        if not (strikes.shape == ivs.shape == expiries.shape):
            raise ValueError("strikes, ivs and expiries must have the same shape")

        if forwards is None:
            if spot is None:
                raise ValueError("provide either forward prices or a spot price")
            forwards_arr = np.full_like(expiries, float(spot)) * np.exp((r - q) * expiries)
        else:
            forwards_arr = np.asarray(forwards, dtype=float)
            if forwards_arr.shape != expiries.shape:
                raise ValueError("forwards must have the same shape as expiries")

        unique_T = np.unique(expiries)
        params: list[SVIParams] = []
        fwds: list[float] = []
        used_T: list[float] = []
        quality: dict[float, dict[str, float]] = {}

        for T in unique_T:
            mask = (expiries == T) & (strikes > 0) & (ivs > 0) & np.isfinite(ivs)
            if mask.sum() < min_quotes:
                continue
            fwd = float(np.median(forwards_arr[mask]))
            try:
                p = fit_svi_slice(strikes[mask], ivs[mask], float(T), forward=fwd, **svi_kwargs)
            except (ValueError, RuntimeError):
                continue
            params.append(p)
            fwds.append(fwd)
            used_T.append(float(T))
            k = np.log(strikes[mask] / fwd)
            quality[float(T)] = {
                "n_quotes": float(mask.sum()),
                "iv_rmse": iv_rmse(p, k, ivs[mask], float(T)),
            }

        if not params:
            raise ValueError(
                f"no expiry had at least {min_quotes} usable quotes; cannot build a surface"
            )

        return cls(
            maturities=np.asarray(used_T, dtype=float),
            forwards=np.asarray(fwds, dtype=float),
            params=params,
            fit_quality=quality,
        )

    # -- reading the surface ------------------------------------------------
    def forward(self, T: float) -> float:
        """Forward price at ``T``, log-linearly interpolated in time."""
        T = float(T)
        log_f = np.log(self.forwards)
        if T <= self.maturities[0]:
            return float(self.forwards[0])
        if T >= self.maturities[-1]:
            return float(self.forwards[-1])
        return float(np.exp(np.interp(T, self.maturities, log_f)))

    def total_variance(self, k, T: float) -> np.ndarray:
        """Interpolated ``w(k, T)`` at fixed log-moneyness, linear in ``T``."""
        k_arr = np.asarray(k, dtype=float)
        T = float(T)
        if T <= self.maturities[0]:
            return np.asarray(self.params[0].total_variance(k_arr), dtype=float)
        if T >= self.maturities[-1]:
            return np.asarray(self.params[-1].total_variance(k_arr), dtype=float)

        upper = int(np.searchsorted(self.maturities, T))
        if np.isclose(T, self.maturities[upper]):
            return np.asarray(self.params[upper].total_variance(k_arr), dtype=float)
        lower = upper - 1
        T0, T1 = self.maturities[lower], self.maturities[upper]
        weight = (T - T0) / (T1 - T0)
        w0 = np.asarray(self.params[lower].total_variance(k_arr), dtype=float)
        w1 = np.asarray(self.params[upper].total_variance(k_arr), dtype=float)
        return (1.0 - weight) * w0 + weight * w1

    def iv(self, K, T: float) -> np.ndarray:
        """Implied vol at strike ``K`` and expiry ``T`` (years).

        Inside the fitted maturity range the total variance is interpolated
        linearly in ``T`` at fixed log-moneyness. Outside it we hold the nearest
        expiry's *vol*, which is the conservative choice: interpolating vol
        linearly would produce a term structure that is not arbitrage-free.
        """
        T = float(T)
        if T <= 0:
            raise ValueError("T must be positive")
        K_arr = np.asarray(K, dtype=float)
        k = np.log(K_arr / self.forward(T))
        if T <= self.maturities[0]:
            iv = np.asarray(self.params[0].iv(k, float(self.maturities[0])), dtype=float)
        elif T >= self.maturities[-1]:
            iv = np.asarray(self.params[-1].iv(k, float(self.maturities[-1])), dtype=float)
        else:
            iv = np.sqrt(np.maximum(self.total_variance(k, T), 1e-16) / T)
        return float(iv) if np.ndim(iv) == 0 else iv

    def iv_grid(self, strikes, maturities) -> np.ndarray:
        """Vol on a (maturity x strike) grid — the array a 3-D plot wants."""
        return np.array([[self.iv(K, T) for K in np.asarray(strikes, dtype=float)]
                         for T in np.asarray(maturities, dtype=float)])

    def delta_strike_skew(self, K, T: float, h: float = 1e-4) -> np.ndarray:
        """``d sigma / dK`` by central difference — the slope a trader quotes."""
        K_arr = np.asarray(K, dtype=float)
        up = self.iv(K_arr * (1.0 + h), T)
        down = self.iv(K_arr * (1.0 - h), T)
        return (up - down) / (2.0 * h * K_arr)

    def diagnostics(self) -> "pd.DataFrame":  # noqa: F821 - pandas imported lazily
        """Fit quality per expiry as a DataFrame."""
        import pandas as pd

        rows = [
            {
                "T": float(T),
                "forward": self.forward(float(T)),
                "n_quotes": int(self.fit_quality.get(float(T), {}).get("n_quotes", 0)),
                "iv_rmse": self.fit_quality.get(float(T), {}).get("iv_rmse", float("nan")),
                "a": self.params[i].a,
                "b": self.params[i].b,
                "rho": self.params[i].rho,
                "m": self.params[i].m,
                "sigma_svi": self.params[i].sigma,
            }
            for i, T in enumerate(self.maturities)
        ]
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.params)

    def __repr__(self) -> str:
        Ts = ", ".join(f"{T:.3f}" for T in self.maturities[:5])
        more = "" if len(self.maturities) <= 5 else ", ..."
        return f"VolSurface({len(self.params)} expiries: T=[{Ts}{more}])"
