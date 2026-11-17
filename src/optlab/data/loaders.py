"""Loading option chains: a committed CSV sample, plus optional live fetchers.

The repository must run offline, so the default path is the sample snapshot in
``data/sample/``. The two live fetchers are convenience wrappers and are the
only code in the package that touches the network:

* :func:`fetch_yahoo_chain` needs the optional ``yfinance`` dependency
  (``pip install -e ".[data]"``);
* :func:`fetch_deribit_chain` uses Deribit's public JSON API over plain
  ``urllib``, no extra dependency, and crypto options are quoted in the base coin
  so prices are converted to USD before inverting to vol.

Both are best-effort: vendors change field names, so these are meant for
exploration, not for a production pipeline. Failures raise with a clear message
rather than silently returning junk.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from optlab.data.cleaning import normalize_kind
from optlab.implied_vol.solver import ImpliedVolError, implied_vol

_DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"


def load_csv_chain(path: str | Path, valuation_date=None) -> pd.DataFrame:
    """Read a chain CSV and normalise its columns.

    Expected columns: ``expiry``, ``strike``, ``kind``, ``bid``, ``ask``.
    Optional: ``volume``, ``open_interest``, ``T``. Column names are lowercased
    and stripped, and a ``spot`` column is hoisted into ``df.attrs["spot"]`` so
    the caller does not have to carry it separately.

    Parameters
    ----------
    path : str or Path
        CSV location, e.g. ``data/sample/option_chain_sample.csv``.
    valuation_date : optional
        Reference date used to derive ``T`` in :func:`optlab.data.cleaning.clean_chain`.
        Ignored if the file already has a ``T`` column, which is how the sample
        snapshot stays reproducible years later.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The committed sample lives at "
            "data/sample/option_chain_sample.csv"
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in ("expiry", "strike", "kind", "bid", "ask") if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df["kind"] = df["kind"].map(normalize_kind)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    for col in ("bid", "ask", "volume", "open_interest", "T"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")

    if "spot" in df.columns:
        df.attrs["spot"] = float(df["spot"].dropna().iloc[0])
        df.attrs["valuation_date"] = valuation_date

    return df.dropna(subset=["expiry", "strike", "bid", "ask"]).reset_index(drop=True)


def fetch_yahoo_chain(
    symbol: str,
    expiry: str | None = None,
    spot: float | None = None,
    r: float = 0.0,
    q: float = 0.0,
) -> pd.DataFrame:
    """Fetch one expiry's chain from Yahoo Finance via ``yfinance``.

    Returns a DataFrame with the same columns :func:`load_csv_chain` produces, so
    it can be piped straight into :func:`optlab.data.cleaning.clean_chain`.

    ``yfinance`` returns last traded prices rather than a live two-sided market,
    so ``bid``/``ask`` can be stale or zero; the cleaning filters exist precisely
    for that. Requires ``pip install -e ".[data]"``.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "fetch_yahoo_chain needs the optional 'yfinance' package: "
            'pip install -e ".[data]"'
        ) from exc

    ticker = yf.Ticker(symbol)
    expiries = list(ticker.options)
    if not expiries:
        raise ValueError(f"Yahoo returned no option expiries for {symbol!r}")
    chosen = expiry or expiries[0]
    if chosen not in expiries:
        raise ValueError(f"expiry {chosen!r} not available; choose from {expiries[:6]}")

    frame = ticker.option_chain(chosen)
    frames = []
    for kind, part in (("call", frame.calls), ("put", frame.puts)):
        part = part.copy()
        part["kind"] = kind
        frames.append(part)
    raw = pd.concat(frames, ignore_index=True)

    if spot is None:
        try:
            spot = float(ticker.fast_info["lastPrice"])
        except Exception:  # pragma: no cover - vendor field drift
            spot = float(raw["strike"].median())

    out = pd.DataFrame(
        {
            "expiry": pd.Timestamp(chosen),
            "strike": pd.to_numeric(raw["strike"], errors="coerce"),
            "kind": raw["kind"],
            "bid": pd.to_numeric(raw["bid"], errors="coerce"),
            "ask": pd.to_numeric(raw["ask"], errors="coerce"),
            "volume": pd.to_numeric(raw.get("volume"), errors="coerce"),
            "open_interest": pd.to_numeric(raw.get("openInterest"), errors="coerce"),
        }
    )
    out.attrs["spot"] = float(spot)
    out.attrs["r"] = r
    out.attrs["q"] = q
    return out.dropna(subset=["strike", "bid", "ask"]).reset_index(drop=True)


def fetch_deribit_chain(
    currency: str = "BTC",
    kind: str = "option",
    r: float = 0.0,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Fetch the live Deribit option chain (crypto) from the public API.

    Deribit quotes option premia in the base coin (BTC/ETH), so the USD price is
    ``mark_price * underlying_price``. We then invert the USD price through
    Black-Scholes to obtain the implied vol ourselves, which keeps the vol
    convention consistent with the rest of the package.

    Returns a DataFrame with ``expiry``, ``strike``, ``kind``, ``bid``, ``ask``,
    ``mark``, ``spot`` and ``iv``. Network failures raise ``RuntimeError`` with
    the underlying error attached.
    """
    url = f"{_DERIBIT_URL}?currency={currency.upper()}&kind={kind}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"could not reach the Deribit API at {_DERIBIT_URL}: {exc}. "
            "Use the committed sample CSV for offline work."
        ) from exc

    rows = payload.get("result", [])
    if not rows:
        raise RuntimeError(
            "Deribit returned no instruments; the public API may be rate-limiting "
            "or the currency code may be wrong"
        )

    records = []
    for item in rows:
        name = item.get("instrument_name", "")
        parts = name.split("-")
        if len(parts) != 4:
            continue
        _, expiry_str, strike_str, cp = parts
        if cp not in ("C", "P"):
            continue
        underlying = item.get("underlying_price") or item.get("index_price")
        mark_coin = item.get("mark_price")
        if not underlying or mark_coin is None:
            continue
        try:
            strike = float(strike_str)
            expiry = pd.Timestamp.strptime(expiry_str, "%d%b%y")
        except ValueError:
            continue
        mark_usd = float(mark_coin) * float(underlying)
        records.append(
            {
                "expiry": expiry,
                "strike": strike,
                "kind": "call" if cp == "C" else "put",
                "bid": np.nan,
                "ask": np.nan,
                "mark": mark_usd,
                "spot": float(underlying),
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    df["T"] = (df["expiry"] - pd.Timestamp.today().normalize()).dt.days / 365.0
    spot = float(df["spot"].iloc[0])
    ivs = []
    for row in df.itertuples(index=False):
        if row.T <= 0:
            ivs.append(np.nan)
            continue
        try:
            ivs.append(implied_vol(row.mark, spot, row.strike, row.T, r, 0.0, row.kind))
        except ImpliedVolError:
            ivs.append(np.nan)
    df["iv"] = ivs
    # "bid"/"ask" are unavailable for crypto options in this feed; Deribit does
    # publish them per-instrument, which would need one request per strike.
    df["bid"] = df["mark"]
    df["ask"] = df["mark"]
    return df.dropna(subset=["iv"]).reset_index(drop=True)
