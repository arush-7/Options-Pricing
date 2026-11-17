"""Cleaning an option chain into something you can actually fit.

Raw quotes are hostile. The problems, in the order they appear in
:func:`clean_chain`:

* zero or crossed bids (no tradeable market),
* absurdly wide spreads (the mid is not a price anyone believes),
* strikes far from the money (the IV there is pure noise: vega is tiny, so a
  one-cent error in price becomes a hundred vol points). The quantitative version
  of "far from the money" is a **delta band**: below ~3% delta the option's price
  is quote granularity rather than volatility information, and it will drag the
  surface fit into a nonsense shape,
* quotes that violate no-arbitrage bounds for any positive volatility,
* quotes near expiry, where one day of uncertainty dominates the price.

We also estimate the **forward** from put-call parity rather than trusting a
spot print, because the forward is the correct moneyness anchor:

    C - P = e^{-rT} (F - K)

so regressing ``C - P`` on ``K`` gives slope ``-e^{-rT}`` and intercept
``e^{-rT} F``. With a dozen strikes, that regression pins F down to a few basis
points even when individual mids are noisy — a standard desk trick.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from optlab.implied_vol.solver import ImpliedVolError, implied_vol
from optlab.pricing.black_scholes import bs_greeks

_DAYS_PER_YEAR = 365.0
_KIND_ALIASES = {
    "c": "call",
    "call": "call",
    "ce": "call",
    "p": "put",
    "put": "put",
    "pe": "put",
}


def normalize_kind(value: str) -> str:
    """Map vendor spellings (``"C"``, ``"call"``, ``"Put"``) to ``"call"``/``"put"``."""
    key = str(value).strip().lower()
    if key not in _KIND_ALIASES:
        raise ValueError(f"unrecognised option kind {value!r}")
    return _KIND_ALIASES[key]


def implied_forward(
    calls, puts, strikes, T: float, r: float
) -> tuple[float, float]:
    """Estimate ``(forward, discount_factor)`` from put-call parity.

    Parameters
    ----------
    calls, puts : array-like
        Mid prices for the same strikes/expiry.
    strikes : array-like
        Strikes, aligned with the price arrays.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate (only used to sanity-check the discount factor).

    Returns
    -------
    (forward, discount_factor)

    Raises
    ------
    ValueError
        If fewer than two usable strikes or the implied discount factor is not
        in a sane range, which means the quotes are not a call/put pair.
    """
    c = np.asarray(calls, dtype=float)
    p = np.asarray(puts, dtype=float)
    k = np.asarray(strikes, dtype=float)
    mask = np.isfinite(c) & np.isfinite(p) & (k > 0) & (c > 0) & (p > 0)
    if mask.sum() < 2:
        raise ValueError("need at least two call/put pairs to back out the forward")

    diff = c[mask] - p[mask]
    design = np.column_stack([np.ones(mask.sum()), k[mask]])
    (intercept, slope), *_ = np.linalg.lstsq(design, diff, rcond=None)

    discount = -float(slope)
    if not 0.5 < discount <= 1.5:
        raise ValueError(
            f"implied discount factor {discount:.4f} is implausible; the call and put "
            "quotes are probably not matched or are stale"
        )
    forward = float(intercept) / discount
    if not np.isfinite(forward) or forward <= 0:
        raise ValueError("implied forward is not positive; check the quote alignment")
    return forward, discount


@dataclass
class CleaningReport:
    """How many rows each filter removed — the audit trail for a cleaning run."""

    steps: dict[str, int] = field(default_factory=dict)
    n_input: int = 0
    n_output: int = 0

    def add(self, step: str, removed: int) -> None:
        if removed:
            self.steps[step] = self.steps.get(step, 0) + int(removed)

    def summary(self) -> str:
        lines = [f"cleaning report: {self.n_input} -> {self.n_output} quotes"]
        for step, removed in self.steps.items():
            lines.append(f"  dropped {removed:5d}  {step}")
        total = sum(self.steps.values())
        lines.append(f"  total removed {total} ({total / max(self.n_input, 1):.1%})")
        return "\n".join(lines)


def clean_chain(
    chain: pd.DataFrame,
    spot: float,
    r: float,
    q: float = 0.0,
    *,
    valuation_date=None,
    min_days: int = 3,
    max_days: int = 730,
    min_moneyness: float = 0.5,
    max_moneyness: float = 1.6,
    max_rel_spread: float = 0.5,
    min_iv: float = 0.005,
    max_iv: float = 3.0,
    min_abs_delta: float = 0.03,
    max_abs_delta: float = 0.97,
    use_parity_forward: bool = True,
    return_report: bool = False,
):
    """Filter a raw chain and attach the columns the surface fit needs.

    Required input columns: ``expiry``, ``strike``, ``kind``, ``bid``, ``ask``.
    Optional: ``volume``, ``open_interest``, ``T`` (if you already have expiries
    in years and want to skip the calendar maths — useful for a frozen sample).

    Adds: ``mid``, ``rel_spread``, ``T``, ``forward``, ``k`` (log-moneyness),
    ``iv`` (solved with :func:`optlab.implied_vol.solver.implied_vol`).

    Parameters
    ----------
    spot : float
        Spot price of the underlying.
    r, q : float
        Continuously compounded rate and carry yield.
    min_days, max_days : int
        Expiry window. The short end is where quotes are thickest in noise.
    min_moneyness, max_moneyness : float
        Coarse strike band; drop strikes outside ``[min, max] * spot``.
    min_abs_delta, max_abs_delta : float
        The real wing filter, applied after the implied vol is solved. Options
        whose |delta| falls outside this band have almost no vega, so their
        implied vol is dominated by quote granularity. Filtering by delta rather
        than by a fixed strike range adapts automatically to expiry and vol level:
        three sigma away is a very different strike for a week and for a year.
    max_rel_spread : float
        Drop quotes whose ``(ask - bid) / mid`` exceeds this.
    use_parity_forward : bool
        Estimate each expiry's forward from put-call parity when both kinds are
        quoted, instead of using ``spot * exp((r-q)T)``.
    return_report : bool
        Return ``(chain, report)`` instead of just the chain.

    Returns
    -------
    DataFrame (and CleaningReport if ``return_report``)
    """
    report = CleaningReport()
    df = chain.copy()
    report.n_input = len(df)

    required = ("expiry", "strike", "kind", "bid", "ask")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"chain is missing required columns: {missing}")

    # --- normalise types ---------------------------------------------------
    df["kind"] = df["kind"].map(normalize_kind)
    for col in ("strike", "bid", "ask"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("volume", "open_interest"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["expiry", "strike", "kind", "bid", "ask"])
    report.add("unparseable rows", before - len(df))

    # --- tradeable quotes --------------------------------------------------
    before = len(df)
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])]
    report.add("missing/crossed quotes", before - len(df))

    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    before = len(df)
    df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"]
    df = df[df["rel_spread"] <= max_rel_spread]
    report.add(f"relative spread > {max_rel_spread}", before - len(df))

    # --- time to expiry ----------------------------------------------------
    if "T" in df.columns and valuation_date is None:
        df["T"] = pd.to_numeric(df["T"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["T"])
        report.add("missing T", before - len(df))
    else:
        valuation_date = pd.Timestamp.today().normalize() if valuation_date is None else pd.Timestamp(valuation_date)
        days = (df["expiry"] - valuation_date).dt.days
        df["T"] = days / _DAYS_PER_YEAR
        before = len(df)
        df = df[df["T"] >= min_days / _DAYS_PER_YEAR]
        report.add(f"expiry inside {min_days} days", before - len(df))
        before = len(df)
        df = df[df["T"] <= max_days / _DAYS_PER_YEAR]
        report.add(f"expiry beyond {max_days} days", before - len(df))

    # --- moneyness band ----------------------------------------------------
    before = len(df)
    df["_moneyness"] = df["strike"] / float(spot)
    df = df[(df["_moneyness"] >= min_moneyness) & (df["_moneyness"] <= max_moneyness)]
    report.add("strikes outside moneyness band", before - len(df))
    df = df.drop(columns=["_moneyness"])

    before = len(df)
    df = df.drop_duplicates(subset=["expiry", "strike", "kind"], keep="last")
    report.add("duplicate quotes", before - len(df))

    if df.empty:
        report.n_output = 0
        out = df.reset_index(drop=True)
        return (out, report) if return_report else out

    # --- forward per expiry ------------------------------------------------
    forwards: dict[pd.Timestamp, float] = {}
    for expiry, grp in df.groupby("expiry", sort=True):
        T = float(grp["T"].iloc[0])
        forward = float(spot) * float(np.exp((r - q) * T))
        if use_parity_forward:
            calls = grp[grp["kind"] == "call"].set_index("strike")["mid"]
            puts = grp[grp["kind"] == "put"].set_index("strike")["mid"]
            common = calls.index.intersection(puts.index)
            if len(common) >= 3:
                try:
                    forward, _ = implied_forward(
                        calls.loc[common].to_numpy(),
                        puts.loc[common].to_numpy(),
                        common.to_numpy(),
                        T,
                        r,
                    )
                except ValueError:
                    pass  # fall back to the carry-based forward
        forwards[expiry] = forward

    df["forward"] = df["expiry"].map(forwards)
    df["k"] = np.log(df["strike"] / df["forward"])

    # --- implied vol, then the delta band ----------------------------------
    ivs: list[float] = []
    abs_deltas: list[float] = []
    for row in df.itertuples(index=False):
        try:
            iv = implied_vol(
                row.mid, float(spot), float(row.strike), float(row.T), r, q, row.kind
            )
        except (ImpliedVolError, ZeroDivisionError):
            ivs.append(np.nan)
            abs_deltas.append(np.nan)
            continue
        ivs.append(iv)
        greeks = bs_greeks(float(spot), float(row.strike), float(row.T), r, iv, q, row.kind)
        abs_deltas.append(abs(float(greeks.delta)))

    df["iv"] = ivs
    df["abs_delta"] = abs_deltas

    before = len(df)
    df = df[df["iv"].notna()]
    report.add("implied vol could not be solved (no-arbitrage bounds)", before - len(df))

    before = len(df)
    df = df[(df["iv"] > min_iv) & (df["iv"] < max_iv)]
    report.add(f"implied vol outside ({min_iv}, {max_iv})", before - len(df))

    before = len(df)
    df = df[(df["abs_delta"] >= min_abs_delta) & (df["abs_delta"] <= max_abs_delta)]
    report.add(
        f"|delta| outside [{min_abs_delta}, {max_abs_delta}]: vega too small to quote a vol",
        before - len(df),
    )

    out = df.sort_values(["T", "strike", "kind"]).reset_index(drop=True)
    report.n_output = len(out)
    out.attrs["cleaning_report"] = report
    return (out, report) if return_report else out
