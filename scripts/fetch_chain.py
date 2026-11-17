#!/usr/bin/env python3
"""Fetch or load an option chain, clean it, and save the result.

Examples
--------
    python scripts/fetch_chain.py --source sample
    python scripts/fetch_chain.py --source csv --path data/raw/my_chain.csv --spot 4520
    python scripts/fetch_chain.py --source yahoo --symbol AAPL --rate 0.045
    python scripts/fetch_chain.py --source deribit --rate 0.0

The cleaned file has the columns the surface fit needs (``T``, ``strike``,
``forward``, ``k``, ``iv``) plus the surviving quotes. The cleaning report is
printed so you can see exactly what was thrown away and why.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from optlab.data.cleaning import clean_chain
from optlab.data.loaders import fetch_deribit_chain, fetch_yahoo_chain, load_csv_chain

SAMPLE_PATH = Path("data/sample/option_chain_sample.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["sample", "csv", "yahoo", "deribit"], default="sample")
    parser.add_argument("--path", type=Path, help="CSV path for --source csv")
    parser.add_argument("--symbol", default="SPY", help="ticker for --source yahoo")
    parser.add_argument("--expiry", default=None, help="expiry for --source yahoo (YYYY-MM-DD)")
    parser.add_argument("--spot", type=float, default=None, help="spot price if the file does not carry one")
    parser.add_argument("--rate", type=float, default=0.04, help="continuously compounded risk-free rate")
    parser.add_argument("--carry", type=float, default=0.0, help="dividend/carry yield")
    parser.add_argument("--out", type=Path, default=Path("data/raw/chain_clean.csv"))
    parser.add_argument("--max-rel-spread", type=float, default=0.5)
    parser.add_argument("--min-iv", type=float, default=0.005)
    parser.add_argument("--max-iv", type=float, default=3.0)
    return parser


def load(args: argparse.Namespace) -> tuple[pd.DataFrame, float | None]:
    if args.source == "sample":
        frame = load_csv_chain(SAMPLE_PATH)
        return frame, frame.attrs.get("spot")
    if args.source == "csv":
        if args.path is None:
            raise SystemExit("--source csv requires --path")
        frame = load_csv_chain(args.path)
        return frame, frame.attrs.get("spot")
    if args.source == "yahoo":
        frame = fetch_yahoo_chain(args.symbol, args.expiry, spot=args.spot, r=args.rate, q=args.carry)
        return frame, frame.attrs.get("spot")
    frame = fetch_deribit_chain("BTC", r=args.rate)
    return frame, frame.attrs.get("spot", args.spot)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chain, spot = load(args)
    spot = args.spot if args.spot is not None else spot
    if spot is None:
        raise SystemExit("could not determine spot; pass --spot")

    cleaned, report = clean_chain(
        chain,
        spot=float(spot),
        r=args.rate,
        q=args.carry,
        max_rel_spread=args.max_rel_spread,
        min_iv=args.min_iv,
        max_iv=args.max_iv,
        return_report=True,
    )

    print(report.summary())
    if cleaned.empty:
        print("nothing survived cleaning; check --spot, --rate and the quote columns", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.out, index=False)

    print(f"\nspot {spot:.2f}  expiries {cleaned['expiry'].nunique()}  quotes {len(cleaned)}")
    print(cleaned.groupby("expiry").agg(n=("iv", "size"), iv_atm=("iv", "median")).round(4).to_string())
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
