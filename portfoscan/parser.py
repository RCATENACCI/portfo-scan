"""Parse Revolut trading CSV exports into a clean transactions DataFrame.

Revolut's "Invest" export covers stocks, ETFs, and bonds through the same
CSV shape, differing mainly in which `Type` values show up (e.g. bonds add
BOND COUPON / BOND REDEMPTION instead of DIVIDEND / SELL).
"""
from __future__ import annotations

import re

import pandas as pd

from .fx import convert_amounts_to_usd

# Opening a position (adds shares/units).
BUY_TYPES = {"BUY - MARKET", "BUY - LIMIT"}
# Closing/reducing a position. A bond redemption behaves exactly like a sell
# for accounting purposes: it returns cash and zeroes out the holding.
SELL_TYPES = {"SELL - MARKET", "SELL - LIMIT", "BOND REDEMPTION"}
TRADE_TYPES = BUY_TYPES | SELL_TYPES

# Income paid out on a held position without changing quantity.
INCOME_TYPES = {"DIVIDEND", "BOND COUPON"}

CASH_TYPES = {
    "CASH TOP-UP",
    "CASH WITHDRAWAL",
    "STOCKS PROMOTION REWARD",
    "STOCKS PROMOTION CLAWBACK",
    "REWARD",
}

KNOWN_TYPES = TRADE_TYPES | INCOME_TYPES | CASH_TYPES

_COLUMN_MAP = {
    "Date": "date",
    "Ticker": "ticker",
    "Type": "type",
    "Quantity": "quantity",
    "Price per share": "price",
    "Total Amount": "amount",
    "Currency": "currency",
    "FX Rate": "fx_rate",
}

_MONEY_RE = re.compile(r"^[A-Z]{3}\s*(-?[\d,]+\.?\d*)$")


def _parse_money(value: object) -> float:
    """Turn a 'USD 93.78' style cell into a plain float."""
    if pd.isna(value) or value == "":
        return float("nan")
    match = _MONEY_RE.match(str(value).strip())
    if not match:
        raise ValueError(f"Unrecognized money format: {value!r}")
    return float(match.group(1).replace(",", ""))


def load_transactions(csv_path: str) -> pd.DataFrame:
    """Load a Revolut transactions CSV export into a normalized DataFrame.

    Columns: date, ticker, type, quantity, price, amount, currency, fx_rate,
    plus price_usd/amount_usd — `price`/`amount` as originally recorded in
    the trade's native currency (`currency`), converted to USD using that
    day's actual exchange rate. Every money computation elsewhere in the app
    (cost basis, cash balance, P&L, performance series) uses the _usd
    columns so a EUR-denominated trade (e.g. a Xetra-listed UCITS ETF)
    doesn't get silently treated as if it were USD. Rows are sorted
    chronologically.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns=_COLUMN_MAP)
    # Revolut exports mix microsecond-precision and whole-second timestamps
    # in the same file (e.g. "2025-01-01T18:35:57.965947Z" next to
    # "2025-01-02T07:01:29Z"). A plain pd.to_datetime() infers one fixed
    # format from the first rows and then raises on the rest; ISO8601 mode
    # accepts either precision per-row.
    df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    df["price"] = df["price"].apply(_parse_money)
    df["amount"] = df["amount"].apply(_parse_money)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["fx_rate"] = pd.to_numeric(df["fx_rate"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    df["amount_usd"] = convert_amounts_to_usd(df["amount"], df["currency"], df["date"])
    # Same helper, same (currency, date) cache entries as amount_usd above —
    # this doesn't cost a second round of API calls.
    df["price_usd"] = convert_amounts_to_usd(df["price"], df["currency"], df["date"])

    return df


def find_unknown_types(transactions: pd.DataFrame) -> list[str]:
    """Transaction types this file uses that portfoscan doesn't recognize.

    New Revolut transaction types (or an export from an account with
    instruments/features this app hasn't seen yet) would otherwise be
    silently excluded from P&L — this lets the app flag them instead, for
    any CSV, rather than requiring each new type to be discovered by a bug
    report first.
    """
    unknown = transactions.loc[~transactions["type"].isin(KNOWN_TYPES), "type"]
    return sorted(unknown.dropna().unique())
