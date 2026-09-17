"""USD currency conversion.

Revolut records each trade in whatever currency it was actually executed in
(e.g. EUR for a Xetra-listed UCITS ETF, USD for a US stock or Treasury), and
the price history for a non-US-listed ticker comes back from Yahoo Finance in
that same native currency. Since this app reports one consolidated portfolio
in USD, both need converting.

Rates come from the Frankfurter API (https://frankfurter.dev), a free,
keyless wrapper around the European Central Bank's daily reference rates —
no signup, no rate limit that matters here. A requested date with no ECB
rate (weekend/holiday) automatically falls back to the nearest earlier
business day, which is exactly what you want for a trade settlement date.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import requests
import streamlit as st

_FRANKFURTER_URL = "https://api.frankfurter.dev/v1"


@st.cache_data(ttl=21600, show_spinner=False)
def get_historical_usd_rate(currency: str, as_of: str) -> float | None:
    """USD value of 1 unit of `currency` on (at/before) `as_of` (YYYY-MM-DD).
    Cached for 6h since a historical rate never changes once published."""
    if currency == "USD":
        return 1.0
    try:
        response = requests.get(f"{_FRANKFURTER_URL}/{as_of}", params={"base": currency, "symbols": "USD"}, timeout=10)
        response.raise_for_status()
        return response.json()["rates"]["USD"]
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_latest_usd_rate(currency: str) -> float | None:
    """Today's USD value of 1 unit of `currency`, for pricing open positions
    held in a non-USD currency."""
    if currency == "USD":
        return 1.0
    try:
        response = requests.get(f"{_FRANKFURTER_URL}/latest", params={"base": currency, "symbols": "USD"}, timeout=10)
        response.raise_for_status()
        return response.json()["rates"]["USD"]
    except Exception:
        return None


@st.cache_data(ttl=21600, show_spinner=False)
def get_usd_rate_series(currency: str, start: str, end: str) -> dict[str, float]:
    """Daily USD value of 1 unit of `currency` for every ECB business day
    between `start` and `end` (YYYY-MM-DD, inclusive) — one request for a
    whole price-history range instead of one call per day."""
    if currency == "USD" or start > end:
        return {}
    try:
        response = requests.get(
            f"{_FRANKFURTER_URL}/{start}..{end}", params={"base": currency, "symbols": "USD"}, timeout=15
        )
        response.raise_for_status()
        return {day: rates["USD"] for day, rates in response.json()["rates"].items()}
    except Exception:
        return {}


def convert_amounts_to_usd(amounts: pd.Series, currencies: pd.Series, dates: pd.Series) -> pd.Series:
    """Convert a Series of native-currency amounts to USD, using each row's
    own currency and transaction date. Rows already in USD (the common case)
    are left untouched; a currency/date pair that fails to fetch leaves the
    native amount as-is rather than dropping the value."""
    usd = amounts.astype(float).copy()
    day_strings = dates.dt.strftime("%Y-%m-%d")
    for currency in currencies.dropna().unique():
        if currency == "USD":
            continue
        currency_mask = currencies == currency
        for day in day_strings[currency_mask].unique():
            rate = get_historical_usd_rate(currency, day)
            if rate is None:
                continue
            row_mask = currency_mask & (day_strings == day)
            usd.loc[row_mask] = amounts.loc[row_mask] * rate
    return usd
