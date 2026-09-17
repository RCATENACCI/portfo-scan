"""Live and historical market prices via yfinance (Revolut has no public API
for this, see project README)."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

from .fx import get_latest_usd_rate, get_usd_rate_series

# Revolut exports the bare ticker with no exchange suffix, but Yahoo Finance
# requires one for anything not listed on a US exchange (US stocks like AAPL
# resolve as-is; a European UCITS ETF like VUAA does not). If a ticker fails
# to resolve, look it up on https://finance.yahoo.com/lookup and add the
# Yahoo-qualified symbol here — matching the exchange/currency the position
# was actually bought in keeps prices consistent with the recorded cost
# basis (e.g. a EUR purchase should map to the Xetra ".DE" listing, not a
# GBP one on the LSE, even though both technically track the same fund).
TICKER_OVERRIDES: dict[str, str] = {
    "VUAA": "VUAA.DE",  # Vanguard S&P 500 UCITS ETF (Acc), Xetra
    "EUNM": "EUNM.DE",  # iShares Core MSCI EM IMI UCITS ETF, Xetra
    "1TY": "1TY.DE",  # short-duration bond ETF, Xetra
}

# The 11 standard GICS sectors, so the sector-allocation view can show every
# sector even at $0.
ALL_SECTORS = [
    "Information Technology",
    "Financials",
    "Health Care",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]

# Yahoo Finance reports sectors using Morningstar's names, not GICS. Map them
# onto the GICS names above; entries absent here (Communication Services,
# Industrials, Energy, Utilities, Real Estate) already match.
_YAHOO_TO_GICS_SECTOR = {
    "Technology": "Information Technology",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
}


def to_yahoo_symbol(ticker: str) -> str:
    """Map a Revolut ticker to the Yahoo Finance symbol it actually resolves
    under (see TICKER_OVERRIDES above). Public so other modules needing a
    Yahoo-qualified symbol — e.g. news.py's per-ticker news lookup — stay
    consistent with prices/sector/name lookups instead of re-deriving it."""
    return TICKER_OVERRIDES.get(ticker, ticker)


def _strip_tz(dates: pd.Series) -> pd.Series:
    """yfinance returns tz-aware timestamps (exchange local time); strip that
    so dates line up cleanly against our tz-naive transaction dates."""
    return dates.dt.tz_localize(None) if dates.dt.tz is not None else dates


@st.cache_data(ttl=300, show_spinner="Fetching live prices...")
def get_live_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    """Latest close price per ticker, converted to USD. Missing/unresolvable
    tickers map to NaN rather than raising, so one bad symbol doesn't break
    the dashboard.

    Yahoo Finance quotes each ticker in its own listing currency — a
    Xetra-listed UCITS ETF like VUAA.DE comes back in EUR, not USD — so a
    non-USD quote is converted at today's rate before being reported,
    keeping every position comparable in the same currency.
    """
    prices: dict[str, float] = {}
    for ticker in tickers:
        try:
            # A few days, not one: some exchanges report today's close with
            # a lag, and a bare period="1d" fetch can land on that one row
            # while it's still NaN, showing "no price" for an otherwise
            # perfectly resolvable ticker until the feed catches up.
            history = yf.Ticker(to_yahoo_symbol(ticker)).history(period="5d")
            closes = history["Close"].dropna()
            native_price = float(closes.iloc[-1]) if not closes.empty else float("nan")
            currency = (get_ticker_info(ticker).get("currency") or "USD").upper()
            rate = get_latest_usd_rate(currency)
            prices[ticker] = native_price * rate if rate is not None else native_price
        except Exception:
            prices[ticker] = float("nan")
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_info(ticker: str) -> dict:
    """Cached raw Yahoo Finance `.info` for one ticker. Sector and company
    name both need this, so sharing one cached fetch (instead of each
    calling yfinance separately) halves the Yahoo requests per ticker —
    which also means less exposure to Yahoo's rate-limiting.
    """
    try:
        return yf.Ticker(to_yahoo_symbol(ticker)).info
    except Exception:
        return {}


_FUND_QUOTE_TYPES = {"ETF", "MUTUALFUND", "INDEX"}

ETF_SECTOR = "ETF & Others"


@st.cache_data(ttl=3600, show_spinner="Fetching sector data...")
def get_sectors(tickers: tuple[str, ...]) -> dict[str, str]:
    """GICS sector per ticker. ETFs/funds structurally have no single GICS
    sector (Yahoo reports none), so they're labeled 'ETF & Others' instead of
    'Unknown' — that label is reserved for genuine fetch failures on an
    individual stock.

    Cached for an hour, not a day: a transient fetch failure (e.g. Yahoo
    rate-limiting) falls back to 'Unknown' same as a real gap, and caching
    that failure for 24h would leave it looking broken for a full day with
    no way to retry sooner than that.
    """
    sectors: dict[str, str] = {}
    for ticker in tickers:
        info = get_ticker_info(ticker)
        raw = info.get("sector")
        if not raw:
            quote_type = (info.get("quoteType") or "").upper()
            sectors[ticker] = ETF_SECTOR if quote_type in _FUND_QUOTE_TYPES else "Unknown"
        else:
            sectors[ticker] = _YAHOO_TO_GICS_SECTOR.get(raw, raw)
    return sectors


@st.cache_data(ttl=3600, show_spinner="Fetching company names...")
def get_company_names(tickers: tuple[str, ...]) -> dict[str, str]:
    """Short company/fund name per ticker, for display instead of the bare
    ticker symbol. Falls back to the ticker itself if Yahoo doesn't have a
    name for it (e.g. a fetch failure, or a ticker Yahoo can't resolve)."""
    names: dict[str, str] = {}
    for ticker in tickers:
        info = get_ticker_info(ticker)
        names[ticker] = info.get("shortName") or info.get("longName") or ticker
    return names


@st.cache_data(ttl=3600, show_spinner="Fetching price history...")
def get_price_history(ticker: str, period: str = "6mo", start: str | None = None) -> pd.DataFrame:
    """Daily close-price history for one ticker in USD, or an empty
    DataFrame if unavailable. Pass `start` (as 'YYYY-MM-DD') for a fixed
    start date instead of a relative `period` — used for since-investment
    and beta comparisons against a fixed benchmark window.

    A non-USD-listed ticker (e.g. a Xetra ETF quoted in EUR) is converted
    day-by-day using that day's actual exchange rate, not one flat rate, so
    the shape of the return series isn't distorted by FX drift over the
    window — this matters for beta/backtest comparisons, not just the
    current value.
    """
    try:
        yf_ticker = yf.Ticker(to_yahoo_symbol(ticker))
        history = yf_ticker.history(start=start) if start else yf_ticker.history(period=period)
        df = history.reset_index()[["Date", "Close"]].dropna(subset=["Close"])
        df["Date"] = _strip_tz(df["Date"])

        currency = (get_ticker_info(ticker).get("currency") or "USD").upper()
        if currency != "USD" and not df.empty:
            start_str = df["Date"].min().strftime("%Y-%m-%d")
            end_str = df["Date"].max().strftime("%Y-%m-%d")
            rate_series = get_usd_rate_series(currency, start_str, end_str)
            if rate_series:
                rates = pd.Series(rate_series, name="rate")
                rates.index = pd.to_datetime(rates.index)
                rates = rates.sort_index().reindex(df["Date"].sort_values().unique(), method="ffill").bfill()
                df["Close"] = df["Close"] * df["Date"].map(rates)

        return df
    except Exception:
        return pd.DataFrame(columns=["Date", "Close"])
