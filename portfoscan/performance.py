"""Performance analytics: CAPM beta regression and a benchmark comparison
that simulates buying the S&P 500 instead of your actual stock picks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .parser import BUY_TYPES, SELL_TYPES
from .portfolio import Position

BENCHMARK_TICKER = "^GSPC"
BENCHMARK_NAME = "S&P 500"


def _normalize_naive(dates: pd.Series) -> pd.Series:
    """Calendar date only, tz-naive — so trade dates line up with price-
    history dates (which are already tz-stripped in prices.py) when
    reindexing against the same DatetimeIndex."""
    dates = pd.to_datetime(dates)
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> tuple[float, float, float]:
    """OLS regression of daily stock returns on daily market returns:
    stock = alpha + beta * market. This is the standard textbook way to
    estimate a stock's beta. Returns (beta, alpha, r_squared); all NaN if
    there isn't enough overlapping history.
    """
    aligned = pd.concat([stock_returns.rename("stock"), market_returns.rename("market")], axis=1, join="inner").dropna()
    if len(aligned) < 2 or aligned["market"].var() == 0:
        return float("nan"), float("nan"), float("nan")

    beta, alpha = np.polyfit(aligned["market"], aligned["stock"], 1)
    r_squared = float(np.corrcoef(aligned["market"], aligned["stock"])[0, 1] ** 2)
    return float(beta), float(alpha), r_squared


def price_return_index(prices: pd.Series) -> pd.Series:
    """Cumulative price-return index, rebased to 100 at the first value."""
    prices = prices.dropna()
    if prices.empty:
        return prices
    return prices / prices.iloc[0] * 100


def build_benchmark_shadow_series(cash_flows: pd.Series, benchmark_prices: pd.Series) -> pd.Series:
    """Simulate investing every dollar actually moved into/out of stocks
    (`cash_flows`, aligned to the same date index, positive = a buy,
    negative = a sell) into the benchmark instead, on the same dates — same
    cost basis, same cash-flow timing, just a different asset. Returns the
    resulting hypothetical benchmark portfolio value, directly comparable in
    dollars to the real portfolio's market-value series.
    """
    price = benchmark_prices.reindex(cash_flows.index).ffill()
    shares_bought_or_sold = cash_flows / price
    shares_held = shares_bought_or_sold.cumsum()
    return shares_held * price


def build_portfolio_series(
    positions: dict[str, Position],
    price_histories: dict[str, pd.DataFrame],
    date_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """Reconstruct daily portfolio market value and same-day external cash
    flows (BUY = +amount, SELL = -amount) from each ticker's trade history
    and price history, aligned to `date_index`.
    """
    total_value = pd.Series(0.0, index=date_index)
    total_cash_flow = pd.Series(0.0, index=date_index)

    for ticker, pos in positions.items():
        if pos.trades.empty:
            continue
        trades = pos.trades[pos.trades["type"].isin(BUY_TYPES | SELL_TYPES)].copy()
        if trades.empty:
            continue
        trades["date"] = _normalize_naive(trades["date"])
        sign = trades["type"].apply(lambda t: 1.0 if t in BUY_TYPES else -1.0)

        shares_delta = (trades["quantity"] * sign).groupby(trades["date"]).sum()
        shares_held = shares_delta.reindex(date_index, fill_value=0.0).cumsum()

        prices = price_histories.get(ticker)
        if prices is None or prices.empty:
            continue
        price_series = prices.set_index(prices["Date"].dt.normalize())["Close"].reindex(date_index, method="ffill")
        total_value = total_value.add(shares_held * price_series.fillna(0.0), fill_value=0.0)

        cash_flow = (trades["amount_usd"] * sign).groupby(trades["date"]).sum()
        total_cash_flow = total_cash_flow.add(cash_flow.reindex(date_index, fill_value=0.0), fill_value=0.0)

    return total_value, total_cash_flow
