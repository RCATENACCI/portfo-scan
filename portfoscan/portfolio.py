"""Turn a transactions DataFrame into per-ticker positions: cost basis, realized
P&L, and dividends. Live prices (for unrealized P&L) are layered on separately
in prices.py, since that requires a network call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .parser import BUY_TYPES, INCOME_TYPES, SELL_TYPES

_OPEN_QTY_EPSILON = 1e-6


@dataclass
class Position:
    ticker: str
    quantity: float = 0.0
    cost_basis: float = 0.0
    cost_basis_sold: float = 0.0
    realized_pnl: float = 0.0
    dividends: float = 0.0
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def avg_price(self) -> float:
        return self.cost_basis / self.quantity if self.quantity > _OPEN_QTY_EPSILON else 0.0

    @property
    def is_open(self) -> bool:
        return self.quantity > _OPEN_QTY_EPSILON

    @property
    def total_invested(self) -> float:
        """All-time capital put into this ticker: what's still in the open
        position plus the cost basis of the shares that have since been sold."""
        return self.cost_basis + self.cost_basis_sold


def build_positions(transactions: pd.DataFrame) -> dict[str, Position]:
    """Replay every trade/dividend in chronological order to build one
    Position per ticker, using average-cost-basis accounting.
    """
    positions: dict[str, Position] = {}
    trade_rows: dict[str, list] = {}

    for row in transactions.itertuples(index=False):
        if pd.isna(row.ticker):
            continue

        pos = positions.setdefault(row.ticker, Position(ticker=row.ticker))

        if row.type in BUY_TYPES:
            pos.cost_basis += row.amount_usd
            pos.quantity += row.quantity
            trade_rows.setdefault(row.ticker, []).append(row)
        elif row.type in SELL_TYPES:
            avg_before = pos.avg_price
            cost_removed = row.quantity * avg_before
            pos.realized_pnl += row.amount_usd - cost_removed
            pos.cost_basis -= cost_removed
            pos.cost_basis_sold += cost_removed
            pos.quantity -= row.quantity
            trade_rows.setdefault(row.ticker, []).append(row)
        elif row.type in INCOME_TYPES:
            pos.dividends += row.amount_usd
            trade_rows.setdefault(row.ticker, []).append(row)

    for ticker, rows in trade_rows.items():
        positions[ticker].trades = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    return positions


def cash_balance(transactions: pd.DataFrame) -> float:
    """Net cash movement across the whole account, in USD: buys/sells
    (stocks, ETFs, bonds), dividends/coupons, redemptions, top-ups,
    withdrawals, and rewards/clawbacks.

    Every type except BUY already carries the correct sign in `amount_usd`
    (sells/income/top-ups/rewards are positive inflows; withdrawals and
    clawbacks arrive pre-negated from Revolut). Each cash flow converts at
    its own transaction date's exchange rate — this is the USD value of the
    cash *when it moved*, not a live valuation of whatever currency wallet
    still holds it.
    """
    cash = 0.0
    for row in transactions.itertuples(index=False):
        if row.type in BUY_TYPES:
            cash -= row.amount_usd
        else:
            cash += row.amount_usd
    return cash
