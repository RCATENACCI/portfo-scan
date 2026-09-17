"""Bond classification and analytics.

Revolut's CSV gives bonds an ISIN in the ticker column (e.g.
"US91282CFW64") rather than a stock ticker, and Yahoo Finance can't
resolve those at all — it doesn't carry bond reference data. For US
Treasuries specifically, the U.S. Treasury's own Fiscal Data API is a
free, public, no-key-required source of the real security terms
(maturity, coupon, payment frequency), keyed by CUSIP. This module:

- Detects which positions are bonds — from their transaction types
  (BOND COUPON / BOND REDEMPTION) if any have happened yet, or from the
  ticker itself looking like an ISIN otherwise, so a bond bought only
  yesterday (no coupon/redemption history yet) is still recognized
  instead of silently being treated as an unresolvable stock.
- Extracts a CUSIP from a US ISIN and looks it up against that API.
- Falls back to estimating economics from the coupon payments actually
  observed in the CSV for anything the Treasury API doesn't cover
  (corporate bonds, non-US government bonds).
- Computes Macaulay/modified duration and a cash-flow schedule once a
  maturity date and coupon are known, from either source.
- Prices open Treasury positions by discounting their remaining cash
  flows at today's actual market yield for that time-to-maturity
  (from the Treasury's public daily par yield curve), instead of
  assuming they trade at par — there's no live per-CUSIP quote
  available anywhere free, but the yield curve gets much closer to a
  real price than a flat par assumption, especially further from
  maturity or after rates have moved since issuance.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import date

import pandas as pd
import requests
import streamlit as st

from .portfolio import Position

BOND_INCOME_TYPE = "BOND COUPON"
BOND_REDEMPTION_TYPE = "BOND REDEMPTION"
GOVERNMENT_SECTOR = "Government Bonds"
CORPORATE_SECTOR = "Corporate / Other Bonds"

_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

_TREASURY_AUCTIONS_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
)
_YIELD_CURVE_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all"
)
# home.treasury.gov blocks requests without a browser-like User-Agent.
_BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; portfoscan/1.0)"}

_FREQUENCY_PER_YEAR = {
    "Monthly": 12,
    "Quarterly": 4,
    "Semi-Annual": 2,
    "Annual": 1,
}

_YIELD_CURVE_TENOR_YEARS = {
    "1 Mo": 1 / 12,
    "1.5 Month": 1.5 / 12,
    "2 Mo": 2 / 12,
    "3 Mo": 3 / 12,
    "4 Mo": 4 / 12,
    "6 Mo": 0.5,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}


def is_bond_position(ticker: str, pos: Position) -> bool:
    """A position is a bond if any of its trades are bond-specific types, or
    if the ticker itself is shaped like an ISIN (Revolut's bond ticker
    format) — no ordinary stock/ETF ticker looks like a 12-character ISIN,
    so this catches a bond bought before its first coupon/redemption ever
    happens, not just ones with payment history already.
    """
    if _ISIN_RE.match(ticker):
        return True
    if pos.trades.empty:
        return False
    return pos.trades["type"].isin({BOND_INCOME_TYPE, BOND_REDEMPTION_TYPE}).any()


def isin_to_us_cusip(ticker: str) -> str | None:
    """Extract the 9-character CUSIP from a US ISIN (e.g. "US91282CFW64"
    -> "91282CFW6"). Returns None for anything that isn't a US ISIN."""
    if not _ISIN_RE.match(ticker) or not ticker.startswith("US"):
        return None
    return ticker[2:11]


def _clean(value: object) -> float | None:
    if value is None or value == "null":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class TreasurySecurity:
    cusip: str
    security_type: str
    security_term: str
    issue_date: pd.Timestamp
    maturity_date: pd.Timestamp
    coupon_rate: float  # annual %, 0 for zero-coupon bills
    payments_per_year: int  # 0 for zero-coupon bills
    yield_at_auction: float | None  # annual %, used as a duration discount-rate proxy


@st.cache_data(ttl=86400, show_spinner="Looking up Treasury security...")
def get_treasury_security(cusip: str) -> TreasurySecurity | None:
    """Look up a US Treasury security's real terms by CUSIP via the
    Treasury's free public Fiscal Data API. Once a security is issued its
    terms never change, so this is safe to cache for a long time.
    """
    try:
        response = requests.get(
            _TREASURY_AUCTIONS_URL,
            params={
                "filter": f"cusip:eq:{cusip}",
                "fields": "cusip,security_type,security_term,auction_date,issue_date,maturity_date,"
                "int_rate,int_payment_frequency,high_yield,high_discnt_rate",
                "sort": "auction_date",
                "page[size]": "1",
            },
            timeout=10,
        )
        response.raise_for_status()
        rows = response.json().get("data", [])
        if not rows:
            return None
        row = rows[0]

        coupon_rate = _clean(row["int_rate"]) or 0.0
        frequency = _FREQUENCY_PER_YEAR.get(row["int_payment_frequency"], 0)
        yield_at_auction = _clean(row["high_yield"])
        if yield_at_auction is None:
            yield_at_auction = _clean(row["high_discnt_rate"])

        return TreasurySecurity(
            cusip=row["cusip"],
            security_type=row["security_type"],
            security_term=row["security_term"],
            issue_date=pd.Timestamp(row["issue_date"]),
            maturity_date=pd.Timestamp(row["maturity_date"]),
            coupon_rate=coupon_rate,
            payments_per_year=frequency,
            yield_at_auction=yield_at_auction,
        )
    except Exception:
        return None


@st.cache_data(ttl=21600, show_spinner="Fetching Treasury yield curve...")
def get_current_treasury_yield_curve() -> dict[float, float] | None:
    """Most recent daily Treasury par yield curve (years-to-maturity ->
    annual yield %), free and public from home.treasury.gov. Used to price
    open Treasury bonds by discounted cash flow instead of assuming they
    trade at par.

    Refreshed every 6 hours: the curve itself only updates once per
    business day, but this keeps a stale value from lingering too long
    once a new one is out, without hammering the endpoint on every rerun.
    """
    try:
        response = requests.get(
            _YIELD_CURVE_URL.format(year=date.today().year),
            params={
                "type": "daily_treasury_yield_curve",
                "field_tdr_date_value": date.today().year,
                "page": "",
                "_format": "csv",
            },
            headers=_BROWSER_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        reader = csv.DictReader(io.StringIO(response.text))
        latest_row = next(reader)  # Treasury lists most-recent date first.

        curve: dict[float, float] = {}
        for label, years in _YIELD_CURVE_TENOR_YEARS.items():
            value = _clean(latest_row.get(label))
            if value is not None:
                curve[years] = value
        return curve or None
    except Exception:
        return None


def _interpolate_yield(curve: dict[float, float], years: float) -> float:
    points = sorted(curve.items())
    if years <= points[0][0]:
        return points[0][1]
    if years >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= years <= x1:
            weight = (years - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    return points[-1][1]


def price_from_yield_curve(
    face_value: float,
    coupon_rate: float,
    payments_per_year: int,
    maturity_date: pd.Timestamp,
    valuation_date: pd.Timestamp | None = None,
) -> float | None:
    """Fair value of a Treasury via discounted cash flow at today's actual
    market yield for its remaining time-to-maturity (interpolated from the
    public daily par yield curve) — the standard way to mark a bond to
    market absent a live per-CUSIP quote, which no free source provides for
    individual Treasury CUSIPs. Returns None if the curve can't be fetched
    or the bond has already matured.
    """
    curve = get_current_treasury_yield_curve()
    valuation_date = valuation_date or pd.Timestamp(date.today())
    if curve is None or maturity_date <= valuation_date:
        return None

    years_to_maturity = (maturity_date - valuation_date).days / 365.25
    yield_rate = _interpolate_yield(curve, years_to_maturity)

    if payments_per_year <= 0:
        return face_value / (1 + yield_rate / 100) ** years_to_maturity

    coupon_amount = face_value * coupon_rate / 100 / payments_per_year
    months_per_period = 12 // payments_per_year
    period_yield = yield_rate / 100 / payments_per_year

    dates = []
    current = maturity_date
    while current > valuation_date:
        dates.append(current)
        current = current - pd.DateOffset(months=months_per_period)
    dates.sort()
    if not dates:
        return None

    price = 0.0
    for i, d in enumerate(dates, start=1):
        cash_flow = coupon_amount + face_value if d == dates[-1] else coupon_amount
        price += cash_flow / (1 + period_yield) ** i
    return price


@dataclass
class BondEconomics:
    face_value: float
    coupon_rate: float | None  # annual %, None if it can't be estimated at all
    payments_per_year: int | None
    maturity_date: pd.Timestamp | None  # only known if already redeemed, absent a Treasury match
    is_estimate: bool


def estimate_bond_economics(pos: Position) -> BondEconomics:
    """Best-effort bond terms derived only from the CSV, for bonds the
    Treasury API doesn't cover (corporate bonds, non-US governments).

    Face value: the redemption payout is the ground truth if the bond has
    matured; otherwise assumes the standard $100 convention Revolut's own
    per-unit bond pricing already quotes against.

    Coupon rate: annualized from the actual coupon payments received, using
    the payment frequency implied by the gaps between them when there are
    at least two — otherwise this can't be determined and is left as None
    rather than guessing a frequency out of thin air.
    """
    redemptions = pos.trades[pos.trades["type"] == BOND_REDEMPTION_TYPE]
    if not redemptions.empty and redemptions["quantity"].sum() > 0:
        face_value = float(redemptions["amount_usd"].sum() / redemptions["quantity"].sum())
        maturity_date = pd.Timestamp(redemptions["date"].max())
    else:
        face_value = 100.0
        maturity_date = None

    coupons = pos.trades[pos.trades["type"] == BOND_INCOME_TYPE].sort_values("date")
    coupon_rate: float | None = None
    payments_per_year: int | None = None
    if len(coupons) >= 2:
        gaps_days = coupons["date"].diff().dropna().dt.days
        avg_gap_days = float(gaps_days.mean())
        payments_per_year = max(1, round(365.25 / avg_gap_days)) if avg_gap_days > 0 else None
        if payments_per_year:
            avg_payment = float(coupons["amount_usd"].mean())
            per_unit_payment = avg_payment / pos.quantity if pos.quantity > 0 else avg_payment
            coupon_rate = per_unit_payment * payments_per_year / face_value * 100

    return BondEconomics(
        face_value=face_value,
        coupon_rate=coupon_rate,
        payments_per_year=payments_per_year,
        maturity_date=maturity_date,
        is_estimate=True,
    )


def build_cash_flow_schedule(
    face_value: float,
    coupon_rate: float,
    payments_per_year: int,
    issue_date: pd.Timestamp,
    maturity_date: pd.Timestamp,
) -> pd.DataFrame:
    """Theoretical coupon + principal cash-flow schedule from issue to
    maturity, evenly spaced by the payment frequency. This is the schedule
    implied by the security's own terms, not adjusted for the quantity
    actually held.
    """
    if payments_per_year <= 0:
        return pd.DataFrame([{"date": maturity_date, "amount": face_value, "type": "Principal"}])

    coupon_amount = face_value * coupon_rate / 100 / payments_per_year
    months_per_period = 12 // payments_per_year

    dates = []
    current = maturity_date
    while current > issue_date:
        dates.append(current)
        current = current - pd.DateOffset(months=months_per_period)
    dates.sort()

    rows = [{"date": d, "amount": coupon_amount, "type": "Coupon"} for d in dates]
    if rows:
        rows[-1] = {"date": maturity_date, "amount": coupon_amount + face_value, "type": "Coupon + Principal"}
    return pd.DataFrame(rows)


def compute_duration(
    face_value: float,
    coupon_rate: float,
    payments_per_year: int,
    maturity_date: pd.Timestamp,
    yield_rate: float,
    valuation_date: pd.Timestamp | None = None,
) -> tuple[float, float] | tuple[None, None]:
    """Macaulay and modified duration in years, from the security's own
    terms and a yield proxy (since no live market price/yield is available
    for a specific bond CUSIP) — an approximation based on original terms,
    not a live market-duration figure. Returns (None, None) if there's no
    remaining cash flow to discount (already matured).
    """
    valuation_date = valuation_date or pd.Timestamp(date.today())
    if maturity_date <= valuation_date:
        return None, None

    if payments_per_year <= 0:
        years_to_maturity = (maturity_date - valuation_date).days / 365.25
        return years_to_maturity, years_to_maturity / (1 + yield_rate / 100)

    coupon_amount = face_value * coupon_rate / 100 / payments_per_year
    months_per_period = 12 // payments_per_year
    period_yield = yield_rate / 100 / payments_per_year

    dates = []
    current = maturity_date
    while current > valuation_date:
        dates.append(current)
        current = current - pd.DateOffset(months=months_per_period)
    dates.sort()
    if not dates:
        return None, None

    total_pv = 0.0
    weighted_pv = 0.0
    for i, d in enumerate(dates, start=1):
        cash_flow = coupon_amount + face_value if d == dates[-1] else coupon_amount
        pv = cash_flow / (1 + period_yield) ** i
        years = (d - valuation_date).days / 365.25
        total_pv += pv
        weighted_pv += pv * years

    if total_pv == 0:
        return None, None

    macaulay = weighted_pv / total_pv
    modified = macaulay / (1 + yield_rate / 100 / payments_per_year)
    return macaulay, modified


def classify_bond_sector(treasury: TreasurySecurity | None) -> str:
    return GOVERNMENT_SECTOR if treasury is not None else CORPORATE_SECTOR
