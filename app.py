"""portfoscan — interactive dashboard for a Revolut investing portfolio.

Run with: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from portfoscan.bonds import (
    BOND_INCOME_TYPE,
    BOND_REDEMPTION_TYPE,
    build_cash_flow_schedule,
    classify_bond_sector,
    compute_duration,
    estimate_bond_economics,
    get_treasury_security,
    is_bond_position,
    isin_to_us_cusip,
    price_from_yield_curve,
)
from portfoscan.news import (
    ECB_MEETINGS,
    ECB_SOURCE_URL,
    FOMC_MEETINGS,
    FOMC_SOURCE_URL,
    get_ecb_deposit_rate,
    get_fed_funds_target_range,
    get_fed_meeting_probabilities,
    get_ticker_news,
    next_meeting,
    search_news,
    search_news_topics,
    time_ago,
)
from portfoscan.parser import BUY_TYPES, SELL_TYPES, find_unknown_types, load_transactions
from portfoscan.performance import (
    BENCHMARK_NAME,
    BENCHMARK_TICKER,
    build_benchmark_shadow_series,
    build_portfolio_series,
    compute_beta,
    price_return_index,
)
from portfoscan.portfolio import build_positions, cash_balance
from portfoscan.prices import (
    ALL_SECTORS,
    get_company_names,
    get_live_prices,
    get_price_history,
    get_sectors,
    get_ticker_info,
)

DEFAULT_CSV = Path(__file__).parent / "data" / "raw" / "transactions.csv"
EXAMPLE_CSV = Path(__file__).parent / "data" / "raw" / "example-portfolio.csv"

st.set_page_config(page_title="portfoscan", page_icon="📊", layout="wide")


@st.cache_data
def _load(csv_bytes_or_path):
    return load_transactions(csv_bytes_or_path)


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def qty(value: float) -> str:
    return f"{value:,.2f}"


def style_fig(fig: go.Figure) -> go.Figure:
    """Apply a template matching the active light/dark theme and make the
    chart background transparent, so it blends into the Streamlit page
    instead of showing Plotly's own (light-by-default) background."""
    is_light = st.context.theme.type == "light"
    fig.update_layout(
        template="plotly_white" if is_light else "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------- sidebar --
st.sidebar.title("📊 portfoscan")
st.sidebar.caption(
    "An interactive dashboard for your Revolut investing portfolio — P&L, "
    "allocation, and performance vs the S&P 500."
)
uploaded = st.sidebar.file_uploader("Upload a Revolut CSV export", type="csv")
if st.sidebar.button("🔄 Refresh live prices"):
    get_live_prices.clear()
    get_price_history.clear()
    get_sectors.clear()
    get_company_names.clear()
    get_ticker_info.clear()
    search_news.clear()
    get_ticker_news.clear()

st.sidebar.divider()
st.sidebar.caption(
    "Built by **Elouan Bahri**  \n"
    "Questions? [elouan.bahri1@berkeley.edu](mailto:elouan.bahri1@berkeley.edu)"
)

using_example = False
if uploaded is not None:
    source = uploaded
elif DEFAULT_CSV.exists():
    source = DEFAULT_CSV
elif EXAMPLE_CSV.exists():
    source = EXAMPLE_CSV
    using_example = True
else:
    st.info(
        "👋 **Upload your Revolut transactions CSV in the sidebar to get started.**\n\n"
        "In the Revolut app: **Invest → Statements → Export → CSV**, "
        "then upload the file here."
    )
    st.stop()

if using_example:
    st.info(
        "👀 **You're viewing an example portfolio** — real trades of mine, in companies I like, "
        "shown here to demonstrate how portfoscan works. Upload your own Revolut CSV in the sidebar "
        "to see your own data instead."
    )

transactions = _load(source)

unknown_types = find_unknown_types(transactions)
if unknown_types:
    st.warning(
        f"Found transaction type(s) portfoscan doesn't recognize yet: **{', '.join(unknown_types)}**. "
        "Those rows are skipped for now, so any positions/cash they affect may be understated. "
        "Let Elouan know (see sidebar) and they can be added."
    )

# --------------------------------------------------------------- compute --
positions = build_positions(transactions)
cash = cash_balance(transactions)
open_positions = {t: p for t, p in positions.items() if p.is_open}
closed_positions = {t: p for t, p in positions.items() if not p.is_open}
bond_positions = {t: p for t, p in positions.items() if is_bond_position(t, p)}
open_bond_tickers = set(bond_positions) & set(open_positions)

# Yahoo Finance has no data at all for bond ISINs/CUSIPs (unlike stocks/ETFs,
# which just occasionally fail to fetch) — pull live prices only for
# non-bond tickers. Open Treasury bonds are priced by discounting their
# remaining cash flows at today's actual market yield (from the public
# yield curve) — no free source has a live per-CUSIP quote, but this gets
# much closer to a real price than assuming par. Anything else (corporate
# bonds, or if the yield curve fetch fails) falls back to par.
live_prices = get_live_prices(tuple(sorted(set(open_positions) - open_bond_tickers)))
for ticker, pos in bond_positions.items():
    if not pos.is_open:
        continue
    econ = estimate_bond_economics(pos)
    cusip = isin_to_us_cusip(ticker)
    treasury = get_treasury_security(cusip) if cusip else None
    market_price = (
        price_from_yield_curve(100.0, treasury.coupon_rate, treasury.payments_per_year, treasury.maturity_date)
        if treasury is not None
        else None
    )
    live_prices[ticker] = market_price if market_price is not None else econ.face_value

rows = []
for ticker, pos in open_positions.items():
    current_price = live_prices.get(ticker, float("nan"))
    market_value = pos.quantity * current_price if pd.notna(current_price) else float("nan")
    unrealized = market_value - pos.cost_basis if pd.notna(market_value) else float("nan")
    unrealized_pct = (unrealized / pos.cost_basis * 100) if pos.cost_basis > 0 and pd.notna(unrealized) else float("nan")
    rows.append(
        {
            "Ticker": ticker,
            "Quantity": pos.quantity,
            "Avg Entry": pos.avg_price,
            "Current Price": current_price,
            "Market Value": market_value,
            "Unrealized P&L": unrealized,
            "Unrealized %": unrealized_pct,
            "Realized P&L": pos.realized_pnl,
            "Dividends": pos.dividends,
        }
    )
holdings_df = pd.DataFrame(rows).sort_values("Market Value", ascending=False).reset_index(drop=True)

total_market_value = holdings_df["Market Value"].sum(skipna=True)
total_cost_basis = sum(p.cost_basis for p in open_positions.values())
total_unrealized = total_market_value - total_cost_basis
total_realized = sum(p.realized_pnl for p in positions.values())
total_dividends = sum(p.dividends for p in positions.values())
account_value = total_market_value + cash

sectors = get_sectors(tuple(sorted(set(open_positions) - open_bond_tickers)))
for ticker in open_bond_tickers:
    cusip = isin_to_us_cusip(ticker)
    treasury = get_treasury_security(cusip) if cusip else None
    sectors[ticker] = classify_bond_sector(treasury)

company_names = get_company_names(tuple(sorted(set(open_positions) - open_bond_tickers)))
for ticker in open_bond_tickers:
    cusip = isin_to_us_cusip(ticker)
    treasury = get_treasury_security(cusip) if cusip else None
    company_names[ticker] = f"{treasury.security_type} ({treasury.security_term})" if treasury else "Bond"
holdings_df["Name"] = holdings_df["Ticker"].map(company_names)

sector_value = (
    holdings_df.assign(Sector=holdings_df["Ticker"].map(sectors)).groupby("Sector")["Market Value"].sum(min_count=1)
    if not holdings_df.empty
    else pd.Series(dtype=float)
)
all_sector_names = list(dict.fromkeys(ALL_SECTORS + [s for s in sector_value.index if s not in ALL_SECTORS]))
sector_df = pd.DataFrame({"Sector": all_sector_names})
sector_df["Amount"] = sector_df["Sector"].map(sector_value).fillna(0.0)
sector_df["Percentage"] = (sector_df["Amount"] / total_market_value * 100) if total_market_value else 0.0
sector_df = sector_df.sort_values("Amount", ascending=False).reset_index(drop=True)

# ------------------------------------------------------------------ tabs --
tab_names = ["Overview", "News", "Stock Detail"]
if bond_positions:
    tab_names.append("Bond Detail")
tab_names.append("Transactions")
tabs = st.tabs(tab_names)
tab_overview, tab_news, tab_detail = tabs[0], tabs[1], tabs[2]
tab_bonds = tabs[3] if bond_positions else None
tab_transactions = tabs[-1]

with tab_overview:
    failed_price_tickers = sorted(
        t for t in open_positions if pd.isna(live_prices.get(t, float("nan")))
    )
    if failed_price_tickers:
        st.warning(
            f"Couldn't fetch a live price for: {', '.join(failed_price_tickers)}. Account Value, "
            "Unrealized P&L, and the charts below exclude them until the next successful fetch — try "
            "the sidebar's 🔄 **Refresh live prices** button; Yahoo Finance sometimes rate-limits requests."
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Account Value", money(account_value))
    c2.metric("Unrealized P&L", money(total_unrealized), pct(total_unrealized / total_cost_basis * 100) if total_cost_basis else None)
    c3.metric("Realized P&L", money(total_realized))
    c4.metric("Dividends Received", money(total_dividends))
    c5.metric("Cash Balance", money(cash))
    if open_bond_tickers:
        st.caption(
            f"Bonds ({', '.join(sorted(open_bond_tickers))}) in the figures above: US Treasuries are priced "
            "by discounting their remaining cash flows at today's actual market yield (no free source has a "
            "live per-CUSIP quote); anything else falls back to par. See the Bond Detail tab for each bond's "
            "actual terms."
        )

    st.subheader("Allocation")
    if not holdings_df.empty and holdings_df["Market Value"].notna().any():
        treemap_col, badge_col = st.columns([5, 1])

        with treemap_col:
            fig = px.treemap(
                holdings_df.dropna(subset=["Market Value"]),
                path=["Ticker"],
                values="Market Value",
                color="Unrealized %",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                custom_data=["Name"],
            )
            fig.update_traces(
                # `label` stays the ticker (path key) so click-to-select still
                # drives the Stock/Bond Detail tabs correctly — only the
                # displayed text swaps to the company/fund name for clarity.
                texttemplate="%{customdata[0]}<br>%{percentRoot:.0%}",
                hovertemplate="<b>%{customdata[0]}</b> (%{label})<br>Market Value: $%{value:,.2f}<br>Allocation: %{percentRoot:.1%}<br>Unrealized: %{color:.2f}%<extra></extra>",
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            style_fig(fig)
            treemap_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="allocation_treemap")
            clicked = treemap_event.selection.points[0]["label"] if treemap_event and treemap_event.selection and treemap_event.selection.points else None
            if clicked and clicked != st.session_state.get("_last_treemap_ticker"):
                st.session_state["selected_ticker"] = clicked
            st.session_state["_last_treemap_ticker"] = clicked

        with badge_col:
            badge_ticker = st.session_state.get("selected_ticker")
            badge_row = holdings_df[holdings_df["Ticker"] == badge_ticker] if badge_ticker else pd.DataFrame()
            if not badge_row.empty:
                st.markdown(f"**{badge_row['Name'].iloc[0]}** ({badge_ticker})")
                st.metric("Unrealized P&L", money(badge_row["Unrealized P&L"].iloc[0]), pct(badge_row["Unrealized %"].iloc[0]))
    else:
        st.info("No live prices available yet for an allocation chart.")

    st.subheader(f"What if you'd bought {BENCHMARK_NAME} instead?")
    all_trade_dates = transactions.loc[transactions["ticker"].notna(), "date"]
    if not all_trade_dates.empty:
        perf_start = all_trade_dates.min().normalize()
        benchmark_hist = get_price_history(BENCHMARK_TICKER, start=perf_start.strftime("%Y-%m-%d"))
        if not benchmark_hist.empty:
            date_index = pd.DatetimeIndex(sorted(benchmark_hist["Date"].dt.normalize().unique()))
            price_histories = {
                ticker: get_price_history(ticker, start=perf_start.strftime("%Y-%m-%d")) for ticker in positions
            }
            portfolio_value, cash_flows = build_portfolio_series(positions, price_histories, date_index)
            benchmark_price_series = benchmark_hist.set_index(benchmark_hist["Date"].dt.normalize())["Close"]
            shadow_value = build_benchmark_shadow_series(cash_flows, benchmark_price_series)

            if not portfolio_value.dropna().empty:
                perf_fig = go.Figure()
                perf_fig.add_trace(
                    go.Scatter(x=portfolio_value.index, y=portfolio_value, name="Your portfolio", line=dict(color="#636EFA"))
                )
                perf_fig.add_trace(
                    go.Scatter(
                        x=shadow_value.index,
                        y=shadow_value,
                        name=f"If {BENCHMARK_NAME} instead",
                        line=dict(color="#9AA0A6", dash="dash"),
                    )
                )
                perf_fig.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(orientation="h"),
                    yaxis_title="Value ($)",
                    yaxis_tickprefix="$",
                )
                style_fig(perf_fig)
                st.caption(
                    f"Simulates putting every dollar you actually invested — same cost basis, same dates, same "
                    f"deposits/withdrawals since {perf_start.date()} — into {BENCHMARK_NAME} instead of your stock "
                    "picks. Both lines are real dollars, so they're directly comparable without any indexing."
                )
                st.plotly_chart(perf_fig, use_container_width=True)

                final_actual = portfolio_value.dropna().iloc[-1]
                final_shadow = shadow_value.dropna().iloc[-1]
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Your portfolio", money(final_actual))
                pc2.metric(f"If {BENCHMARK_NAME} instead", money(final_shadow))
                pc3.metric("Difference", money(final_actual - final_shadow))
            else:
                st.info("Not enough price history to compute this comparison yet.")
        else:
            st.info(f"Couldn't fetch {BENCHMARK_NAME} history right now.")

    st.subheader("Sector Allocation")
    st.caption(
        "Percentage of invested (non-cash) portfolio value per sector. \"ETF & Others\" covers funds, which "
        "don't have a single GICS sector; \"Unknown\" means Yahoo Finance didn't return sector data for that "
        "ticker — often a temporary fetch issue rather than a real gap."
    )
    bar_fig = px.bar(
        sector_df,
        x="Amount",
        y="Sector",
        orientation="h",
        text=sector_df.apply(lambda r: f"${r['Amount']:,.2f} ({r['Percentage']:.1f}%)", axis=1),
    )
    bar_fig.update_traces(marker_color="#636EFA", textposition="outside")
    bar_fig.update_layout(margin=dict(t=10, b=10, l=10, r=120), xaxis_title="Market Value ($)", yaxis_title=None)
    bar_fig.update_yaxes(autorange="reversed")
    style_fig(bar_fig)
    st.plotly_chart(bar_fig, use_container_width=True)

    sector_display = sector_df.copy()
    sector_display["Amount"] = sector_display["Amount"].map(money)
    sector_display["Percentage"] = sector_display["Percentage"].map(lambda v: f"{v:.2f}%")
    st.dataframe(sector_display, use_container_width=True, hide_index=True)

    st.subheader("Holdings")
    display_df = holdings_df.copy()
    for col in ["Avg Entry", "Current Price", "Market Value", "Unrealized P&L", "Realized P&L", "Dividends"]:
        display_df[col] = display_df[col].map(lambda v: money(v) if pd.notna(v) else "—")
    display_df["Unrealized %"] = holdings_df["Unrealized %"].map(lambda v: pct(v) if pd.notna(v) else "—")
    display_df["Quantity"] = holdings_df["Quantity"].map(qty)

    table_event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="holdings_table",
    )
    clicked = holdings_df.iloc[table_event.selection.rows[0]]["Ticker"] if table_event.selection.rows else None
    if clicked and clicked != st.session_state.get("_last_table_ticker"):
        st.session_state["selected_ticker"] = clicked
    st.session_state["_last_table_ticker"] = clicked

    if closed_positions:
        with st.expander(f"Closed positions ({len(closed_positions)})"):
            closed_df = pd.DataFrame(
                [
                    {"Ticker": t, "Realized P&L": money(p.realized_pnl), "Dividends": money(p.dividends)}
                    for t, p in closed_positions.items()
                ]
            )
            st.dataframe(closed_df, use_container_width=True, hide_index=True)

    stock_tickers = sorted(set(open_positions) - open_bond_tickers)
    if len(stock_tickers) >= 2:
        st.subheader("Stock Correlation")
        st.caption(
            "Pairwise correlation of daily returns over the past year, for open stock/ETF positions only "
            "(bonds excluded — they don't trade with a comparable daily price series)."
        )
        stock_returns = {}
        for ticker in stock_tickers:
            hist = get_price_history(ticker, period="1y")
            if not hist.empty:
                daily_returns = hist.set_index("Date")["Close"].pct_change().dropna()
                if not daily_returns.empty:
                    stock_returns[ticker] = daily_returns
        if len(stock_returns) >= 2:
            corr_matrix = pd.DataFrame(stock_returns).corr()
            corr_fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                aspect="auto",
            )
            corr_fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), coloraxis_colorbar_title="")
            style_fig(corr_fig)
            st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.info("Not enough price history to compute correlations yet.")

with tab_news:
    now_utc = pd.Timestamp.now(tz="UTC")
    today = pd.Timestamp.now().normalize()

    st.subheader("🌍 Economy Overview")
    st.caption(
        "Current policy rates and each central bank's next scheduled meeting, plus the most recent macro "
        "headlines for the US/Fed, Europe, and Asia. Rates refresh hourly; headlines refresh every 15 minutes."
    )

    fed_rate = get_fed_funds_target_range()
    ecb_rate = get_ecb_deposit_rate()
    next_fomc = next_meeting(FOMC_MEETINGS, today)
    next_ecb = next_meeting(ECB_MEETINGS, today)

    def _meeting_label(meeting: tuple[pd.Timestamp, pd.Timestamp] | None) -> tuple[str, str | None, str]:
        # Short value (fits st.metric's big-number display without
        # truncating) + a longer string for the help tooltip.
        if meeting is None:
            return "Not on file", None, "No meeting date on file — check the source link below."
        start, end = meeting
        short = f"{start.strftime('%b %d')}–{end.strftime('%d')}"
        full = (
            f"{start.strftime('%b %d')}–{end.strftime('%d, %Y')}"
            if start.month == end.month
            else f"{start.strftime('%b %d')} – {end.strftime('%b %d, %Y')}"
        )
        days = (start - today).days
        delta = f"in {days}d" if days >= 0 else "underway / just concluded"
        return short, delta, full

    m1, m2, m3, m4 = st.columns([1.3, 1, 1, 1])
    m1.metric(
        "Fed Funds Target Range",
        fed_rate["display"].replace(" ", "") if fed_rate else "n/a",
        help=f"As of {fed_rate['as_of'].date()}." if fed_rate else None,
    )
    fomc_short, fomc_delta, fomc_full = _meeting_label(next_fomc)
    m2.metric("Next FOMC Meeting", fomc_short, fomc_delta, help=fomc_full)
    m3.metric(
        "ECB Deposit Rate",
        ecb_rate["display"] if ecb_rate else "n/a",
        help=f"As of {ecb_rate['as_of'].date()}." if ecb_rate else None,
    )
    ecb_short, ecb_delta, ecb_full = _meeting_label(next_ecb)
    m4.metric("Next ECB Meeting", ecb_short, ecb_delta, help=ecb_full)

    fed_odds = get_fed_meeting_probabilities(next_fomc, fed_rate)
    if fed_odds:
        avg = fed_odds["average"]
        st.markdown(f"**Fed Rate Decision Odds — {fomc_short} Meeting**")
        odds_fig = go.Figure()
        for name, value, color in [
            ("Cut 25bp+", avg["cut"], "#636EFA"),
            ("Hold", avg["hold"], "#9AA0A6"),
            ("Hike 25bp+", avg["hike"], "#EF553B"),
        ]:
            odds_fig.add_trace(
                go.Bar(
                    y=["Odds"],
                    x=[value],
                    name=name,
                    orientation="h",
                    marker_color=color,
                    text=f"{value:.0f}%" if value >= 6 else "",
                    textposition="inside",
                    insidetextfont=dict(color="white"),
                    hovertemplate=f"{name}: %{{x:.1f}}%<extra></extra>",
                )
            )
        odds_fig.update_layout(
            barmode="stack",
            height=90,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
            xaxis=dict(visible=False, range=[0, 100]),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(style_fig(odds_fig), use_container_width=True, config={"displayModeBar": False})
        per_source = " · ".join(
            f"[{s['source_name']}]({s['source_url']}) — cut {s['cut']:.0f}% / hold {s['hold']:.0f}% / hike {s['hike']:.0f}%"
            for s in fed_odds["sources"]
        )
        st.caption(
            f"Average of {len(fed_odds['sources'])} live source(s) — CME's own Fed Funds futures (what CME's "
            f"FedWatch tool and Bloomberg's Fed-o-meter are themselves built on) plus real-money prediction "
            f"markets, no paid feed required: {per_source}."
        )
    else:
        st.caption("Fed rate-move odds unavailable right now — prediction-market sources didn't respond.")

    source_bits = []
    if fed_rate:
        source_bits.append(f"[{fed_rate['source_name']}]({fed_rate['source_url']}) — Fed funds rate, as of {fed_rate['as_of'].date()}")
    if ecb_rate:
        source_bits.append(f"[{ecb_rate['source_name']}]({ecb_rate['source_url']}) — ECB deposit rate, as of {ecb_rate['as_of'].date()}")
    source_bits.append(f"[federalreserve.gov]({FOMC_SOURCE_URL}) — FOMC meeting calendar")
    source_bits.append(f"[ecb.europa.eu]({ECB_SOURCE_URL}) — ECB meeting calendar")
    st.caption("Sources: " + " · ".join(source_bits))

    st.divider()
    # Yahoo's search endpoint matches best on a short 2-3 word phrase (see
    # search_news_topics's docstring), so each region merges a couple of
    # short queries rather than one long, more "precise" one.
    region_queries = {
        "🇺🇸 United States / Fed": ["Federal Reserve", "Fed interest rate"],
        "🇪🇺 Europe": ["European Central Bank", "Eurozone economy"],
        "🌏 Asia": ["China economy", "Bank of Japan"],
    }
    region_cols = st.columns(3)
    for col, (region_label, queries) in zip(region_cols, region_queries.items()):
        with col:
            st.markdown(f"**{region_label}**")
            headlines = search_news_topics(queries, count_per_query=5, limit=5)
            if not headlines:
                st.caption("No headlines available right now.")
            for h in headlines:
                st.markdown(f"[{h['title']}]({h['url']})")
                st.caption(f"Source: {h['source']} · {time_ago(h['published'], now_utc)}")

    st.divider()
    st.subheader("📁 Your Portfolio News")
    st.caption(
        "Recent headlines for each open position in your uploaded portfolio, pulled from Yahoo Finance's "
        "per-security news feed — this section is generic and adapts automatically to whatever's in your CSV."
    )

    news_tickers = sorted(open_positions.keys())
    if not news_tickers:
        st.info("No open positions to show news for.")
    else:

        def _ticker_label(t: str) -> str:
            name = company_names.get(t, t)
            return f"{t} — {name}" if name != t else t

        ALL_HOLDINGS = "All holdings"
        choice = st.selectbox(
            "Filter by holding",
            [ALL_HOLDINGS] + news_tickers,
            format_func=lambda t: t if t == ALL_HOLDINGS else _ticker_label(t),
        )
        tickers_to_show = news_tickers if choice == ALL_HOLDINGS else [choice]
        per_ticker_count = 3 if choice == ALL_HOLDINGS else 8

        found_any = False
        for ticker in tickers_to_show:
            if ticker in open_bond_tickers:
                st.markdown(f"**{_ticker_label(ticker)}**")
                st.caption(
                    "No news feed available for bonds — Yahoo Finance doesn't index bond CUSIPs/ISINs. "
                    "Try the Economy Overview above, or search the issuer/benchmark directly."
                )
                continue
            headlines = get_ticker_news(ticker, count=per_ticker_count)
            if not headlines:
                continue
            found_any = True
            st.markdown(f"**{_ticker_label(ticker)}**")
            for h in headlines:
                st.markdown(f"[{h['title']}]({h['url']})")
                st.caption(f"Source: {h['source']} · {time_ago(h['published'], now_utc)}")

        if not found_any and not all(t in open_bond_tickers for t in tickers_to_show):
            st.info("No recent news found for the selected holding(s).")

with tab_detail:
    all_tickers = sorted(set(positions.keys()) - set(bond_positions))
    if st.session_state.get("selected_ticker") not in all_tickers:
        st.session_state["selected_ticker"] = all_tickers[0] if all_tickers else None

    # Bound to the same session_state key that the Overview tab's treemap/table
    # click handlers write to, so a click there drives this selectbox too.
    selected = st.selectbox("Stock", all_tickers, key="selected_ticker")

    if selected:
        pos = positions[selected]
        current_price = live_prices.get(selected, float("nan"))
        market_value = pos.quantity * current_price if pos.is_open and pd.notna(current_price) else float("nan")
        unrealized = market_value - pos.cost_basis if pd.notna(market_value) else float("nan")
        unrealized_pct = (unrealized / pos.cost_basis * 100) if pos.cost_basis > 0 and pd.notna(unrealized) else float("nan")
        realized_pct = (pos.realized_pnl / pos.cost_basis_sold * 100) if pos.cost_basis_sold > 0 else float("nan")

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Quantity Held", qty(pos.quantity))
        d2.metric("Avg Entry Price", money(pos.avg_price) if pos.is_open else "—")
        d3.metric("Current Price", money(current_price) if pd.notna(current_price) else "—")
        d4.metric("Unrealized P&L", money(unrealized) if pd.notna(unrealized) else "—", pct(unrealized_pct) if pd.notna(unrealized_pct) else None)
        d5.metric("Realized P&L", money(pos.realized_pnl), pct(realized_pct) if pd.notna(realized_pct) else None)
        st.caption(f"Dividends received: {money(pos.dividends)}")

        unrealized_pct_total = (unrealized / pos.total_invested * 100) if pos.total_invested > 0 and pd.notna(unrealized) else float("nan")
        realized_pct_total = (pos.realized_pnl / pos.total_invested * 100) if pos.total_invested > 0 else float("nan")

        st.subheader("Capital invested")
        st.caption("Unrealized/Realized % here are both against total capital ever invested in this stock.")
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Total Invested (all-time)", money(pos.total_invested))
        i2.metric("Currently Invested", money(pos.cost_basis))
        i3.metric("Unrealized P&L", money(unrealized) if pd.notna(unrealized) else "—", pct(unrealized_pct_total) if pd.notna(unrealized_pct_total) else None)
        i4.metric("Realized P&L", money(pos.realized_pnl), pct(realized_pct_total) if pd.notna(realized_pct_total) else None)

        st.subheader(f"Beta vs {BENCHMARK_NAME}")
        beta_stock_hist = get_price_history(selected, period="1y")
        beta_benchmark_hist = get_price_history(BENCHMARK_TICKER, period="1y")
        if not beta_stock_hist.empty and not beta_benchmark_hist.empty:
            stock_returns = beta_stock_hist.set_index(beta_stock_hist["Date"].dt.normalize())["Close"].pct_change().dropna()
            market_returns = beta_benchmark_hist.set_index(beta_benchmark_hist["Date"].dt.normalize())["Close"].pct_change().dropna()
            beta, alpha, r_squared = compute_beta(stock_returns, market_returns)

            if pd.notna(beta):
                b1, b2, b3 = st.columns(3)
                b1.metric("Beta", f"{beta:.2f}")
                b2.metric("Alpha (daily)", f"{alpha * 100:.3f}%")
                b3.metric("R²", f"{r_squared:.2f}")
                st.caption(
                    f"OLS regression of {selected}'s daily returns on {BENCHMARK_NAME}'s daily returns over the "
                    "trailing year — the standard way beta is estimated. Beta above 1 means the stock has "
                    "historically swung more than the market; below 1, less. R² shows how much of that daily "
                    "movement the market actually explains."
                )

                aligned = pd.concat([market_returns.rename("Market"), stock_returns.rename("Stock")], axis=1, join="inner").dropna()
                x_line = [aligned["Market"].min(), aligned["Market"].max()]
                y_line = [alpha + beta * x for x in x_line]

                scatter_fig = go.Figure()
                scatter_fig.add_trace(
                    go.Scatter(
                        x=aligned["Market"],
                        y=aligned["Stock"],
                        mode="markers",
                        name="Daily returns",
                        marker=dict(color="#636EFA", size=5, opacity=0.6),
                    )
                )
                scatter_fig.add_trace(
                    go.Scatter(x=x_line, y=y_line, mode="lines", name=f"Fit (β={beta:.2f})", line=dict(color="#EF553B"))
                )
                scatter_fig.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(orientation="h"),
                    xaxis_title=f"{BENCHMARK_NAME} daily return",
                    yaxis_title=f"{selected} daily return",
                    xaxis_tickformat=".1%",
                    yaxis_tickformat=".1%",
                )
                style_fig(scatter_fig)
                st.plotly_chart(scatter_fig, use_container_width=True)
            else:
                st.info("Not enough overlapping price history to compute beta.")
        else:
            st.info("Not enough price history to compute beta.")

        st.subheader(f"{selected} vs {BENCHMARK_NAME} since your first trade")
        if not pos.trades.empty:
            invest_start = pos.trades["date"].min().normalize()
            perf_stock_hist = get_price_history(selected, start=invest_start.strftime("%Y-%m-%d"))
            perf_benchmark_hist = get_price_history(BENCHMARK_TICKER, start=invest_start.strftime("%Y-%m-%d"))
            if not perf_stock_hist.empty and not perf_benchmark_hist.empty:
                stock_index = price_return_index(perf_stock_hist.set_index(perf_stock_hist["Date"].dt.normalize())["Close"])
                bench_index = price_return_index(perf_benchmark_hist.set_index(perf_benchmark_hist["Date"].dt.normalize())["Close"])

                perf_fig = go.Figure()
                perf_fig.add_trace(go.Scatter(x=stock_index.index, y=stock_index, name=selected, line=dict(color="#636EFA")))
                perf_fig.add_trace(
                    go.Scatter(x=bench_index.index, y=bench_index, name=BENCHMARK_NAME, line=dict(color="#9AA0A6", dash="dash"))
                )
                perf_fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h"), yaxis_title="Growth of 100")
                style_fig(perf_fig)
                st.caption(
                    f"Price only (excludes dividends), indexed to 100 on {invest_start.date()} — your first trade "
                    f"in {selected}. Assumes a single buy-and-hold from that date, so it won't reflect the exact "
                    "return of positions built up over several trades."
                )
                st.plotly_chart(perf_fig, use_container_width=True)

                sc1, sc2 = st.columns(2)
                sc1.metric(selected, pct(stock_index.dropna().iloc[-1] - 100))
                sc2.metric(BENCHMARK_NAME, pct(bench_index.dropna().iloc[-1] - 100))
            else:
                st.info("Not enough price history for this comparison.")

        st.subheader(f"{selected} price history with your trades")
        history = get_price_history(selected)
        if not history.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history["Date"], y=history["Close"], mode="lines", name="Close price", line=dict(color="#636EFA")))
            buys = pos.trades[pos.trades["type"].isin(BUY_TYPES)] if not pos.trades.empty else pd.DataFrame()
            sells = pos.trades[pos.trades["type"].isin(SELL_TYPES)].copy() if not pos.trades.empty else pd.DataFrame()
            if not sells.empty:
                # Bond redemptions carry no per-share price in the export;
                # derive one from the payout so the marker still plots.
                sells["price_usd"] = sells["price_usd"].fillna(sells["amount_usd"] / sells["quantity"])
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys["date"], y=buys["price_usd"], mode="markers", name="Buy",
                    marker=dict(symbol="line-ns-open", size=22, color="#2ecc71", line=dict(width=3)),
                    hovertemplate="Buy<br>%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
                ))
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells["date"], y=sells["price_usd"], mode="markers", name="Sell",
                    marker=dict(symbol="line-ns-open", size=22, color="#e74c3c", line=dict(width=3)),
                    hovertemplate="Sell<br>%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
                ))
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h"))
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price history available for this ticker.")

        st.subheader("Trade history")
        if not pos.trades.empty:
            trades_display = pos.trades.copy()
            trades_display["date"] = trades_display["date"].dt.strftime("%Y-%m-%d %H:%M")
            trades_display["price_usd"] = trades_display["price_usd"].map(lambda v: money(v) if pd.notna(v) else "—")
            trades_display["amount_usd"] = trades_display["amount_usd"].map(money)
            trades_display["quantity"] = trades_display["quantity"].map(lambda v: qty(v) if pd.notna(v) else "—")
            st.dataframe(
                trades_display[["date", "type", "quantity", "price_usd", "amount_usd"]].rename(
                    columns={"date": "Date", "type": "Type", "quantity": "Quantity", "price_usd": "Price", "amount_usd": "Amount"}
                ),
                use_container_width=True,
                hide_index=True,
            )

if tab_bonds is not None:
    with tab_bonds:
        bond_tickers = sorted(bond_positions)
        selected_bond = st.selectbox("Bond", bond_tickers, key="selected_bond")

        if selected_bond:
            bond_pos = bond_positions[selected_bond]
            cusip = isin_to_us_cusip(selected_bond)
            treasury = get_treasury_security(cusip) if cusip else None
            econ = estimate_bond_economics(bond_pos)

            status = "Open" if bond_pos.is_open else "Redeemed"
            bd1, bd2, bd3, bd4 = st.columns(4)
            bd1.metric("Status", status)
            bd2.metric("Quantity Held", qty(bond_pos.quantity) if bond_pos.is_open else "—")
            bd3.metric("Coupon Income", money(bond_pos.dividends))
            bd4.metric("Realized P&L", money(bond_pos.realized_pnl))

            st.subheader("Bond characteristics")
            if treasury is not None:
                st.caption(
                    f"Real terms from the U.S. Treasury's public Fiscal Data API, looked up by CUSIP "
                    f"({treasury.cusip}) extracted from the ISIN Revolut exports."
                )
                face_value = 100.0
                coupon_rate = treasury.coupon_rate
                payments_per_year = treasury.payments_per_year
                maturity_date = treasury.maturity_date
                yield_proxy = treasury.yield_at_auction or coupon_rate

                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Type", f"{treasury.security_type} ({treasury.security_term})")
                t2.metric("Maturity", maturity_date.strftime("%Y-%m-%d"))
                t3.metric("Coupon Rate", f"{coupon_rate:.3f}%")
                t4.metric("Payment Frequency", "None (zero-coupon)" if payments_per_year == 0 else f"{payments_per_year}x / yr")
                st.caption(f"Issued {treasury.issue_date.strftime('%Y-%m-%d')} · Auction high yield {treasury.yield_at_auction:.3f}%" if treasury.yield_at_auction else f"Issued {treasury.issue_date.strftime('%Y-%m-%d')}")

                market_price = price_from_yield_curve(face_value, coupon_rate, payments_per_year, maturity_date)
                p1, p2 = st.columns(2)
                if market_price is not None:
                    p1.metric("Est. Market Price", f"{market_price:.2f}", f"{market_price - face_value:+.2f} vs. par")
                else:
                    p1.metric("Est. Market Price", "n/a")
                p2.metric("Face Value (Par)", f"{face_value:.2f}")
                st.caption(
                    "Market price is estimated by discounting the bond's remaining cash flows at today's actual "
                    "Treasury par yield curve — not a live per-CUSIP quote, since no free source offers one."
                )
            else:
                st.caption(
                    "No Treasury record found (not a US Treasury security, or it's a corporate/foreign bond). "
                    "These figures are estimated from the coupon payments in your own transaction history, not "
                    "an official source — treat them as approximate."
                )
                face_value = econ.face_value
                coupon_rate = econ.coupon_rate
                payments_per_year = econ.payments_per_year
                maturity_date = econ.maturity_date
                yield_proxy = coupon_rate

                e1, e2, e3 = st.columns(3)
                e1.metric("Face Value (assumed)", money(face_value))
                e2.metric(
                    "Est. Coupon Rate",
                    f"{coupon_rate:.3f}%" if coupon_rate is not None else "Not enough data",
                )
                e3.metric(
                    "Maturity",
                    maturity_date.strftime("%Y-%m-%d") + " (redeemed)" if maturity_date is not None else "Unknown — still open",
                )
                if coupon_rate is None:
                    st.info(
                        "Need at least two coupon payments to estimate a payment frequency and annualize the "
                        "rate — this bond doesn't have enough history yet."
                    )

            can_compute = (
                coupon_rate is not None
                and payments_per_year is not None
                and payments_per_year > 0
                and maturity_date is not None
                and bond_pos.is_open
            )
            if can_compute:
                macaulay, modified = compute_duration(face_value, coupon_rate, payments_per_year, maturity_date, yield_proxy)
                if macaulay is not None:
                    st.subheader("Duration")
                    st.caption(
                        "Based on the security's original terms and its issue-time yield, not a live market "
                        "price — there's no free live pricing source for individual bond CUSIPs, so this won't "
                        "capture real interest-rate moves since issuance."
                    )
                    d1, d2 = st.columns(2)
                    d1.metric("Macaulay Duration", f"{macaulay:.2f} yrs")
                    d2.metric("Modified Duration", f"{modified:.2f} yrs")

            if coupon_rate is not None and payments_per_year and payments_per_year > 0 and maturity_date is not None:
                issue_date = treasury.issue_date if treasury is not None else bond_pos.trades["date"].min().normalize()
                schedule = build_cash_flow_schedule(face_value, coupon_rate, payments_per_year, issue_date, maturity_date)
                if not schedule.empty:
                    st.subheader("Cash flow schedule")
                    today = pd.Timestamp.now(tz=schedule["date"].dt.tz) if schedule["date"].dt.tz is not None else pd.Timestamp.now()
                    schedule["status"] = schedule["date"].le(today).map({True: "Paid", False: "Projected"})

                    cash_fig = go.Figure()
                    for label, color in [("Paid", "#22c55e"), ("Projected", "#9AA0A6")]:
                        subset = schedule[schedule["status"] == label]
                        if not subset.empty:
                            cash_fig.add_trace(
                                go.Bar(x=subset["date"], y=subset["amount"], name=label, marker_color=color)
                            )
                    cash_fig.update_layout(
                        margin=dict(t=10, b=10, l=10, r=10),
                        legend=dict(orientation="h"),
                        yaxis_title="Cash flow per unit ($)",
                        barmode="overlay",
                    )
                    style_fig(cash_fig)
                    st.plotly_chart(cash_fig, use_container_width=True)

                    schedule_display = schedule.copy()
                    schedule_display["date"] = schedule_display["date"].dt.strftime("%Y-%m-%d")
                    schedule_display["amount"] = schedule_display["amount"].map(money)
                    st.dataframe(
                        schedule_display.rename(columns={"date": "Date", "amount": "Per Unit", "type": "Type", "status": "Status"}),
                        use_container_width=True,
                        hide_index=True,
                    )

            st.subheader("Trade history")
            if not bond_pos.trades.empty:
                bond_trades_display = bond_pos.trades.copy()
                bond_trades_display["date"] = bond_trades_display["date"].dt.strftime("%Y-%m-%d %H:%M")
                bond_trades_display["price_usd"] = bond_trades_display["price_usd"].map(lambda v: money(v) if pd.notna(v) else "—")
                bond_trades_display["amount_usd"] = bond_trades_display["amount_usd"].map(money)
                bond_trades_display["quantity"] = bond_trades_display["quantity"].map(lambda v: qty(v) if pd.notna(v) else "—")
                st.dataframe(
                    bond_trades_display[["date", "type", "quantity", "price_usd", "amount_usd"]].rename(
                        columns={"date": "Date", "type": "Type", "quantity": "Quantity", "price_usd": "Price", "amount_usd": "Amount"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

with tab_transactions:
    st.subheader("Full transaction log")
    type_filter = st.multiselect("Filter by type", sorted(transactions["type"].unique()))
    ticker_filter = st.multiselect("Filter by ticker", sorted(transactions["ticker"].dropna().unique()))

    log = transactions.copy()
    if type_filter:
        log = log[log["type"].isin(type_filter)]
    if ticker_filter:
        log = log[log["ticker"].isin(ticker_filter)]

    log_display = log.copy()
    log_display["date"] = log_display["date"].dt.strftime("%Y-%m-%d %H:%M")
    log_display["price"] = log_display["price"].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")
    log_display["amount"] = log_display["amount"].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")
    log_display["amount_usd"] = log_display["amount_usd"].map(money)
    log_display["quantity"] = log_display["quantity"].map(lambda v: qty(v) if pd.notna(v) else "—")
    st.caption(
        "Price/Amount are as recorded, in the trade's own currency; Amount (USD) is that value converted at the "
        "day's actual exchange rate — the figure used everywhere else in this app."
    )
    st.dataframe(
        log_display[["date", "ticker", "type", "quantity", "price", "amount", "currency", "amount_usd"]].rename(
            columns={
                "date": "Date", "ticker": "Ticker", "type": "Type", "quantity": "Quantity",
                "price": "Price", "amount": "Amount", "currency": "Currency", "amount_usd": "Amount (USD)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
