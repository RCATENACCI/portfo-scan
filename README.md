# portfoscan

An interactive local dashboard for a Revolut investing portfolio. Revolut has
no personal API for account/portfolio data, so this works off the CSV
transaction export from the Revolut app, and pulls live market prices
separately from Yahoo Finance via `yfinance`.

## Setup

```bash
conda env create -f environment-dev.yml
conda activate portfoscan
```

(`requirements.txt` is what Streamlit Community Cloud installs from for the
deployed app; `environment-dev.yml` is only for local conda setup — it's
named that way, instead of `environment.yml`, so Streamlit Cloud doesn't
pick it up too and warn about having two requirements files.)

## Get your data

In the Revolut app: **Invest → Statements → Export → CSV**. Save the file as
`data/raw/transactions.csv` (or upload it from the dashboard's sidebar instead
— either works, and the file is never committed to git).

## Run

```bash
streamlit run app.py
```

## What it shows

- **Overview** — account value, unrealized/realized P&L, dividends, cash
  balance, an allocation treemap, and a holdings table. Click a stock in the
  treemap or table to jump to its detail view.
- **News** — an economy overview (current Fed/ECB policy rates, each
  central bank's next scheduled meeting, and recent US/Europe/Asia macro
  headlines), plus a portfolio news section with recent headlines for each
  open position — generic, so it adapts to whatever's in the uploaded CSV.
  Every headline is shown with its publisher and how recent it is.
- **Stock Detail** — for one ticker: quantity held, average entry price,
  current price, unrealized/realized P&L, dividends received, a price chart
  with your buy/sell points marked, and the full trade history.
- **Transactions** — the raw transaction log (trades, dividends, cash
  top-ups/withdrawals, promotions), filterable by type and ticker.

## Notes

- Position accounting uses average cost basis (not FIFO/LIFO).
- If a ticker doesn't resolve on Yahoo Finance, add a mapping in
  `portfoscan/prices.py::TICKER_OVERRIDES`.
- Live prices are cached for 5 minutes; use the sidebar's refresh button to
  force an update.
- FOMC/ECB meeting dates in the News tab are a hand-maintained lookup table
  (`portfoscan/news.py::FOMC_MEETINGS` / `ECB_MEETINGS`) — extend it once the
  next year's calendar is published; policy rates themselves are fetched
  live so they don't need updating.
