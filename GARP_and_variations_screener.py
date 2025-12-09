# garp_screener_v2.py
# Requires: pip install yfinance pandas numpy
import yfinance as yf
import pandas as pd
import numpy as np
import time

# -----------------------------------------
# Utility helpers
# -----------------------------------------
def safe_div(a, b):
    try:
        if a is None or b is None or b == 0:
            return None
        return a / b
    except Exception:
        return None

def cagr(start, end, periods):
    try:
        if start is None or end is None or start <= 0 or periods <= 0:
            return None
        return (end / start) ** (1.0 / periods) - 1.0
    except Exception:
        return None

# -----------------------------------------
# Earnings and fundamentals
# -----------------------------------------
def get_net_income_history(ticker_obj):
    """
    Returns list of Net Income values from the income statement (most recent to oldest).
    Compatible with yfinance>=0.2.50
    """
    try:
        df = ticker_obj.income_stmt
        if df is None or df.empty or "Net Income" not in df.index:
            return None
        net_series = df.loc["Net Income"].dropna().astype(float)
        # Reverse to chronological order (oldest → newest)
        return list(net_series[::-1])
    except Exception:
        return None

def earnings_cagr_from_list(values, years=3):
    if not values or len(values) < 2:
        return None
    n = min(len(values), years + 1)
    vals = values[-n:]
    start, end = vals[0], vals[-1]
    return cagr(start, end, len(vals) - 1)

def get_financial_metrics(ticker):
    """
    Fetch basic valuation and quality metrics for a ticker.
    """
    try:
        obj = yf.Ticker(ticker)
        info = obj.info or {}

        pe = info.get("trailingPE") or info.get("forwardPE")
        market_cap = info.get("marketCap")
        dividend_yield = info.get("dividendYield") or 0.0

        # Get Net Income and Equity for ROE
        bs = obj.balance_sheet
        income_stmt = obj.income_stmt
        total_equity = None
        total_liab = None
        net_income = None

        if bs is not None and not bs.empty:
            if "Total Stockholder Equity" in bs.index:
                total_equity = bs.loc["Total Stockholder Equity"].dropna().iloc[0]
            if "Total Liab" in bs.index:
                total_liab = bs.loc["Total Liab"].dropna().iloc[0]

        if income_stmt is not None and not income_stmt.empty:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna().iloc[0]

        roe = safe_div(net_income, total_equity)
        debt_equity = safe_div(total_liab, total_equity)

        # Earnings CAGR
        earnings_hist = get_net_income_history(obj)
        earnings_cagr = earnings_cagr_from_list(earnings_hist, years=3)

        # PEG ratio
        peg = None
        if pe and earnings_cagr and earnings_cagr > 0:
            peg = pe / (earnings_cagr * 100) if earnings_cagr < 1 else pe / earnings_cagr

        return {
            "ticker": ticker,
            "name": info.get("shortName"),
            "pe": pe,
            "peg": peg,
            "earnings_cagr": earnings_cagr,
            "roe": roe,
            "debt_equity": debt_equity,
            "market_cap": market_cap,
            "dividend_yield": dividend_yield,
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return {"ticker": ticker}

# -----------------------------------------
# Screeners (core + variations)
# -----------------------------------------
def apply_garp(df):
    cond = (
        df["peg"].notnull() &
        (df["peg"] <= 2.0) &
        df["earnings_cagr"].notnull() &
        (df["earnings_cagr"] > 0.00)
    )
    return df[cond].sort_values(["peg", "earnings_cagr"], ascending=[True, False])

def apply_quality_garp(df):
    cond = (
        df["peg"].notnull() &
        (df["peg"] <= 1.5) &
        (df["roe"].notnull()) &
        (df["roe"] >= 0.10) &
        (df["debt_equity"].notnull()) &
        (df["debt_equity"] <= 1.0)
    )
    return df[cond].sort_values(["peg", "roe"], ascending=[True, False])

def apply_dividend_garp(df):
    cond = (
        df["peg"].notnull() &
        (df["peg"] <= 1.8) &
        (df["dividend_yield"].notnull()) &
        (df["dividend_yield"] >= 0.02)
    )
    return df[cond].sort_values(["dividend_yield", "peg"], ascending=[False, True])

# -----------------------------------------
# Runner
# -----------------------------------------
def screen_stocks(tickers, debug=False):
    data = []
    for i, t in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Fetching {t}...")
        row = get_financial_metrics(t)
        data.append(row)
        time.sleep(0.1)  # polite throttling

    df = pd.DataFrame(data).set_index("ticker")

    if debug:
        print("\n--- Raw Metrics ---")
        print(df[["pe", "earnings_cagr", "peg", "roe", "debt_equity"]].head(10))

    print("\nSaved raw results to garp_output.csv")
    df.to_csv("garp_output.csv")

    print("\n=== GARP Results ===")
    print(apply_garp(df).head(10).to_string())

    print("\n=== Quality GARP Results ===")
    print(apply_quality_garp(df).head(10).to_string())

    print("\n=== Dividend GARP Results ===")
    print(apply_dividend_garp(df).head(10).to_string())

# -----------------------------------------
# Example run
# -----------------------------------------
if __name__ == "__main__":
    sample_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "META", "PG", "KO", "PEP"]
    screen_stocks(sample_tickers, debug=True)
