# ------------------------------------------------------------------------------------
#                            FabTrader Algo Trading Academy
# ------------------------------------------------------------------------------------
# Copyright (c) 2025 FabTrader
# CONTACT:
# - Website: https://fabtrader.in
# - Email: hello@fabtrader.in
# Usage: Personal use only. Not for commercial redistribution.
# Permissions and licensing inquiries should be directed to the contact email.

"""
Bottom-Out-Hunting (BOH) Filter
-------------------------------

Author: FabTrader
Website: https://fabtrader.in

Description:
-------------
This script implements the Bottom-Out-Hunting (BOH) filter — a simple yet effective idea
used by swing and algo traders to identify potential recovery stocks.

The logic is straightforward:
- If the stock's 52-week LOW occurred more recently than its 52-week HIGH,
  it indicates the stock might have bottomed out and could be in a recovery phase.
- If the 52-week HIGH is more recent, the stock may still be in a downward or topping phase.

The program fetches 1 year of daily price data from Yahoo Finance and evaluates
whether the BOH condition is True or False for any given stock ticker.

Usage:
------
1. Install dependencies:
   pip install yfinance

2. Run the script after inputting in a stock short code
"""
import yfinance as yf
from datetime import datetime, timedelta


def bottom_out_hunting(ticker: str):
    """
    Determines if a stock meets the Bottom-Out-Hunting (BOH) condition.

    Condition:
    True if the stock's 52-week low is more recent than its 52-week high.
    False otherwise.

    Args:
        ticker (str): Stock ticker symbol (e.g., "RELIANCE.NS", "AAPL")

    Returns:
        dict: {
            'Ticker': str,
            '52_Week_High': float,
            '52_Week_High_Date': str,
            '52_Week_Low': float,
            '52_Week_Low_Date': str,
            'BOH_Filter': bool
        }
    """

    # Fetch 1 year (52 weeks) of daily data
    end_date = datetime.today()
    start_date = end_date - timedelta(weeks=52)

    try:
        data = yf.download(
            ticker + '.NS', start=start_date, end=end_date,
            progress=False, auto_adjust=True,
            multi_level_index=False
                                    )
        if data.empty:
            return {"Error": f"No data found for {ticker}"}

        # Calculate 52-week high and low
        high_price = data['High'].max()
        low_price = data['Low'].min()

        # Get corresponding dates
        high_date = data['High'].idxmax().date()
        low_date = data['Low'].idxmin().date()

        # Apply the BOH logic
        boh_filter = low_date > high_date

        return {
            'Ticker': ticker,
            '52_Week_High': round(high_price, 2),
            '52_Week_High_Date': str(high_date),
            '52_Week_Low': round(low_price, 2),
            '52_Week_Low_Date': str(low_date),
            'BOH_Filter': boh_filter
        }
    except Exception as e:
        print("Error : Data extract failed. Error :", e)
        return None

if __name__ == "__main__":
    ticker = "INFY"
    result = bottom_out_hunting(ticker)
    if result is not None:
        print("***** Bottom Out Hunting (BOH) Filter *****")
        print("Ticker : ", result['Ticker'])
        print("52 Week High : ", result['52_Week_High'])
        print("52 Week High Date : ", result['52_Week_High_Date'])
        print("52 Week Low : ", result['52_Week_Low'])
        print("52 Week Low Date : ", result['52_Week_Low_Date'])
        print("BOH Filter Flag : ", result['BOH_Filter'])
        if result['BOH_Filter']:
            print("Filter Verdict : Suitable for a swing trade")
        else:
            print("Filter Verdict : Not Suitable for a swing trade")