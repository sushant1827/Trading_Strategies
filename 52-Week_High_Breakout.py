# -----------------------------------------------

# Python implementation of the 52-Week High Breakout Strategy for Nifty 50 stocks using daily candle data:

# Strategy Overview

#   Universe: Nifty 50 stocks (daily data)

#   Entry Condition:

#       Stock makes new 52-week high

#       Confirmatory filters:

#           Close above 10-day SMA

#           SMA alignment: SMA(10) > SMA(20) > SMA(50) > SMA(100)

#   Exit:

#       Use a trailing stop: 20-day SMA

#       Or exit on fixed % stop loss (optional)

# -----------------------------------------------

import pandas as pd
import numpy as np


def calculate_sma(df, windows=[10, 20, 50, 100]):
    for window in windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
    return df

def find_52_week_high_breakouts(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date'])

    results = []
    grouped = df.groupby('Symbol')

    for symbol, data in grouped:
        data = calculate_sma(data.copy())
        data['52w_high'] = data['High'].rolling(252).max()

        in_position = False
        entry_price = sl_price = entry_date = None

        for i in range(100, len(data)):  # Ensure SMA_100 is available
            row = data.iloc[i]
            prev_row = data.iloc[i - 1]

            # Entry conditions
            if not in_position:
                is_new_52w_high = row['Close'] > prev_row['52w_high']
                sma_alignment = row['Close'] > row['SMA_10'] and \
                                row['SMA_10'] > row['SMA_20'] > row['SMA_50'] > row['SMA_100']

                if is_new_52w_high and sma_alignment:
                    in_position = True
                    entry_price = row['Close']
                    entry_date = row['Date']
                    sl_price = row['SMA_20']
            else:
                # Exit conditions
                if row['Close'] < row['SMA_20']:
                    exit_price = row['Close']
                    exit_date = row['Date']
                    pnl = exit_price - entry_price
                    results.append({
                        'Symbol': symbol,
                        'Entry Date': entry_date,
                        'Entry Price': entry_price,
                        'Exit Date': exit_date,
                        'Exit Price': exit_price,
                        'PnL': pnl
                    })
                    in_position = False

        # Handle open position at the end
        if in_position:
            last_row = data.iloc[-1]
            pnl = last_row['Close'] - entry_price
            results.append({
                'Symbol': symbol,
                'Entry Date': entry_date,
                'Entry Price': entry_price,
                'Exit Date': last_row['Date'],
                'Exit Price': last_row['Close'],
                'PnL': pnl,
                'Note': 'Open Position'
            })

    return pd.DataFrame(results)


# -----------------------------------------------
# Usage Example
# -----------------------------------------------

df = pd.read_csv('nifty50_daily.csv')  # Must include 'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume'
breakouts = find_52_week_high_breakouts(df)
print(breakouts)

# -----------------------------------------------
