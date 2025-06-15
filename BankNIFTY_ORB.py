# -----------------------------------------------

# BankNIFTY Opening Range Breakout strategy using 15-minute OHLC data. 

# The strategy assumes:

# Entry after 9:30 AM, based on the first 15-minute candle (9:15–9:30).

# Long if price closes above the high of the first candle.

# Short if price closes below the low of the first candle.

# Stop Loss is the opposite end of the first candle.

# Exit at 3:15 PM or at SL hit.

# -----------------------------------------------

import pandas as pd

def apply_opening_range_breakout_strategy(df):
    # Ensure Datetime is datetime type
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    # Filter data only for one trading day (or use groupby if running on multiple days)
    trading_date = df.index[0].date()
    day_df = df[df.index.date == trading_date]

    # Get first 15-min candle (9:15–9:30 AM)
    opening_range_candle = day_df.between_time("09:15", "09:30").iloc[0]
    open_time = opening_range_candle.name
    opening_high = opening_range_candle['High']
    opening_low = opening_range_candle['Low']

    # Define SLs and breakout conditions
    in_position = False
    trade = {}
    
    for time, row in day_df.loc[open_time:].iterrows():
        if not in_position:
            if row['Close'] > opening_high:
                # Bullish breakout
                in_position = True
                trade = {
                    'Entry Time': time,
                    'Position': 'Long',
                    'Entry Price': row['Close'],
                    'SL': opening_low
                }
            elif row['Close'] < opening_low:
                # Bearish breakout
                in_position = True
                trade = {
                    'Entry Time': time,
                    'Position': 'Short',
                    'Entry Price': row['Close'],
                    'SL': opening_high
                }
        else:
            # Monitor SL and Exit at 3:15 PM
            if trade['Position'] == 'Long' and row['Low'] <= trade['SL']:
                trade['Exit Time'] = time
                trade['Exit Price'] = trade['SL']
                trade['Exit Reason'] = 'Stop Loss Hit'
                break
            elif trade['Position'] == 'Short' and row['High'] >= trade['SL']:
                trade['Exit Time'] = time
                trade['Exit Price'] = trade['SL']
                trade['Exit Reason'] = 'Stop Loss Hit'
                break
            elif time.time() >= pd.to_datetime("15:15").time():
                trade['Exit Time'] = time
                trade['Exit Price'] = row['Close']
                trade['Exit Reason'] = 'Time Exit'
                break

    if not trade:
        return "No Trade Triggered"

    trade['PnL'] = (trade['Exit Price'] - trade['Entry Price']) * (1 if trade['Position'] == 'Long' else -1)
    return trade


# -----------------------------------------------
# Usage
# -----------------------------------------------


df = pd.read_csv('banknifty_15min.csv')  # Your 15-min OHLC data with 'Datetime', 'Open', 'High', 'Low', 'Close'
result = apply_opening_range_breakout_strategy(df)
print(result)


# -----------------------------------------------
# Output Sample
# -----------------------------------------------


{
 'Entry Time': Timestamp('2025-06-14 09:45:00'),
 'Position': 'Long',
 'Entry Price': 48000.0,
 'SL': 47720.0,
 'Exit Time': Timestamp('2025-06-14 15:15:00'),
 'Exit Price': 48350.0,
 'Exit Reason': 'Time Exit',
 'PnL': 350.0
}


# -----------------------------------------------
