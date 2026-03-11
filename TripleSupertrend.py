import pandas as pd
import numpy as np

# Load OHLC data from CSV file
df = pd.read_csv('NIFTY50_INDEX_15_Min.csv')

# Define function to calculate Heikin Ashi candles
def heikin_ashi(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (ha_close.shift(1) + ha_close) / 2
    ha_high = df[['High', 'Open', 'Close']].max(axis=1)
    ha_low = df[['Low', 'Open', 'Close']].min(axis=1)
    return pd.DataFrame({'Open': ha_open, 'High': ha_high, 'Low': ha_low, 'Close': ha_close})

# Define function to calculate SuperTrend
def supertrend(df, atr_length, factor):
    df['ATR'] = df['High'] - df['Low']
    df['ATR'] = df['ATR'].rolling(atr_length).mean()
    df['Upperband'] = ((df['High'] + df['Low']) / 2) + (factor * df['ATR'])
    df['Lowerband'] = ((df['High'] + df['Low']) / 2) - (factor * df['ATR'])
    df['Upperband'] = df['Upperband'].shift(1).fillna(0) if df['Upperband'].iloc[0] == 0 else df['Upperband']
    df['Lowerband'] = df['Lowerband'].shift(1).fillna(0) if df['Lowerband'].iloc[0] == 0 else df['Lowerband']
    df['Trend'] = np.nan

    for i in range(atr_length, len(df)):
        if df['Close'][i] <= df['Upperband'][i-1]:
            df['Trend'][i] = df['Upperband'][i]
        elif df['Close'][i] > df['Upperband'][i-1]:
            df['Trend'][i] = df['Lowerband'][i]
        else:
            df['Trend'][i] = 0

    return df['Trend']

# Calculate Heikin Ashi candles
ha_df = heikin_ashi(df)

# Calculate SuperTrend with atr length 21 and factor 1
df['SuperTrend21_1'] = supertrend(ha_df, atr_length=21, factor=1)

# Calculate SuperTrend with atr length 14 and factor 2
df['SuperTrend14_2'] = supertrend(ha_df, atr_length=14, factor=2)

# Calculate SuperTrend with atr length 7 and factor 3
df['SuperTrend7_3'] = supertrend(ha_df, atr_length=7, factor=3)

# Combine all SuperTrend values into one dataframe
# supertrend_df = pd.DataFrame({'Supertrend 1': supertrend_1,
#                               'Supertrend 2': supertrend_2,
#                               'Supertrend 3': supertrend_3})

# Calculate signals
signals = pd.DataFrame(index=ha_df.index)
signals['Close'] = ha_df['Close']

# Buy signal when Close is above all SuperTrends
signals['Buy'] = (signals['Close'] > supertrend_df['Supertrend 1']) & \
                 (signals['Close'] > supertrend_df['Supertrend 2']) & \
                 (signals['Close'] > supertrend_df['Supertrend 3'])

# Sell signal
signals['Buy'] = (signals['Close'] > supertrend_df['Supertrend 1']) & \
                 (signals['Close'] > supertrend_df['Supertrend 2']) & \
                 (signals['Close'] > supertrend_df['Supertrend 3'])