import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------

def calculate_supertrend(df, period=10, multiplier=3.0):
    # Calculate ATR
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(window=period).mean()

    # Calculate basic upper and lower bands
    df['Upper Basic'] = (df['High'] + df['Low']) / 2 - (multiplier * df['ATR'])
    df['Lower Basic'] = (df['High'] + df['Low']) / 2 + (multiplier * df['ATR'])

    # Initialize Supertrend bands
    df['Upper Band'] = 0.0
    df['Lower Band'] = 0.0

    # Calculate Upper Band and Lower Band
    for i in range(1, len(df)):
        if df['Close'][i - 1] > df['Upper Band'][i - 1]:
            df['Upper Band'][i] = max(df['Upper Basic'][i], df['Upper Band'][i - 1])
        else:
            df['Upper Band'][i] = df['Upper Basic'][i]

        if df['Close'][i - 1] < df['Lower Band'][i - 1]:
            df['Lower Band'][i] = min(df['Lower Basic'][i], df['Lower Band'][i - 1])
        else:
            df['Lower Band'][i] = df['Lower Basic'][i]

    # Initialize trend
    df['Supertrend'] = 0

    # Calculate Supertrend and signals
    for i in range(1, len(df)):
        if df['Close'][i] > df['Lower Band'][i - 1]:
            df['Supertrend'][i] = 1
        elif df['Close'][i] < df['Upper Band'][i - 1]:
            df['Supertrend'][i] = -1
        else:
            df['Supertrend'][i] = df['Supertrend'][i - 1]

    # Clean up the DataFrame
    df.drop(columns=['TR', 'Upper Basic', 'Lower Basic', 'Upper Band', 'Lower Band'], inplace=True)
    
    return df

# -----------------------------------------------------------------------------------

df = pd.read_csv('./BankNifty_Index/NIFTYBANK_INDEX_5_Min.csv')
df.drop(columns=['Unnamed: 0', 'Volume'], inplace=True)
df = df.tail(111500)
df.reset_index(drop=True, inplace=True)

df = calculate_supertrend(df, period=10, multiplier=3.0)
df.dropna(inplace=True)

df.to_csv("TV_SuperTrend_5Mins.csv")

# -----------------------------------------------------------------------------------