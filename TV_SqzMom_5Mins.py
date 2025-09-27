import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------

def linreg(source, length, offset=0):
    def linear_regression(y):
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        return intercept + slope * (length - 1 - offset)
    
    return source.rolling(window=length).apply(linear_regression, raw=False)

# -----------------------------------------------------------------------------------

def squeeze_momentum_indicator(df, length, mult, lengthKC, multKC, useTrueRange):
    # Calculate BB
    df['basis'] = df['Close'].rolling(window=length).mean()
    df['dev'] = df['Close'].rolling(window=length).std() * mult
    df['upperBB'] = df['basis'] + df['dev']
    df['lowerBB'] = df['basis'] - df['dev']

    # Calculate KC (Keltner Channel)
    df['ma'] = df['Close'].rolling(window=lengthKC).mean()
    # df['ema'] = df['Close'].ewm(span=lengthKC, adjust=False).mean()

    if useTrueRange:
        df['tr'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())))
    else:
        df['tr'] = df['High'] - df['Low']

    df['rangema'] = df['tr'].rolling(window=lengthKC).mean()
    df['upperKC'] = df['ma'] + df['rangema'] * multKC
    df['lowerKC'] = df['ma'] - df['rangema'] * multKC
    # df['upperKC'] = df['ema'] + df['rangema'] * multKC
    # df['lowerKC'] = df['ema'] - df['rangema'] * multKC

    # Squeeze Conditions
    df['sqzOn']   = (df['lowerBB'] > df['lowerKC']) & (df['upperBB'] < df['upperKC']).astype(int)
    df['sqzOff']  = (df['lowerBB'] < df['lowerKC']) & (df['upperBB'] > df['upperKC']).astype(int)
    df['noSqz']   = (~df['sqzOn'] & ~df['sqzOff']).astype(int)

    # Linear Regression Value
    # df['linreg_val'] = df['Close'] - (
    #     (df['High'].rolling(window=lengthKC).max() + df['Low'].rolling(window=lengthKC).min()) / 2 + 
    #     df['Close'].rolling(window=lengthKC).mean()
    # ).rolling(window=lengthKC).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0])

    highest_high = df['High'].rolling(window=lengthKC).max()
    lowest_low = df['Low'].rolling(window=lengthKC).min()

    # Calculate the average of the highest high and lowest low
    avg_high_low = (highest_high + lowest_low) / 2

    # Calculate the SMA of the close prices over the lengthKC period
    sma_close = df['Close'].rolling(window=lengthKC).mean()

    # Calculate the average of avg_high_low and sma_close
    avg_val = (avg_high_low + sma_close) / 2

    # Calculate the difference between the source and avg_val
    source = df['Close']  # Replace 'Close' with the specific source you're using
    diff = source - avg_val

    # Apply the linear regression on the diff series
    df['linreg_val'] = linreg(diff, lengthKC, offset=0)

    # Color conditions
    df['val_prev'] = df['linreg_val'].shift(1)
    df['lime'] = ((df['linreg_val'] > 0) & (df['linreg_val'] > df['val_prev'])).astype(int)    
    df['green'] = ((df['linreg_val'] > 0) & (df['linreg_val'] <= df['val_prev'])).astype(int)
    df['red'] = ((df['linreg_val'] < 0) & (df['linreg_val'] < df['val_prev'])).astype(int)
    df['maroon'] = ((df['linreg_val'] < 0) & (df['linreg_val'] >= df['val_prev'])).astype(int)
    
    # Squeeze colors
    df['blue'] = df['noSqz'].astype(int)
    df['black'] = df['sqzOn'].astype(int)
    df['gray'] = (~df['noSqz'] & ~df['sqzOn']).astype(int)

    df['SqzMoment'] = 0

    # Calculate SqzMoment signals
    for i in range(len(df)):
        if df['gray'][i] == 1 and df['lime'][i] == 1:
            df['SqzMoment'][i] = 1
        elif df['gray'][i] == 1 and df['red'][i] == 1:
            df['SqzMoment'][i] = -1
        else:
            df['SqzMoment'][i] = 0

    return df[['Date','Open','High','Low','Close','linreg_val','lime','green','red','maroon','black','gray','SqzMoment']]

# -----------------------------------------------------------------------------------

df = pd.read_csv('./BankNifty_Index/NIFTYBANK_INDEX_30_Min.csv')
df.drop(columns=['Unnamed: 0', 'Volume'], inplace=True)
df = df.tail(111500)
df.reset_index(drop=True, inplace=True)
# print(df.head())

df = squeeze_momentum_indicator(df, length=20, mult=1.5, lengthKC=20, multKC=1.5, useTrueRange=True)
df.dropna(inplace=True)

df.to_csv("TV_SqzMom_5Mins.csv")

# -----------------------------------------------------------------------------------

