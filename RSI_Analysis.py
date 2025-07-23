import pandas as pd
import numpy as np

# Load the uploaded Excel file
file_path = "D:\\AlgoTrade\\Fyers_AlgoTrade\\EMA_initial_df.xlsx"
df = pd.read_excel(file_path)

# Rename columns for convenience
df.rename(columns={'Date': 'Date', 'Open_Mins': 'Open', 'High_Mins': 'High', 'Low_Mins': 'Low', 'Close_Mins': 'Close'}, inplace=True)

# Function to calculate RSI
def calculate_rsi(df, period=14):
    df = df.copy()
    df['Change'] = df['Close'].diff()
    df['Gain'] = np.where(df['Change'] > 0, df['Change'], 0)
    df['Loss'] = np.where(df['Change'] < 0, -df['Change'], 0)
    df['Avg Gain'] = df['Gain'].rolling(window=period, min_periods=1).mean()
    df['Avg Loss'] = df['Loss'].rolling(window=period, min_periods=1).mean()
    df['RS'] = df['Avg Gain'] / df['Avg Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    return df['RSI']

# Function to backtest the RSI strategy
def backtest_rsi_strategy(df, rsi_period, lower_limit, upper_limit):
    df = df.copy()
    df['RSI'] = calculate_rsi(df, period=rsi_period)
    
    # Define the signals
    df['Buy Signal'] = np.where(df['RSI'] < lower_limit, 1, 0)
    df['Sell Signal'] = np.where(df['RSI'] > upper_limit, -1, 0)
    
    # Generate trading positions
    df['Position'] = df['Buy Signal'] + df['Sell Signal']
    df['Position'] = df['Position'].replace(to_replace=0, method='ffill') # Carry forward the last position
    df['Position'] = df['Position'].shift(1) # Shift position to avoid lookahead bias
    
    # Calculate returns
    df['Market Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Market Return'] * df['Position']
    
    # Calculate cumulative returns
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod() - 1
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod() - 1
    
    return df['Cumulative Strategy Return'].iloc[-1]

# Optimize RSI parameters
def optimize_rsi_parameters(df, rsi_periods, lower_limits, upper_limits):
    best_params = None
    best_return = -np.inf
    for period in rsi_periods:
        for lower in lower_limits:
            for upper in upper_limits:
                if lower >= upper:
                    continue
                cum_return = backtest_rsi_strategy(df, period, lower, upper)
                if cum_return > best_return:
                    best_return = cum_return
                    best_params = (period, lower, upper)
    return best_params, best_return

# Define parameter ranges for optimization
rsi_periods = range(5, 30, 2)
lower_limits = range(10, 50, 2)
upper_limits = range(50, 90, 2)

# Perform optimization
best_params, best_return = optimize_rsi_parameters(df, rsi_periods, lower_limits, upper_limits)

print(best_params, best_return)
