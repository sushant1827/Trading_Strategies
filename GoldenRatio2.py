# Golden Ratio with Day Data and Minute Data
# ---------------------------------------------

import pandas as pd
import numpy as np

# Define the Fibonacci levels and Golden Ratio
fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
golden_ratio = 0.618

daily_data = pd.read_csv(r'D:\AlgoTrade\Fyers_AlgoTrade\BankNifty_Index\NIFTYBANK_INDEX_D_Min.csv')

# Calculate the daily high and low for the previous day

daily_high = daily_data['High'][-2]
daily_low = daily_data['Low'][-2]

# Calculate the price range for the previous day
daily_range = daily_high - daily_low

# Calculate the Fibonacci retracement levels for the previous day's range
fib_values = [(daily_low + level * daily_range) for level in fib_levels]

# Calculate the buy and sell prices based on the Golden Ratio
buy_price = fib_values[3] + golden_ratio * (fib_values[2] - fib_values[3])
sell_price = fib_values[3] + golden_ratio * (fib_values[4] - fib_values[3])

# Define a function to apply the Golden Ratio strategy to the minute data
def golden_ratio_strategy(minute_data, buy_price, sell_price):
    # Calculate the minute high and low for the current minute
    minute_high = minute_data['High'][-1]
    minute_low = minute_data['Low'][-1]
    
    # Check if the price crosses the buy or sell level
    if minute_high > buy_price:
        return 'Buy'
    elif minute_low < sell_price:
        return 'Sell'
    else:
        return 'Hold'

minute_data = pd.read_csv(r'D:\AlgoTrade\Fyers_AlgoTrade\BankNifty_Index\NIFTYBANK_INDEX_1_Min.csv')

# Apply the Golden Ratio strategy to the minute data
minute_data['Signal'] = minute_data.apply(golden_ratio_strategy, args=(buy_price, sell_price), axis=1)

# Calculate the overall returns and other financial metrics
total_return = (minute_data['Close'][-1] - minute_data['Close'][0]) / minute_data['Close'][0]
sharpe_ratio = np.sqrt(252) * (minute_data['Close'].pct_change().mean() / minute_data['Close'].pct_change().std())
sortino_ratio = np.sqrt(252) * (minute_data['Close'].pct_change().mean() / minute_data[minute_data['Close'].pct_change() < 0]['Close'].pct_change().std())
drawdown = (minute_data['Close'] - minute_data['Close'].cummax()).min()
winrate = (minute_data['Signal'].value_counts()['Buy'] / len(minute_data))

# Print the results
print('Total return:', total_return)
print('Sharpe ratio:', sharpe_ratio)
print('Sortino ratio:', sortino_ratio)
print('Max drawdown:', drawdown)
print('Win rate:', winrate)
