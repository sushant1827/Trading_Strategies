import pandas as pd
import numpy as np
import talib
import os


spot_file = 'NIFTYBANK_INDEX_60_Min.csv'
spot_path = 'D:/AlgoTrade/Fyers_AlgoTrade/Fyers_Data/BankNifty_Index'

data = pd.read_csv(os.path.join(spot_path, spot_file))
data = data.iloc[-4649:-42]
# spot_df = spot_df.iloc[-308:-42]
data.drop(columns=['Unnamed: 0', 'Volume'], inplace=True)
data.reset_index(inplace=True, drop=True)

# print(data.head())
# print(data.tail())

# Parameters for optimization
initial_cash = 100000  # Initial investment
best_rsi_period = None
best_rsi_band = None
best_roi = -np.inf
best_results = None

# Define the grid search range for RSI periods and RSI band values
rsi_period_values = range(3, 28, 3)  # RSI period range (5 to 30)
rsi_band_values = range(39, 61, 3)   # RSI band range for buy condition (60 to 90)

# Function to simulate buy strategy
def simulate_buy_strategy(rsi_period, rsi_band):
    # Calculate RSI for the given period
    data['RSI'] = talib.RSI(data['Close'], timeperiod=rsi_period)
    
    cash = initial_cash
    position = 0  # 1 for long, 0 for no position
    entry_price = 0
    results = []

    for index, row in data.iterrows():
        rsi = row['RSI']
        close = row['Close']

        # Buy condition
        if position == 0 and rsi > rsi_band:
            position = 1
            entry_price = close
            results.append({'Date': row['Date'], 'Action': 'Buy', 'Price': close, 'Cash': cash, 'Position': position})

        # Close Buy position
        elif position == 1 and rsi < rsi_band:
            profit = (close - entry_price) * 1  # Adjust the multiplier based on lot size if needed
            cash += profit
            position = 0
            results.append({'Date': row['Date'], 'Action': 'Close Buy', 'Price': close, 'Cash': cash, 'Position': position})

    # Calculate final ROI
    final_roi = (cash - initial_cash) / initial_cash
    return final_roi, results

# Perform grid search over RSI periods and RSI band values
for rsi_period in rsi_period_values:
    for rsi_band in rsi_band_values:
        roi, results = simulate_buy_strategy(rsi_period, rsi_band)
        if roi > best_roi:
            best_roi = roi
            best_rsi_period = rsi_period
            best_rsi_band = rsi_band
            best_results = results

# Output the best parameters and ROI
print(f"Best RSI Period: {best_rsi_period}, Best RSI Band: {best_rsi_band}, Best ROI: {best_roi:.2%}")
print("\nDetailed Results:")
for result in best_results:
    print(result)

# Optional: Save best results to a CSV
# pd.DataFrame(best_results).to_csv('best_buy_strategy_results.csv', index=False)
