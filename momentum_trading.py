import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Load the data
file_path = "D:\\AlgoTrade\\Fyers_AlgoTrade\\Nifty50_Index\\NIFTY50_INDEX_1_Min.csv" #'NIFTY50_INDEX_1_Min.csv'  # Update with the correct path to your file
# Load the data
#file_path = 'NIFTY50_INDEX_1_Min.csv'  # Update with the correct path to your file
data = pd.read_csv(file_path, parse_dates=['Date'])

# Ensure the data is sorted by timestamp
data.sort_values('Date', inplace=True)

# Calculate Momentum Indicators
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_stochastic(data, k_window=14, d_window=3):
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    stoch_k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=d_window).mean()
    return stoch_k, stoch_d

def generate_signals(data, rsi_threshold=30, stoch_k_threshold=20):
    buy_signals = []
    sell_signals = []
    position = 0  # 1 indicates holding a position, 0 indicates no position
    
    for i in range(len(data)):
        buy_signal = (data['RSI'].iloc[i] < rsi_threshold and
                      data['MACD'].iloc[i] > data['MACD_Signal'].iloc[i] and
                      data['Stochastic_%K'].iloc[i] < stoch_k_threshold)
        
        sell_signal = (data['RSI'].iloc[i] > (100 - rsi_threshold) and
                       data['MACD'].iloc[i] < data['MACD_Signal'].iloc[i] and
                       data['Stochastic_%K'].iloc[i] > (100 - stoch_k_threshold))
        
        if buy_signal and position == 0:
            buy_signals.append(data['Close'].iloc[i])
            sell_signals.append(np.nan)
            position = 1
        elif sell_signal and position == 1:
            buy_signals.append(np.nan)
            sell_signals.append(data['Close'].iloc[i])
            position = 0
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
    
    return buy_signals, sell_signals

def calculate_profit(data, buy_signals, sell_signals):
    profit = 0
    for buy_price, sell_price in zip(buy_signals, sell_signals):
        if not np.isnan(buy_price) and not np.isnan(sell_price):
            profit += sell_price - buy_price
    return profit

def optimize_parameters(data):
    rsi_windows = [10, 14, 20]
    macd_short_windows = [10, 12, 15]
    macd_long_windows = [20, 26, 30]
    macd_signal_windows = [7, 9, 11]
    stoch_k_windows = [10, 14, 20]
    stoch_d_windows = [3, 5, 7]
    rsi_thresholds = [20, 30, 40]
    stoch_k_thresholds = [10, 20, 30]

    best_profit = -np.inf
    best_params = None
    
    for rsi_window, macd_short, macd_long, macd_signal, stoch_k, stoch_d, rsi_th, stoch_th in product(
            rsi_windows, macd_short_windows, macd_long_windows, macd_signal_windows, stoch_k_windows, stoch_d_windows, rsi_thresholds, stoch_k_thresholds):
        
        data['RSI'] = calculate_rsi(data, window=rsi_window)
        data['MACD'], data['MACD_Signal'] = calculate_macd(data, short_window=macd_short, long_window=macd_long, signal_window=macd_signal)
        data['Stochastic_%K'], data['Stochastic_%D'] = calculate_stochastic(data, k_window=stoch_k, d_window=stoch_d)
        
        buy_signals, sell_signals = generate_signals(data, rsi_threshold=rsi_th, stoch_k_threshold=stoch_th)
        profit = calculate_profit(data, buy_signals, sell_signals)
        
        if profit > best_profit:
            best_profit = profit
            best_params = (rsi_window, macd_short, macd_long, macd_signal, stoch_k, stoch_d, rsi_th, stoch_th)
    
    return best_params, best_profit

# Optimize parameters
best_params, best_profit = optimize_parameters(data)

print(f'Best Parameters: RSI Window: {best_params[0]}, MACD Short Window: {best_params[1]}, MACD Long Window: {best_params[2]}, MACD Signal Window: {best_params[3]}, Stochastic %K Window: {best_params[4]}, Stochastic %D Window: {best_params[5]}, RSI Threshold: {best_params[6]}, Stochastic %K Threshold: {best_params[7]}')
print(f'Best Profit: {best_profit}')

# Calculate indicators with best parameters
data['RSI'] = calculate_rsi(data, window=best_params[0])
data['MACD'], data['MACD_Signal'] = calculate_macd(data, short_window=best_params[1], long_window=best_params[2], signal_window=best_params[3])
data['Stochastic_%K'], data['Stochastic_%D'] = calculate_stochastic(data, k_window=best_params[4], d_window=best_params[5])

# Generate signals with best parameters
data['Buy_Signal_Price'], data['Sell_Signal_Price'] = generate_signals(data, rsi_threshold=best_params[6], stoch_k_threshold=best_params[7])

# Plot the results
plt.figure(figsize=(14,8))
plt.plot(data['Date'], data['Close'], label='Close Price', alpha=0.5)
plt.scatter(data['Date'], data['Buy_Signal_Price'], label='Buy Signal', marker='^', color='green', alpha=1)
plt.scatter(data['Date'], data['Sell_Signal_Price'], label='Sell Signal', marker='v', color='red', alpha=1)
plt.title('Stock Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(loc='upper left')
plt.show()
