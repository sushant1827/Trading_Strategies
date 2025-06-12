import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your OHLC data (ensure 'Date' is the index and 'Close' is the closing price)
data = pd.read_csv('nifty_ohlc_data.csv', parse_dates=['Date'], index_col='Date')

# Calculate 50-day and 200-day SMAs
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()

# Initialize columns for signals and positions
data['Signal'] = 0
data['Signal'][50:] = np.where(data['SMA50'][50:] > data['SMA200'][50:], 1, 0)
data['Position'] = data['Signal'].diff()

# Calculate daily returns
data['Daily Return'] = data['Close'].pct_change()

# Filter out the first row as it will have NaN values
data = data.dropna()

# Initialize portfolio value
initial_capital = 100000
data['Portfolio Value'] = initial_capital

# Simulate the strategy
for i in range(1, len(data)):
    if data['Position'][i] == 1:  # Buy signal
        data['Portfolio Value'][i] = data['Portfolio Value'][i-1] * (1 + data['Daily Return'][i])
    elif data['Position'][i] == -1:  # Sell signal
        data['Portfolio Value'][i] = data['Portfolio Value'][i-1] * (1 - data['Daily Return'][i])
    else:
        data['Portfolio Value'][i] = data['Portfolio Value'][i-1] * (1 + data['Daily Return'][i])

# Calculate performance metrics
# Annualized Return
total_return = data['Portfolio Value'].iloc[-1] / data['Portfolio Value'].iloc[0] - 1
annualized_return = (1 + total_return) ** (252 / len(data)) - 1

# Maximum Drawdown
data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
data['Cumulative Max'] = data['Cumulative Return'].cummax()
data['Drawdown'] = data['Cumulative Return'] / data['Cumulative Max'] - 1
max_drawdown = data['Drawdown'].min()

# Sharpe Ratio
risk_free_rate = 0.05  # 5% annual risk-free rate
excess_daily_return = data['Daily Return'] - (risk_free_rate / 252)
sharpe_ratio = np.mean(excess_daily_return) / np.std(excess_daily_return) * np.sqrt(252)

# Win Rate
winning_trades = len(data[data['Daily Return'] > 0])
total_trades = len(data)
win_rate = winning_trades / total_trades

# Profit Factor
gross_profit = data[data['Daily Return'] > 0]['Daily Return'].sum()
gross_loss = data[data['Daily Return'] < 0]['Daily Return'].sum()
profit_factor = gross_profit / abs(gross_loss)

# Display performance metrics
print(f"Annualized Return: {annualized_return * 100:.2f}%")
print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Win Rate: {win_rate * 100:.2f}%")
print(f"Profit Factor: {profit_factor:.2f}")

# Plotting the portfolio value over time
plt.figure(figsize=(10, 6))
plt.plot(data['Portfolio Value'], label='Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.show()
