import pandas as pd
import numpy as np
import os

# Load data
# file_path = r"D:\AlgoTrade\Fyers_AlgoTrade\Fyers_Data\HA.csv"
file_path = r"D:\AlgoTrade\Fyers_AlgoTrade\Fyers_Data\BankNifty_Index\NIFTYBANK_INDEX_1_Min.csv"
data = pd.read_csv(file_path)
# print(data.head(10))

data.rename(columns={'Date': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
data.drop(columns=["Unnamed: 0", "Volume"], inplace=True)
# print(data.head(10))

start_datetime = '2023-01-01 09:15:00'
data = data[data['datetime'] >= start_datetime]
# print(data.head(10))

data['datetime'] = pd.to_datetime(data['datetime'])

# Set the start and end time for filtering
start_time = pd.to_datetime('09:15:00').time()
end_time = pd.to_datetime('15:25:00').time()
data = data[data['datetime'].dt.time.between(start_time, end_time)]
data.set_index('datetime', inplace=True)

# print(data.head(10))

# User input for signal and order timeframes
signal_timeframe = '5min'
order_timeframe = '5min'

# Resample data to the signal timeframe
signal_data = data.resample(signal_timeframe, origin='start').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    # 'HA_Open': 'first',
    # 'HA_High': 'max'
}).dropna()

# print(signal_data.head(10))

# Initialize variables
initial_cash = 100000
cash = initial_cash
position = 0
trade_qty = 100
brokerage = 100
trades = []
entry_price = 0
candle_range = 100
candle_body = 10
sl_tolerance = 3
stoploss_pts = 30

current_consecutive_wins = 0
current_consecutive_losses = 0
max_consecutive_wins = 0
max_consecutive_losses = 0

daily_trade_count = 0
current_date = None

# Dictionary to store signals
signals = {}

# Generate signals based on the signal timeframe
for i in range(2, len(signal_data)):
    current = signal_data.iloc[i]
    prev = signal_data.iloc[i - 1]
    pprev = signal_data.iloc[i - 2]

    if (
        pprev['open'] > pprev['close']
        and (pprev['open'] - pprev['close']) >= candle_body
        and prev['open'] > prev['close']
        and (prev['open'] - prev['close']) >= candle_body
        and prev['close'] < pprev['close']
        and prev['high'] < pprev['high']
        and prev['low'] < pprev['low']
        and (pprev['high'] - prev['low']) >= candle_range
        and current['low'] <= prev['low'] - sl_tolerance
        # and current['HA_Open'] == current['HA_High']
    ):
        signal_time = current.name
        signals[signal_time] = {
            'entry_price': prev['low'] - sl_tolerance,
            'stoploss_price1': prev['low'] - sl_tolerance + stoploss_pts + sl_tolerance * 2.0,
            'stoploss_price2': round((prev['low'] - (0.236 * (prev['low'] - prev['high']))), 2)
        }

# Resample data to the order timeframe
order_data = data.resample(order_timeframe).agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
}).dropna()

# Process the order timeframe data to place orders
for i in range(len(order_data)):
    current = order_data.iloc[i]
    current_time = current.name

    # Check if there is an active signal
    for signal_time, signal_values in signals.items():
        if signal_time <= current_time < signal_time + pd.Timedelta(signal_timeframe):
            if position == 0:
                entry_price = signal_values['entry_price']
                stoploss_price = min(signal_values['stoploss_price1'], signal_values['stoploss_price2'])
                position = 1
                trades.append(('sell', current_time, round(entry_price, 1), 0.0, cash))
            else:
                if (
                    current['close'] > current['open'] and current['high'] >= stoploss_price
                    or current['close'] >= stoploss_price and current['close'] < current['open']
                ):
                    cash += (entry_price - stoploss_price) * trade_qty
                    cash -= brokerage
                    trades.append(('loss', current_time, stoploss_price, entry_price - stoploss_price, cash))
                    current_consecutive_losses += 1
                    current_consecutive_wins = 0
                else:
                    val = round(entry_price - current['close'], 2)
                    cash += val * trade_qty
                    cash -= brokerage
                    if entry_price > current['close']:
                        trades.append(('profit', current_time, round(entry_price - val, 2), val, cash))
                        current_consecutive_wins += 1
                        current_consecutive_losses = 0
                    else:
                        trades.append(('loss', current_time, round(entry_price - val, 2), val, cash))
                        current_consecutive_losses += 1
                        current_consecutive_wins = 0

                max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)

                position = 0
                entry_price = 0
                stop_loss = 0

                daily_trade_count += 1

# Calculate performance metrics
total_trades = int(len(trades) / 2)
win_trades = len([trade for trade in trades if trade[0] == 'profit'])
lose_trades = len([trade for trade in trades if trade[0] == 'loss'])

profit_win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
loss_win_rate = (lose_trades / total_trades) * 100 if total_trades > 0 else 0

net_profit = cash - initial_cash

# Calculate drawdowns
portfolio_values = [initial_cash]
for trade in trades:
    if trade[0] in ['profit', 'loss']:
        portfolio_values.append(trade[-1])
max_drawdown_value = 0
max_drawdown_percentage = 0
peak = portfolio_values[0]
for value in portfolio_values:
    if value > peak:
        peak = value
    drawdown_value = peak - value
    drawdown_percentage = (drawdown_value / peak) * 100
    if drawdown_value > max_drawdown_value:
        max_drawdown_value = drawdown_value
    if drawdown_percentage > max_drawdown_percentage:
        max_drawdown_percentage = drawdown_percentage

# Calculate trade returns for Sharpe and Sortino ratios
returns = []
for trade in trades:
    if trade[0] in ['profit', 'loss']:
        returns.append((trade[-1] - initial_cash) / initial_cash)

# Annualize returns for Sharpe and Sortino ratios
mean_return = np.mean(returns)
std_return = np.std(returns)
risk_free_rate = 0.05
sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return != 0 else 0

# Sortino ratio (only downside deviation)
negative_returns = [r for r in returns if r < 0]
downside_std_return = np.std(negative_returns)
sortino_ratio = (mean_return - risk_free_rate) / downside_std_return if downside_std_return != 0 else 0

# Profit factor
gross_profit = sum([trade[3] * trade_qty for trade in trades if trade[0] == 'profit'])
gross_loss = sum([trade[3] * trade_qty for trade in trades if trade[0] == 'loss'])
profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')

# Calculate win/loss statistics
win_amounts = [trade[3] * trade_qty for trade in trades if trade[0] == 'profit']
lose_amounts = [trade[3] * trade_qty for trade in trades if trade[0] == 'loss']

largest_win = max(win_amounts) if win_amounts else 0
largest_loss = min(lose_amounts) if lose_amounts else 0

avg_win = np.mean(win_amounts) if win_amounts else 0
avg_loss = np.mean(lose_amounts) if lose_amounts else 0

ratio_avg_win_avg_loss = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')

# Commission paid
commission_paid = total_trades * brokerage

# Print trading measures
print("========================================")
print(f'Started From          : {data.iloc[0].name}')
print(f'Candle Body           : {candle_body}')
print(f'Start Portfolio Value : {initial_cash}')
print(f'End Portfolio Value   : {cash}')
print(f'Net Profit            : {net_profit}')
print("----------------------------------------")
print(f'Profit Trades         : {win_trades}')
print(f'Profit Rate           : {profit_win_rate:.2f}%')
print(f'Loss Trades           : {lose_trades}')
print(f'Loss Rate             : {loss_win_rate:.2f}%')
print("----------------------------------------")
print(f'Total Trades          : {total_trades}')
print(f'Total Win Rate        : {profit_win_rate:.2f}%')
print(f'Total Loss Rate       : {loss_win_rate:.2f}%')
print("----------------------------------------")
print(f'Max Drawdown Value    : {max_drawdown_value:.2f}')
print(f'Max Drawdown          : {max_drawdown_percentage:.2f}%')
print("----------------------------------------")
print(f'Largest Win           : {largest_win}')
print(f'Largest Loss          : {largest_loss}')
print(f'Average Win           : {avg_win}')
print(f'Average Loss          : {avg_loss}')
print(f'Win/Loss Ratio        : {ratio_avg_win_avg_loss:.2f}')
print("----------------------------------------")
print(f'Profit Factor         : {profit_factor:.2f}')
print(f'Sharpe Ratio          : {sharpe_ratio:.2f}')
print(f'Sortino Ratio         : {sortino_ratio:.2f}')
print(f'Commission Paid       : {commission_paid}')
print("========================================")
