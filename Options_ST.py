import pandas as pd
import numpy as np

# Define the path to the CSV file
csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_15_Min.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Remove 'Unnamed: 0' and 'Volume' columns
df = df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')

# Rename 'Date' column to 'DateTime'
df = df.rename(columns={'Date': 'DateTime'})

# Set 'DateTime' column as index
df = df.set_index('DateTime')

# Convert index to datetime objects
df.index = pd.to_datetime(df.index)

# Filter DataFrame to include data from 2021 onwards
df = df[df.index.year >= 2021]

print(f"Original DataFrame length: {len(df)}")

# Identify duplicate index entries
duplicate_rows = df[df.index.duplicated(keep='first')]
print(f"Number of duplicate rows found: {len(duplicate_rows)}")

# Drop duplicate index entries, keeping the first one
df = df[~df.index.duplicated(keep='first')]
print(f"DataFrame length after dropping duplicates: {len(df)}")

# =========================================================
# Linear Regression Calculation
# =========================================================

# def linreg(series, length):
#     if len(series) < length:
#         return np.nan
    
#     x = np.arange(length)
#     y = series.iloc[-length:]
    
#     N = length
#     sum_xy = (x * y).sum()
#     sum_x = x.sum()
#     sum_y = y.sum()
#     sum_x2 = (x**2).sum()
    
#     denominator = (N * sum_x2 - sum_x**2)
    
#     if denominator == 0: # Avoid division by zero
#         return np.nan
        
#     m = (N * sum_xy - sum_x * sum_y) / denominator
#     c = (sum_y - m * sum_x) / N
    
#     return m * (length - 1) + c

# linreg_length = 10 # Default value from Pine Script

# df['LinReg_Open'] = df['Open'].rolling(window=linreg_length).apply(lambda x: linreg(x, linreg_length), raw=False)
# df['LinReg_High'] = df['High'].rolling(window=linreg_length).apply(lambda x: linreg(x, linreg_length), raw=False)
# df['LinReg_Low'] = df['Low'].rolling(window=linreg_length).apply(lambda x: linreg(x, linreg_length), raw=False)
# df['LinReg_Close'] = df['Close'].rolling(window=linreg_length).apply(lambda x: linreg(x, linreg_length), raw=False)

# =========================================================
# Heikin Ashi Calculation (based on Pine Script)
# =========================================================

# Calculate Heikin Ashi OHLC
df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

# Initialize HA_Open for the first row
df['HA_Open'] = 0.0
df.loc[df.index[0], 'HA_Open'] = df['Open'].iloc[0]

for i in range(1, len(df)):
    df.loc[df.index[i], 'HA_Open'] = (df['HA_Open'].iloc[i-1] + df['HA_Close'].iloc[i-1]) / 2

df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

# =========================================================
# Smoothed Heikin Ashi Calculation (based on Pine Script)
# =========================================================

# Define lengths for EMA
len_ema1 = 10
len_ema2 = 10

# Calculate EMA for original OHLC
df['o_ema1'] = df['Open'].ewm(span=len_ema1, adjust=False).mean()
df['c_ema1'] = df['Close'].ewm(span=len_ema1, adjust=False).mean()
df['h_ema1'] = df['High'].ewm(span=len_ema1, adjust=False).mean()
df['l_ema1'] = df['Low'].ewm(span=len_ema1, adjust=False).mean()

# Calculate Heikin Ashi Close (haclose)
df['HA_Close_Smoothed'] = (df['o_ema1'] + df['h_ema1'] + df['l_ema1'] + df['c_ema1']) / 4

# Calculate Heikin Ashi Open (haopen) - this is a recursive calculation
# Initialize HA_Open_Smoothed for the first row
df['HA_Open_Smoothed'] = 0.0
df.loc[df.index[0], 'HA_Open_Smoothed'] = (df['o_ema1'].iloc[0] + df['c_ema1'].iloc[0]) / 2

for i in range(1, len(df)):
    df.loc[df.index[i], 'HA_Open_Smoothed'] = (df['HA_Open_Smoothed'].iloc[i-1] + df['HA_Close_Smoothed'].iloc[i-1]) / 2

# Calculate Heikin Ashi High and Low (hahigh, halow)
df['HA_High_Smoothed'] = df[['h_ema1', 'HA_Open_Smoothed', 'HA_Close_Smoothed']].max(axis=1)
df['HA_Low_Smoothed'] = df[['l_ema1', 'HA_Open_Smoothed', 'HA_Close_Smoothed']].min(axis=1)

# Apply second EMA for further smoothing
df['SHA_Open'] = df['HA_Open_Smoothed'].ewm(span=len_ema2, adjust=False).mean()
df['SHA_High'] = df['HA_High_Smoothed'].ewm(span=len_ema2, adjust=False).mean()
df['SHA_Low'] = df['HA_Low_Smoothed'].ewm(span=len_ema2, adjust=False).mean()
df['SHA_Close'] = df['HA_Close_Smoothed'].ewm(span=len_ema2, adjust=False).mean()

# Drop intermediate EMA columns if not needed
df = df.drop(columns=['o_ema1', 'c_ema1', 'h_ema1', 'l_ema1'], errors='ignore')
df = df.drop(columns=['HA_Open_Smoothed', 'HA_Close_Smoothed', 'HA_High_Smoothed', 'HA_Low_Smoothed'], errors='ignore')

# =========================================================
# UT Bot Calculation
# =========================================================

def calculate_ut_bot(df_input, open_col, high_col, low_col, close_col, key_val=1, atr_period=10):
    df = df_input.copy()

    df['src'] = df[close_col]
    df['high_src'] = df[high_col]
    df['low_src'] = df[low_col]

    # ATR Calculation
    df['tr'] = np.maximum(
        df['high_src'] - df['low_src'],
        np.maximum(abs(df['high_src'] - df['src'].shift()), abs(df['low_src'] - df['src'].shift()))
    )
    df['atr'] = df['tr'].rolling(atr_period).mean()
    df['nLoss'] = key_val * df['atr']

    # Initialize trailing stop
    df['xATRTrailingStop'] = np.nan

    for i in range(1, len(df)):
        prev_stop = df.iloc[i-1]['xATRTrailingStop'] if not np.isnan(df.iloc[i-1]['xATRTrailingStop']) else df.iloc[i]['src']
        curr_src = df.iloc[i]['src']
        prev_src = df.iloc[i-1]['src']
        nLoss = df.iloc[i]['nLoss']

        if curr_src > prev_stop and prev_src > prev_stop:
            df.loc[df.index[i], 'xATRTrailingStop'] = max(prev_stop, curr_src - nLoss)
        elif curr_src < prev_stop and prev_src < prev_stop:
            df.loc[df.index[i], 'xATRTrailingStop'] = min(prev_stop, curr_src + nLoss)
        elif curr_src > prev_stop:
            df.loc[df.index[i], 'xATRTrailingStop'] = curr_src - nLoss
        else:
            df.loc[df.index[i], 'xATRTrailingStop'] = curr_src + nLoss

    # Position detection
    df['pos'] = 0
    for i in range(1, len(df)):
        prev_src = df.iloc[i-1]['src']
        curr_src = df.iloc[i]['src']
        prev_stop = df.iloc[i-1]['xATRTrailingStop']
        curr_stop = df.iloc[i]['xATRTrailingStop']
        prev_pos = df.iloc[i-1]['pos']

        if prev_src < prev_stop and curr_src > curr_stop:
            df.loc[df.index[i], 'pos'] = 1
        elif prev_src > prev_stop and curr_src < curr_stop:
            df.loc[df.index[i], 'pos'] = -1
        else:
            df.loc[df.index[i], 'pos'] = prev_pos

    # EMA(1)
    df['ema1'] = df['src'].ewm(span=1, adjust=False).mean()

    # Signals
    df['Buy'] = ((df['src'] > df['xATRTrailingStop']) & 
                 (df['ema1'] > df['xATRTrailingStop']) & 
                 (df['ema1'].shift() <= df['xATRTrailingStop'].shift()))
    df['Sell'] = ((df['src'] < df['xATRTrailingStop']) & 
                  (df['ema1'] < df['xATRTrailingStop']) & 
                  (df['ema1'].shift() >= df['xATRTrailingStop'].shift()))

    # Convert boolean signals to integers (1 for signal, 0 for no signal)
    df['Buy'] = df['Buy'].astype(int)
    df['Sell'] = df['Sell'].astype(int)

    return df[['Buy', 'Sell']]

# =========================================================
# Performance Metrics Calculation
# =========================================================

def calculate_metrics(df_signals, initial_capital=100000, trade_size=1, stop_loss_percent=None, take_profit_percent=None):
    df = df_signals.copy()
    
    trades = []
    current_capital = initial_capital
    
    df['Strategy_Returns'] = 0.0
    
    position = 0 # 0: no position, 1: long, -1: short
    entry_price = 0
    entry_date = None
    stop_loss_price = 0
    take_profit_price = 0
    
    for i in range(1, len(df)):
        current_date = df.index[i]
        current_open = df['Open'].iloc[i] 
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        current_close = df['Close'].iloc[i]

        # Check for take profit hit first if a position is open
        if position == 1 and take_profit_percent is not None: # Long position
            if current_high >= take_profit_price:
                profit_loss = (take_profit_price - entry_price) * trade_size
                current_capital += profit_loss
                trades.append(
                    {
                        "trade_type": "Long_Exit",
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": current_date,
                        "exit_price": take_profit_price,
                        "profit_loss": profit_loss,
                        "trade_size": trade_size,
                        "exit_reason": "TakeProfit",
                        "capital_after_trade": current_capital,
                    }
                )
                position = 0
                entry_price = 0
                entry_date = None
                stop_loss_price = 0
                take_profit_price = 0
                continue # Move to next bar after take profit exit

        elif position == -1 and take_profit_percent is not None: # Short position
            if current_low <= take_profit_price:
                profit_loss = (entry_price - take_profit_price) * trade_size
                current_capital += profit_loss
                trades.append(
                    {
                        "trade_type": "Short_Exit",
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": current_date,
                        "exit_price": take_profit_price,
                        "profit_loss": profit_loss,
                        "trade_size": trade_size,
                        "exit_reason": "TakeProfit",
                        "capital_after_trade": current_capital,
                    }
                )
                position = 0
                entry_price = 0
                entry_date = None
                stop_loss_price = 0
                take_profit_price = 0
                continue # Move to next bar after take profit exit

        # Check for stop loss hit if a position is open (after checking take profit)
        if position == 1 and stop_loss_percent is not None: # Long position
            if current_low <= stop_loss_price:
                profit_loss = (stop_loss_price - entry_price) * trade_size
                current_capital += profit_loss
                trades.append(
                    {
                        "trade_type": "Long_Exit",
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": current_date,
                        "exit_price": stop_loss_price,
                        "profit_loss": profit_loss,
                        "trade_size": trade_size,
                        "exit_reason": "StopLoss",
                        "capital_after_trade": current_capital,
                    }
                )
                position = 0
                entry_price = 0
                entry_date = None
                stop_loss_price = 0
                take_profit_price = 0
                continue # Move to next bar after stop loss exit

        elif position == -1 and stop_loss_percent is not None: # Short position
            if current_high >= stop_loss_price:
                profit_loss = (entry_price - stop_loss_price) * trade_size
                current_capital += profit_loss
                trades.append(
                    {
                        "trade_type": "Short_Exit",
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": current_date,
                        "exit_price": stop_loss_price,
                        "profit_loss": profit_loss,
                        "trade_size": trade_size,
                        "exit_reason": "StopLoss",
                        "capital_after_trade": current_capital,
                    }
                )
                position = 0
                entry_price = 0
                entry_date = None
                stop_loss_price = 0
                take_profit_price = 0
                continue # Move to next bar after stop loss exit

        if df['Buy'].iloc[i] == 1 and position != 1: # Enter long
            if position == -1: # If previously short, close short position
                profit_loss = (entry_price - current_close) * trade_size
                current_capital += profit_loss
                trades.append(
                    {
                        "trade_type": "Short_Exit",
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": current_date,
                        "exit_price": current_close,
                        "profit_loss": profit_loss,
                        "trade_size": trade_size,
                        "exit_reason": "Buy_Signal",
                        "capital_after_trade": current_capital,
                    }
                )
            
            position = 1
            entry_price = current_close
            entry_date = current_date
            if stop_loss_percent is not None:
                stop_loss_price = entry_price * (1 - stop_loss_percent)
            if take_profit_percent is not None:
                take_profit_price = entry_price * (1 + take_profit_percent)
            trades.append(
                {
                    "trade_type": "Long_Entry",
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": np.nan,
                    "exit_price": np.nan,
                    "profit_loss": np.nan,
                    "trade_size": trade_size,
                    "exit_reason": np.nan,
                    "capital_after_trade": current_capital,
                }
            )
        elif df['Sell'].iloc[i] == 1 and position != -1: # Enter short
            if position == 1: # If previously long, close long position
                profit_loss = (current_close - entry_price) * trade_size
                current_capital += profit_loss
                trades.append(
                    {
                        "trade_type": "Long_Exit",
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": current_date,
                        "exit_price": current_close,
                        "profit_loss": profit_loss,
                        "trade_size": trade_size,
                        "exit_reason": "Sell_Signal",
                        "capital_after_trade": current_capital,
                    }
                )
            
            position = -1
            entry_price = current_close
            entry_date = current_date
            if stop_loss_percent is not None:
                stop_loss_price = entry_price * (1 + stop_loss_percent)
            if take_profit_percent is not None:
                take_profit_price = entry_price * (1 - take_profit_percent)
            trades.append(
                {
                    "trade_type": "Short_Entry",
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": np.nan,
                    "exit_price": np.nan,
                    "profit_loss": np.nan,
                    "trade_size": trade_size,
                    "exit_reason": np.nan,
                    "capital_after_trade": current_capital,
                }
            )
            
    # Close any open positions at the end of the data
    if position == 1: # Long position open
        profit_loss = (df['Close'].iloc[-1] - entry_price) * trade_size
        current_capital += profit_loss
        trades.append(
            {
                "trade_type": "Long_Exit",
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": df.index[-1],
                "exit_price": df['Close'].iloc[-1],
                "profit_loss": profit_loss,
                "trade_size": trade_size,
                "exit_reason": "EndOfData",
                "capital_after_trade": current_capital,
            }
        )
    elif position == -1: # Short position open
        profit_loss = (entry_price - df['Close'].iloc[-1]) * trade_size
        current_capital += profit_loss
        trades.append(
            {
                "trade_type": "Short_Exit",
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": df.index[-1],
                "exit_price": df['Close'].iloc[-1],
                "profit_loss": profit_loss,
                "trade_size": trade_size,
                "exit_reason": "EndOfData",
                "capital_after_trade": current_capital,
            }
        )

    # Calculate cumulative returns based on actual trades
    trade_profits = [t['profit_loss'] for t in trades if pd.notna(t['profit_loss'])]
    total_return = sum(trade_profits) if trade_profits else 0

    # Sharpe Ratio
    # Assuming risk-free rate is 0 for simplicity
    # For Sharpe ratio, we need daily returns. Let's approximate from trade profits.
    # This is a simplification and might not be perfectly accurate for daily returns.
    if trade_profits:
        daily_returns = pd.Series(trade_profits) / df['Close'].mean() # Normalize by average close price
        if daily_returns.std() != 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) # Annualized Sharpe
        else:
            sharpe_ratio = np.nan
    else:
        sharpe_ratio = np.nan

    # Drawdown
    # Recalculate cumulative returns based on capital after each trade
    capital_history = [initial_capital] + [t['capital_after_trade'] for t in trades if 'capital_after_trade' in t]
    cumulative_returns_from_capital = pd.Series(capital_history).pct_change().fillna(0).add(1).cumprod()
    if not cumulative_returns_from_capital.empty:
        peak = cumulative_returns_from_capital.expanding(min_periods=1).max()
        drawdown = (cumulative_returns_from_capital - peak) / peak
        max_drawdown = drawdown.min()
    else:
        max_drawdown = np.nan

    # Calculate additional metrics
    num_trades = len([t for t in trades if t['trade_type'].endswith('_Exit')])
    num_winning_trades = len([t for t in trades if pd.notna(t['profit_loss']) and t['profit_loss'] > 0])
    num_losing_trades = len([t for t in trades if pd.notna(t['profit_loss']) and t['profit_loss'] < 0])
    
    win_rate = (num_winning_trades / num_trades) * 100 if num_trades > 0 else 0
    
    winning_profits = [t['profit_loss'] for t in trades if pd.notna(t['profit_loss']) and t['profit_loss'] > 0]
    losing_losses = [t['profit_loss'] for t in trades if pd.notna(t['profit_loss']) and t['profit_loss'] < 0]
    
    avg_profit_per_winning_trade = sum(winning_profits) / num_winning_trades if num_winning_trades > 0 else 0
    avg_loss_per_losing_trade = sum(losing_losses) / num_losing_trades if num_losing_trades > 0 else 0
    
    total_winning_profit = sum(winning_profits)
    total_losing_loss = abs(sum(losing_losses))
    profit_factor = total_winning_profit / total_losing_loss if total_losing_loss > 0 else np.inf
    
    final_capital = current_capital

    return {
        "total_profit": total_return,
        "num_trades": num_trades,
        "num_winning_trades": num_winning_trades,
        "num_losing_trades": num_losing_trades,
        "win_rate": win_rate,
        "avg_profit_per_winning_trade": avg_profit_per_winning_trade,
        "avg_loss_per_losing_trade": avg_loss_per_losing_trade,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "final_capital": final_capital,
        'Trades': trades
    }

# =========================================================
# Strategy Optimization
# =========================================================

ohlc_sources = {
    'OHLC': {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'},
    # 'Heikin Ashi': {'open': 'HA_Open', 'high': 'HA_High', 'low': 'HA_Low', 'close': 'HA_Close'},
    # 'LinReg': {'open': 'LinReg_Open', 'high': 'LinReg_High', 'low': 'LinReg_Low', 'close': 'LinReg_Close'},
    # 'SHA OHLC': {'open': 'SHA_Open', 'high': 'SHA_High', 'low': 'SHA_Low', 'close': 'SHA_Close'}
}

best_profit = -np.inf
best_sharpe = -np.inf
lowest_drawdown = np.inf
best_config = {}
best_signals_df = None
best_trades_list = []

key_vals = [2.2]
atr_periods = [10]
stop_loss_percents = [0.005, 0.01] # Example stop loss percentages (0.5%, 1%, 1.5%, 2%)
take_profit_percents = [0.02] # Example take profit percentages (0.5%, 1%, 1.5%, 2%)

print("\nStarting strategy optimization...")

for source_name, cols in ohlc_sources.items():
    for key_val in key_vals:
        for atr_period in atr_periods:
            for stop_loss_percent in stop_loss_percents:
                for take_profit_percent in take_profit_percents:
                    print(f"Testing config: Source={source_name}, KeyVal={key_val}, ATRPeriod={atr_period}, StopLoss={stop_loss_percent*100:.2f}%, TakeProfit={take_profit_percent*100:.2f}%")
                    
                    # Ensure the necessary columns exist before passing to calculate_ut_bot
                    required_cols = [cols['open'], cols['high'], cols['low'], cols['close']]
                    if not all(col in df.columns for col in required_cols):
                        print(f"Skipping config due to missing columns: {required_cols}")
                        continue

                    signals_df = calculate_ut_bot(df.copy(), 
                                                  open_col=cols['open'], 
                                                  high_col=cols['high'], 
                                                  low_col=cols['low'], 
                                                  close_col=cols['close'], 
                                                  key_val=key_val, 
                                                  atr_period=atr_period)
                    
                    # Merge signals back to the original DataFrame for metric calculation
                    df_with_signals = df.copy()
                    df_with_signals['Buy'] = signals_df['Buy']
                    df_with_signals['Sell'] = signals_df['Sell']
                    
                    metrics_result = calculate_metrics(df_with_signals, stop_loss_percent=stop_loss_percent, take_profit_percent=take_profit_percent)

                    current_profit = metrics_result['total_profit']
                    current_sharpe = metrics_result['sharpe_ratio']
                    current_drawdown = metrics_result['max_drawdown']
                    current_trades = metrics_result['Trades']

                    # Prioritize profit, then Sharpe, then drawdown
                    if current_profit > best_profit:
                        best_profit = current_profit
                        best_sharpe = current_sharpe
                        lowest_drawdown = current_drawdown
                        best_config = {
                            'Source': source_name,
                            'KeyVal': key_val,
                            'ATRPeriod': atr_period,
                            'StopLoss': stop_loss_percent,
                            'TakeProfit': take_profit_percent,
                            'Metrics': metrics_result # Store all metrics
                        }
                        best_signals_df = signals_df.copy()
                        best_trades_list = current_trades
                    elif current_profit == best_profit:
                        if current_sharpe > best_sharpe:
                            best_sharpe = current_sharpe
                            lowest_drawdown = current_drawdown
                            best_config = {
                                'Source': source_name,
                                'KeyVal': key_val,
                                'ATRPeriod': atr_period,
                                'StopLoss': stop_loss_percent,
                                'TakeProfit': take_profit_percent,
                                'Metrics': metrics_result # Store all metrics
                            }
                            best_signals_df = signals_df.copy()
                            best_trades_list = current_trades
                        elif current_sharpe == best_sharpe:
                            if current_drawdown < lowest_drawdown:
                                lowest_drawdown = current_drawdown
                                best_config = {
                                    'Source': source_name,
                                    'KeyVal': key_val,
                                    'ATRPeriod': atr_period,
                                    'StopLoss': stop_loss_percent,
                                    'TakeProfit': take_profit_percent,
                                    'Metrics': metrics_result # Store all metrics
                                }
                                best_signals_df = signals_df.copy()
                                best_trades_list = current_trades

print("\nOptimization Complete.")
print("\nBest Configuration Found:")
if best_config:
    print(f"Source: {best_config['Source']}")
    print(f"Key Value: {best_config['KeyVal']}")
    print(f"ATR Period: {best_config['ATRPeriod']}")
    print(f"Stop Loss: {best_config['StopLoss']*100:.2f}%")
    print(f"Take Profit: {best_config['TakeProfit']*100:.2f}%")
    print(f"Total Profit: {best_config['Metrics']['total_profit']:.2f}")
    print(f"Number of Trades: {best_config['Metrics']['num_trades']}")
    print(f"Number of Winning Trades: {best_config['Metrics']['num_winning_trades']}")
    print(f"Number of Losing Trades: {best_config['Metrics']['num_losing_trades']}")
    print(f"Win Rate: {best_config['Metrics']['win_rate']:.2f}%")
    print(f"Avg Profit per Winning Trade: {best_config['Metrics']['avg_profit_per_winning_trade']:.2f}")
    print(f"Avg Loss per Losing Trade: {best_config['Metrics']['avg_loss_per_losing_trade']:.2f}")
    print(f"Max Drawdown: {best_config['Metrics']['max_drawdown']:.2f}")
    print(f"Profit Factor: {best_config['Metrics']['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {best_config['Metrics']['sharpe_ratio']:.2f}")
    print(f"Final Capital: {best_config['Metrics']['final_capital']:.2f}")
else:
    print("No valid configuration found.")

if best_signals_df is not None:
    print("\nSample of Buy/Sell Signals from Best Configuration:")
    print(best_signals_df.tail())

# Save trades to CSV
if best_trades_list:
    trades_df = pd.DataFrame(best_trades_list)
    trades_df.to_csv('linear_regression_trades.csv', index=False)
    print("\nTrades saved to linear_regression_trades.csv")
else:
    print("\nNo trades to save.")

# =========================================================
# Round all numerical columns to 2 decimal places
# =========================================================
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].round(2)

print("\nDataFrame after rounding all numerical columns to 2 decimal places:")
print(df.tail())
