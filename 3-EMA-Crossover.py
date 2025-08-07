import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_signals(df, fast_len=3, medium_len=6, slow_len=9):
    """
    Generates trading signals for a 3-EMA crossover strategy.

    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices.
        fast_len (int): Period for the fast EMA.
        medium_len (int): Period for the medium EMA.
        slow_len (int): Period for the slow EMA.

    Returns:
        pd.DataFrame: Original DataFrame with EMA values and 'signal' column.
    """
    df['fast_ema'] = df['Close'].ewm(span=fast_len, adjust=False).mean()
    df['medium_ema'] = df['Close'].ewm(span=medium_len, adjust=False).mean()
    df['slow_ema'] = df['Close'].ewm(span=slow_len, adjust=False).mean()

    # Create signal column: 1 for long, -1 for short, 0 for neutral
    df['signal'] = 0.0

    # Long entry condition: fast > medium > slow
    df.loc[(df['fast_ema'] > df['medium_ema']) &
           (df['medium_ema'] > df['slow_ema']), 'signal'] = 1.0

    # Short entry condition: fast < medium < slow
    df.loc[(df['fast_ema'] < df['medium_ema']) &
           (df['medium_ema'] < df['slow_ema']), 'signal'] = -1.0

    # Handle exit conditions based on EMA order disruption
    # We check for a change in the sorted order of the EMAs
    df['ema_order_long'] = (df['fast_ema'] > df['medium_ema']) & (df['medium_ema'] > df['slow_ema'])
    df['ema_order_short'] = (df['fast_ema'] < df['medium_ema']) & (df['medium_ema'] < df['slow_ema'])

    # Shift the signal to check for changes
    df['signal_shifted'] = df['signal'].shift(1)

    # If previously long (signal_shifted == 1) and EMA order for long is no longer true, set signal to 0 (exit)
    df.loc[(df['signal_shifted'] == 1) & (df['ema_order_long'] == False), 'signal'] = 0.0

    # If previously short (signal_shifted == -1) and EMA order for short is no longer true, set signal to 0 (exit)
    df.loc[(df['signal_shifted'] == -1) & (df['ema_order_short'] == False), 'signal'] = 0.0

    return df

def backtest(df, initial_capital=10000, transaction_cost=0.001):
    """
    Performs a simple backtest of the trading strategy.

    Args:
        df (pd.DataFrame): DataFrame with 'signal', 'Close', and 'Date' columns.
        initial_capital (float): Starting capital.
        transaction_cost (float): Cost per trade as a percentage.

    Returns:
        tuple: A tuple containing the equity curve and a list of trades.
    """
    capital = initial_capital
    portfolio_value = [capital]
    trades = []
    position = 0 # 1 for long, -1 for short, 0 for neutral

    for i in range(1, len(df)):
        current_signal = df['signal'].iloc[i]
        price = df['Close'].iloc[i]
        date = df.index[i]

        # If currently neutral
        if position == 0:
            if current_signal == 1: # Enter Long
                position = 1
                entry_price = price
                trade_size = capital
                capital -= trade_size * transaction_cost
                trades.append({'type': 'long', 'entry_date': date, 'entry_price': entry_price, 'exit_date': None, 'exit_price': None})
            elif current_signal == -1: # Enter Short
                position = -1
                entry_price = price
                trade_size = capital
                capital -= trade_size * transaction_cost
                trades.append({'type': 'short', 'entry_date': date, 'entry_price': entry_price, 'exit_date': None, 'exit_price': None})
        
        # If currently long
        elif position == 1:
            if current_signal == -1: # Reversal: Exit Long and Enter Short
                # Exit long
                exit_price = price
                profit = (exit_price - trades[-1]['entry_price']) / trades[-1]['entry_price']
                capital += capital * profit
                capital -= capital * transaction_cost
                trades[-1]['exit_date'] = date
                trades[-1]['exit_price'] = exit_price
                
                # Enter short
                position = -1
                entry_price = price
                trade_size = capital 
                capital -= trade_size * transaction_cost
                trades.append({'type': 'short', 'entry_date': date, 'entry_price': entry_price, 'exit_date': None, 'exit_price': None})
            elif current_signal == 0: # Exit Long to Neutral (EMA order disturbed)
                exit_price = price
                profit = (exit_price - trades[-1]['entry_price']) / trades[-1]['entry_price']
                capital += capital * profit
                capital -= capital * transaction_cost
                trades[-1]['exit_date'] = date
                trades[-1]['exit_price'] = exit_price
                position = 0

        # If currently short
        elif position == -1:
            if current_signal == 1: # Reversal: Exit Short and Enter Long
                # Exit short
                exit_price = price
                profit = (trades[-1]['entry_price'] - exit_price) / trades[-1]['entry_price']
                capital += capital * profit
                capital -= capital * transaction_cost
                trades[-1]['exit_date'] = date
                trades[-1]['exit_price'] = exit_price
                
                # Enter long
                position = 1
                entry_price = price
                trade_size = capital 
                capital -= trade_size * transaction_cost
                trades.append({'type': 'long', 'entry_date': date, 'entry_price': entry_price, 'exit_date': None, 'exit_price': None})
            elif current_signal == 0: # Exit Short to Neutral (EMA order disturbed)
                exit_price = price
                profit = (trades[-1]['entry_price'] - exit_price) / trades[-1]['entry_price']
                capital += capital * profit
                capital -= capital * transaction_cost
                trades[-1]['exit_date'] = date
                trades[-1]['exit_price'] = exit_price
                position = 0

        portfolio_value.append(capital)

    return pd.Series(portfolio_value, index=df.index), trades

def calculate_metrics(equity_curve, trades):
    """
    Calculates key performance metrics.

    Args:
        equity_curve (pd.Series): Series of portfolio values over time.
        trades (list): List of trade dictionaries.

    Returns:
        dict: A dictionary of performance metrics.
    """
    if equity_curve.empty:
        return {}

    # Filter out trades that have not been closed (exit_price is None)
    closed_trades = [t for t in trades if t['exit_price'] is not None]

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    daily_returns = equity_curve.pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0

    max_drawdown = (equity_curve.div(equity_curve.cummax()) - 1).min() * 100

    # Use the filtered list of closed trades for win/loss calculations
    winning_trades = [t for t in closed_trades if (t['type'] == 'long' and t['exit_price'] > t['entry_price']) or
                                                 (t['type'] == 'short' and t['exit_price'] < t['entry_price'])]
    total_trades = len(closed_trades)
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

    return {
        'Total Return (%)': total_return,
        'Max Drawdown (%)': max_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Win Rate (%)': win_rate,
        'Total Trades': total_trades
    }


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# (Keep all previously defined functions: generate_signals, backtest, calculate_metrics)

def optimize_ema_strategy(df, fast_range, medium_range, slow_range, initial_capital=10000):
    """
    Optimizes the 3-EMA crossover strategy by testing different parameter combinations.

    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices.
        fast_range (list): A list of integer values for the fast EMA period.
        medium_range (list): A list of integer values for the medium EMA period.
        slow_range (list): A list of integer values for the slow EMA period.
        initial_capital (float): Starting capital for each backtest.

    Returns:
        pd.DataFrame: A DataFrame of optimization results, sorted by Sharpe Ratio.
    """
    results = []

    for fast_len in fast_range:
        for medium_len in medium_range:
            for slow_len in slow_range:
                # Ensure the EMA periods are in the correct order (fast < medium < slow)
                if not (fast_len < medium_len < slow_len):
                    continue

                # Generate signals for the current parameters
                try:
                    data_with_signals = generate_signals(df.copy(), fast_len, medium_len, slow_len)
                    
                    # Backtest the strategy
                    equity_curve, trades = backtest(data_with_signals, initial_capital)
                    
                    # Calculate metrics
                    metrics = calculate_metrics(equity_curve, trades)
                    
                    if metrics:
                        results.append({
                            'fast_ema': fast_len,
                            'medium_ema': medium_len,
                            'slow_ema': slow_len,
                            **metrics
                        })
                except Exception as e:
                    print(f"Error with parameters ({fast_len}, {medium_len}, {slow_len}): {e}")

    results_df = pd.DataFrame(results)

    # Calculate Calmar Ratio
    if 'Max Drawdown (%)' in results_df.columns and 'Total Return (%)' in results_df.columns:
        results_df['Calmar Ratio'] = results_df['Total Return (%)'] / results_df['Max Drawdown (%)'].abs()

    # Sort results by a key metric, e.g., Calmar Ratio or Sharpe Ratio
    results_df = results_df.sort_values(by='Calmar Ratio', ascending=False)
    
    return results_df

# --- Example Usage ---

# Create some sample data
# np.random.seed(42)
# dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=200, freq='D'))
# prices = 100 + np.random.randn(200).cumsum()
# sample_data = pd.DataFrame({'Close': prices}, index=dates)

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

# Define the parameter ranges for optimization
fast_ema_periods = range(5, 15)
medium_ema_periods = range(10, 20)
slow_ema_periods = range(15, 25)

print("Starting optimization...")
optimization_results = optimize_ema_strategy(
    df, 
    fast_ema_periods, 
    medium_ema_periods, 
    slow_ema_periods
)
print("Optimization complete.")

# Display the top 10 results
print("\n--- Top 10 Optimization Results (Sorted by Calmar Ratio) ---")
print(optimization_results.head(10).to_string())

# --- Visualization for the best performing strategy ---
if not optimization_results.empty:
    best_params = optimization_results.iloc[0]
    best_fast = int(best_params['fast_ema'])
    best_medium = int(best_params['medium_ema'])
    best_slow = int(best_params['slow_ema'])

    print(f"\nBest parameters: Fast EMA = {best_fast}, Medium EMA = {best_medium}, Slow EMA = {best_slow}")
    
    # Run the backtest with the best parameters
    best_strategy_data = generate_signals(df.copy(), best_fast, best_medium, best_slow)
    best_equity_curve, best_trades = backtest(best_strategy_data)

    # Calculate and print metrics for the best performing strategy
    best_metrics = calculate_metrics(best_equity_curve, best_trades)
    print("\n--- Metrics for Best Performing Strategy ---")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.2f}")

    # Plot the price data with EMAs and trades for the best strategy
    plt.figure(figsize=(14, 7))
    plt.plot(best_strategy_data.index, best_strategy_data['Close'], label='Close Price', color='black')
    plt.plot(best_strategy_data.index, best_strategy_data['fast_ema'], label=f'Fast EMA ({best_fast})', color='orange')
    plt.plot(best_strategy_data.index, best_strategy_data['medium_ema'], label=f'Medium EMA ({best_medium})', color='purple')
    plt.plot(best_strategy_data.index, best_strategy_data['slow_ema'], label=f'Slow EMA ({best_slow})', color='blue')

    # Mark entry and exit points
    long_entries = best_strategy_data[(best_strategy_data['signal'] == 1) & (best_strategy_data['signal'].shift(1) == 0)]
    short_entries = best_strategy_data[(best_strategy_data['signal'] == -1) & (best_strategy_data['signal'].shift(1) == 0)]
    
    plt.scatter(long_entries.index, long_entries['Close'], marker='^', color='g', s=100, label='Long Entry', zorder=5)
    plt.scatter(short_entries.index, short_entries['Close'], marker='v', color='r', s=100, label='Short Entry', zorder=5)

    plt.title(f'3-EMA Crossover Strategy with Best Parameters (F:{best_fast}, M:{best_medium}, S:{best_slow})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the equity curve for the best strategy
    plt.figure(figsize=(14, 7))
    plt.plot(best_equity_curve.index, best_equity_curve, label='Portfolio Equity Curve', color='cyan')
    plt.title('Portfolio Equity Curve (Best Parameters)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    plt.show()
