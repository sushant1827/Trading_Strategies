import pandas as pd
import numpy as np
from Multi_Kernel_Regression import MultiKernelRegression, calculate_kernel_regression_numba
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float = 100000) -> dict:
    """
    Calculates various trading performance metrics.

    Parameters:
    trades_df (pd.DataFrame): DataFrame containing trade details with 'profit_loss' column.
    initial_capital (float): Initial capital for drawdown calculation.

    Returns:
    dict: A dictionary of calculated trading metrics.
    """
    if trades_df.empty:
        return {
            "num_total_trades": 0, "num_winning_trades": 0, "num_losing_trades": 0,
            "win_rate": 0, "avg_win": 0, "avg_loss": 0, "risk_reward_ratio": 0,
            "profit_factor": 0, "expectancy": 0, "initial_capital": initial_capital,
            "net_profit": 0, "net_profit_percent": 0, "max_drawdown": 0,
            "return_drawdown_ratio": 0, "max_winning_streak": 0, "max_losing_streak": 0,
            "sharpe_ratio": 0, "sortino_ratio": 0, "profit_by_year": {},
            "profit_by_month_year": {}
        }

    # Calculate equity curve
    trades_df['cumulative_profit'] = trades_df['profit_loss'].cumsum()
    trades_df['equity_curve'] = initial_capital + trades_df['cumulative_profit']

    # Basic trade statistics
    num_total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['profit_loss'] > 0]
    losing_trades = trades_df[trades_df['profit_loss'] < 0]
    num_winning_trades = len(winning_trades)
    num_losing_trades = len(losing_trades)

    win_rate = (num_winning_trades / num_total_trades) * 100 if num_total_trades > 0 else 0
    avg_win = winning_trades['profit_loss'].mean() if num_winning_trades > 0 else 0
    avg_loss = losing_trades['profit_loss'].mean() if num_losing_trades > 0 else 0
    
    risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    profit_factor = winning_trades['profit_loss'].sum() / abs(losing_trades['profit_loss'].sum()) if abs(losing_trades['profit_loss'].sum()) > 0 else np.inf
    expectancy = (win_rate / 100) * avg_win + ((100 - win_rate) / 100) * avg_loss

    net_profit = trades_df['profit_loss'].sum()
    net_profit_percent = (net_profit / initial_capital) * 100

    # Drawdown calculation
    peak = trades_df['equity_curve'].expanding(min_periods=1).max()
    drawdown = (trades_df['equity_curve'] - peak) / peak * 100
    max_drawdown = drawdown.min() if not drawdown.empty else 0

    return_drawdown_ratio = net_profit_percent / abs(max_drawdown) if max_drawdown != 0 else np.inf

    # Streaks
    winning_streak = 0
    losing_streak = 0
    max_winning_streak = 0
    max_losing_streak = 0
    for pl in trades_df['profit_loss']:
        if pl > 0:
            winning_streak += 1
            losing_streak = 0
        else:
            losing_streak += 1
            winning_streak = 0
        max_winning_streak = max(max_winning_streak, winning_streak)
        max_losing_streak = max(max_losing_streak, losing_streak)

    # Sharpe Ratio (assuming daily returns for simplicity, adjust if needed)
    # For trading strategies, often calculated on trade-by-trade returns or daily equity curve changes
    # Here, we'll use daily equity curve changes for a rough estimate
    daily_returns = trades_df['equity_curve'].diff().dropna() / trades_df['equity_curve'].shift(1).dropna()
    if not daily_returns.empty:
        sharpe_ratio = (daily_returns.mean() * np.sqrt(252)) / daily_returns.std() if daily_returns.std() != 0 else 0
    else:
        sharpe_ratio = 0

    # Sortino Ratio (using daily returns, only considering negative deviations)
    if not daily_returns.empty:
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std()
        sortino_ratio = (daily_returns.mean() * np.sqrt(252)) / downside_std if downside_std != 0 else 0
    else:
        sortino_ratio = 0

    # Profit by Year
    trades_df['year'] = trades_df['exit_time'].dt.year
    profit_by_year = trades_df.groupby('year')['profit_loss'].sum().to_dict()

    # Profit by Month (for each year)
    trades_df['month'] = trades_df['exit_time'].dt.month
    profit_by_month_year = trades_df.groupby(['year', 'month'])['profit_loss'].sum().unstack(fill_value=0).to_dict('index')

    return {
        "num_total_trades": num_total_trades,
        "num_winning_trades": num_winning_trades,
        "num_losing_trades": num_losing_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "risk_reward_ratio": risk_reward_ratio,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "initial_capital": initial_capital,
        "net_profit": net_profit,
        "net_profit_percent": net_profit_percent,
        "max_drawdown": max_drawdown,
        "return_drawdown_ratio": return_drawdown_ratio,
        "max_winning_streak": max_winning_streak,
        "max_losing_streak": max_losing_streak,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "profit_by_year": profit_by_year,
        "profit_by_month_year": profit_by_month_year
    }

def run_backtest(df_60min: pd.DataFrame, kernel: str, bandwidth: int, initial_capital: float = 100000, return_signals: bool = False):
    """
    Runs a backtest for the single timeframe MKR strategy on 60-minute data.

    Parameters:
    df_60min (pd.DataFrame): 60-minute OHLC data.
    kernel (str): Kernel type for MKR.
    bandwidth (int): Bandwidth for the kernel.
    initial_capital (float): Initial capital for backtesting.
    return_signals (bool): Whether to return signals dataframe along with metrics.

    Returns:
    dict or tuple: Performance metrics of the strategy, optionally with signals dataframe and trades dataframe.
    """
    # Apply MKR to 60-minute timeframe
    mkr_60min = MultiKernelRegression(bandwidth=bandwidth, kernel=kernel)
    df_60min = mkr_60min.generate_signals(df_60min)

    # Strategy Logic for single timeframe - Fixed Look-ahead Bias
    position = 0  # 0: no position, 1: long, -1: short
    entry_price = 0
    trades = []

    for i in range(1, len(df_60min)):
        current_time = df_60min.index[i]
        current_open = df_60min['Open'].iloc[i]
        
        # Get signal from the previous candle (to avoid look-ahead bias)
        signal = df_60min['signal'].iloc[i-1]

        # Entry Conditions
        if position == 0:
            # Buy signal: positive signal
            if signal == 1:
                position = 1
                entry_price = current_open  # Enter at open to avoid look-ahead bias
                entry_time = current_time
            # Sell signal: negative signal
            elif signal == -1:
                position = -1
                entry_price = current_open  # Enter at open to avoid look-ahead bias
                entry_time = current_time
        # Exit Conditions
        elif position == 1:  # Currently Long
            # Exit if signal turns negative
            if signal == -1:
                exit_price = current_open  # Exit at open to avoid look-ahead bias
                profit_loss = exit_price - entry_price
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'type': 'long',
                    'profit_loss': profit_loss
                })
                position = 0
                entry_price = 0
        elif position == -1:  # Currently Short
            # Exit if signal turns positive
            if signal == 1:
                exit_price = current_open  # Exit at open to avoid look-ahead bias
                profit_loss = entry_price - exit_price
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'type': 'short',
                    'profit_loss': profit_loss
                })
                position = 0
                entry_price = 0
    
    trades_df = pd.DataFrame(trades)
    metrics = calculate_metrics(trades_df, initial_capital)
    
    if return_signals:
        return metrics, df_60min, trades_df
    else:
        return metrics

def optimize_strategy(df_60min: pd.DataFrame, initial_capital: float = 100000):
    """
    Optimizes the single timeframe MKR strategy by iterating through parameters.
    """
    best_metrics = None
    best_params = {}
    best_signals_df = None
    best_trades_df = None

    # Define parameter ranges for optimization
    kernels = ["Laplace", "Gaussian", "Epanechnikov"] # Add more as needed "Logistic", "Triangular"
    
    # Bandwidth ranges for 60-minute timeframe
    bandwidth_ranges = range(10, 51, 5)  # e.g., 15, 20, 25, 30, 35, 40, 45, 50

    total_combinations = len(kernels) * len(bandwidth_ranges)
    
    print(f"Starting optimization with {total_combinations} combinations...")
    
    current_combination = 0
    for kernel in kernels:
        for bandwidth in bandwidth_ranges:
            current_combination += 1
            print(f"Testing combination {current_combination}/{total_combinations}: Kernel={kernel}, Bandwidth={bandwidth}")
            
            metrics = run_backtest(df_60min.copy(), kernel, bandwidth, initial_capital)

            # Optimization objective: Maximize Sharpe Ratio
            current_score = metrics['sharpe_ratio'] # Primary
            
            if best_metrics is None or current_score > best_metrics['sharpe_ratio']:
                best_metrics = metrics
                best_params = {
                    "kernel": kernel,
                    "bandwidth": bandwidth
                }
            elif current_score == best_metrics['sharpe_ratio']:
                # If Sharpe Ratios are equal, compare by Net Profit
                if metrics['net_profit'] > best_metrics['net_profit']:
                    best_metrics = metrics
                    best_params = {
                        "kernel": kernel,
                        "bandwidth": bandwidth
                    }
                # If Net Profits are also equal, compare by Drawdown (lower is better)
                elif metrics['net_profit'] == best_metrics['net_profit'] and metrics['max_drawdown'] > best_metrics['max_drawdown']: # max_drawdown is negative, so greater means less negative (better)
                    best_metrics = metrics
                    best_params = {
                        "kernel": kernel,
                        "bandwidth": bandwidth
                    }

    print("\n--- Optimization Complete ---")
    print("\nBest Parameters Found:")
    print(f"Kernel: {best_params.get('kernel')}")
    print(f"Bandwidth: {best_params.get('bandwidth')}")

    if best_metrics:
        print("\n--- Best Trading Metrics ---")
        print(f"Total Trades: {best_metrics['num_total_trades']}")
        print(f"Winning Trades: {best_metrics['num_winning_trades']}")
        print(f"Losing Trades: {best_metrics['num_losing_trades']}")
        print(f"Win Rate: {best_metrics['win_rate']:.2f}%")
        print(f"Average Win: {best_metrics['avg_win']:.2f}")
        print(f"Average Loss: {best_metrics['avg_loss']:.2f}")
        print(f"Risk-Reward Ratio: {best_metrics['risk_reward_ratio']:.2f}")
        print(f"Profit Factor: {best_metrics['profit_factor']:.2f}")
        print(f"Expectancy: {best_metrics['expectancy']:.2f}")
        print(f"Initial Capital for Drawdown Calculation: {best_metrics['initial_capital']:.2f}")
        print(f"Net Profit: {best_metrics['net_profit']:.2f}")
        print(f"Net Profit (% of Initial Capital): {best_metrics['net_profit_percent']:.2f}%")
        print(f"Maximum Drawdown: {best_metrics['max_drawdown']:.2f}%")
        print(f"Return-Drawdown Ratio: {best_metrics['return_drawdown_ratio']:.2f}")
        print(f"Max Winning Streak: {best_metrics['max_winning_streak']}")
        print(f"Max Losing Streak: {best_metrics['max_losing_streak']}")
        print(f"Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {best_metrics['sortino_ratio']:.2f}")
        print("\nProfit by Year:")
        for year, profit in best_metrics['profit_by_year'].items():
            print(f"  {year}: {profit:.2f}")
        print("\nProfit by Month (for each year):")
        for year, months_data in best_metrics['profit_by_month_year'].items():
            print(f"  Year {year}:")
            for month, profit in months_data.items():
                print(f"    Month {month}: {profit:.2f}")
        # Generate signals and trades for the best parameters to save to CSV
        print("\n--- Generating final results with best parameters for CSV export ---")
        best_metrics_final, best_signals_df, best_trades_df = run_backtest(
            df_60min.copy(), 
            best_params['kernel'], 
            best_params['bandwidth'], 
            initial_capital, 
            return_signals=True
        )
        
        # Save signals to CSV
        signals_filename = f"mkr_60min_signals_{best_params['kernel']}_bw{best_params['bandwidth']}.csv"
        best_signals_df.to_csv(signals_filename)
        print(f"Signals saved to: {signals_filename}")
        
        # Save trades to CSV
        if not best_trades_df.empty:
            trades_filename = f"mkr_60min_trades_{best_params['kernel']}_bw{best_params['bandwidth']}.csv"
            best_trades_df.to_csv(trades_filename, index=False)
            print(f"Trades saved to: {trades_filename}")
        else:
            print("No trades generated - trades CSV not created.")
            
    else:
        print("No profitable combination found.")

if __name__ == "__main__":
    csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'
    
    try:
        df_60min = pd.read_csv(csv_file_path)
        
        # Data Preprocessing
        df_60min = df_60min.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
        df_60min = df_60min.rename(columns={'Date': 'DateTime'})
        df_60min['DateTime'] = pd.to_datetime(df_60min['DateTime'])
        df_60min = df_60min.set_index('DateTime')
        df_60min = df_60min[~df_60min.index.duplicated(keep='first')]
        df_60min = df_60min[df_60min.index.year >= 2021] # Filter data from 2021 onwards
        
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df_60min.columns:
                df_60min[col] = df_60min[col].round(2)

        print(f"Loaded 60-minute data from {csv_file_path}. Shape: {df_60min.shape}")
        print(f"Data range: {df_60min.index.min()} to {df_60min.index.max()}")

        optimize_strategy(df_60min.copy())

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
