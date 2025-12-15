import pandas as pd
import numpy as np
from numba import jit

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Data Configuration
CSV_FILE_PATH = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'
# Alternative data files:
# CSV_FILE_PATH = 'Nifty50_Index/NIFTY50_INDEX_60_Min.csv'
# CSV_FILE_PATH = 'Nifty50_Index/NIFTY50_INDEX_120_Min.csv'

# Supertrend Parameters
ST_PERIOD = 25
ST_MULTIPLIER = 1.2

# Strategy Configuration
INITIAL_CAPITAL = 100000
START_YEAR = 2019

# Data Processing
COLUMNS_TO_ROUND = ['Open', 'High', 'Low', 'Close']
DECIMAL_PLACES = 2

# Output Configuration
RESULTS_CSV_PATH = 'ohlc_supertrend_results.csv'

@jit(nopython=True)
def _calculate_supertrend_numba(high, low, close, tr, atr, period, multiplier, basic_upper_band, basic_lower_band, first_valid_atr_idx):
    """
    Calculate SuperTrend using OHLC data instead of Heikin-Ashi
    """
    trending_up = np.empty_like(close)
    trending_down = np.empty_like(close)
    direction = np.empty_like(close, dtype=np.int32)
    supertrend = np.empty_like(close)

    if first_valid_atr_idx is None or first_valid_atr_idx >= len(close):
        return trending_up, trending_down, direction, supertrend

    trending_up[first_valid_atr_idx] = basic_lower_band[first_valid_atr_idx]
    trending_down[first_valid_atr_idx] = basic_upper_band[first_valid_atr_idx]

    if close[first_valid_atr_idx] > trending_down[first_valid_atr_idx]:
        direction[first_valid_atr_idx] = 1
    elif close[first_valid_atr_idx] < trending_up[first_valid_atr_idx]:
        direction[first_valid_atr_idx] = -1
    else:
        direction[first_valid_atr_idx] = 1

    supertrend[first_valid_atr_idx] = trending_up[first_valid_atr_idx] if direction[first_valid_atr_idx] == 1 else trending_down[first_valid_atr_idx]

    for i in range(first_valid_atr_idx + 1, len(close)):
        prev_close = close[i - 1]
        prev_trending_up = trending_up[i - 1]
        prev_trending_down = trending_down[i - 1]
        prev_direction = direction[i - 1]

        current_basic_upper_band = basic_upper_band[i]
        current_basic_lower_band = basic_lower_band[i]
        current_close = close[i]

        if prev_close > prev_trending_up:
            trending_up[i] = max(current_basic_lower_band, prev_trending_up)
        else:
            trending_up[i] = current_basic_lower_band

        if prev_close < prev_trending_down:
            trending_down[i] = min(current_basic_upper_band, prev_trending_down)
        else:
            trending_down[i] = current_basic_upper_band

        if current_close > trending_down[i - 1]:
            direction[i] = 1
        elif current_close < trending_up[i - 1]:
            direction[i] = -1
        else:
            direction[i] = prev_direction

        supertrend[i] = trending_up[i] if direction[i] == 1 else trending_down[i]

    return trending_up, trending_down, direction, supertrend

def calculate_supertrend(df, period=10, multiplier=3.0, suffix=""):
    """
    Calculate standard SuperTrend indicator using OHLC data

    Parameters:
    df (pd.DataFrame): DataFrame with Open, High, Low, Close columns
    period (int): ATR period (default: 10)
    multiplier (float): ATR multiplier for SuperTrend bands (default: 3.0)
    suffix (str): Suffix for column names
    """
    original_index = df.index
    df_reset = df.reset_index(drop=True)

    if not all(col in df_reset.columns for col in ["Open", "High", "Low", "Close"]):
        raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns for Supertrend calculation.")

    high = df_reset["High"].to_numpy()
    low = df_reset["Low"].to_numpy()
    close = df_reset["Close"].to_numpy()

    # Calculate True Range based on OHLC candles
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # Correct TR for the first element

    # ATR using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]  # Initialize first ATR
    alpha = 1 / period
    for i in range(1, len(tr)):
        atr[i] = (tr[i] * alpha) + (atr[i-1] * (1 - alpha))

    # Calculate SuperTrend bands using OHLC high/low
    basic_upper_band = ((high + low) / 2) + (multiplier * atr)
    basic_lower_band = ((high + low) / 2) - (multiplier * atr)

    first_valid_atr_idx = np.where(~np.isnan(atr))[0][0] if np.any(~np.isnan(atr)) else -1

    trending_up_arr, trending_down_arr, direction_arr, supertrend_arr = _calculate_supertrend_numba(
        high, low, close, tr, atr, period, multiplier, basic_upper_band, basic_lower_band, first_valid_atr_idx
    )

    st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    trending_up_col_name = f"trendingUp_OHLC_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    trending_down_col_name = f"trendingDown_OHLC_{period}_{str(multiplier).replace('.', '_')}{suffix}"

    result_df = pd.DataFrame({
        st_col_name: supertrend_arr,
        st_dir_col_name: direction_arr,
        trending_up_col_name: trending_up_arr,
        trending_down_col_name: trending_down_arr
    }, index=original_index)

    return result_df

@jit(nopython=True)
def _run_strategy_numba(st_dir, close_prices):
    """
    Execute SuperTrend trading strategy

    Parameters:
    st_dir (np.array): SuperTrend direction array (1 for up, -1 for down)
    close_prices (np.array): Close price array

    Returns:
    tuple: Buy/Sell entry and exit price arrays
    """
    buy_entry_prices = np.full_like(close_prices, np.nan)
    buy_exit_prices = np.full_like(close_prices, np.nan)
    sell_entry_prices = np.full_like(close_prices, np.nan)
    sell_exit_prices = np.full_like(close_prices, np.nan)

    in_buy_position = False
    in_sell_position = False

    for i in range(1, len(close_prices)):
        # Check for Buy_Entry condition (Supertrend direction changes to up)
        if not in_buy_position and not in_sell_position and st_dir[i] == 1:
            buy_entry_prices[i] = close_prices[i]
            in_buy_position = True
        # Check for Buy_Exit condition (Supertrend direction changes to down)
        elif in_buy_position and st_dir[i] == -1:
            buy_exit_prices[i] = close_prices[i]
            in_buy_position = False

        # Check for Sell_Entry condition (Supertrend direction changes to down)
        if not in_sell_position and not in_buy_position and st_dir[i] == -1:
            sell_entry_prices[i] = close_prices[i]
            in_sell_position = True
        # Check for Sell_Exit condition (Supertrend direction changes to up)
        elif in_sell_position and st_dir[i] == 1:
            sell_exit_prices[i] = close_prices[i]
            in_sell_position = False

    return buy_entry_prices, buy_exit_prices, sell_entry_prices, sell_exit_prices

def run_strategy_and_get_metrics(df_original, supertrend_params_list):
    """
    Run SuperTrend strategy and calculate performance metrics

    Parameters:
    df_original (pd.DataFrame): Original price data with OHLC columns
    supertrend_params_list (list): List of (period, multiplier) tuples

    Returns:
    dict: Performance metrics
    """
    df = df_original.copy()

    st_direction_cols = []

    # Use the first (and only) Supertrend parameter
    period, multiplier = supertrend_params_list[0]
    suffix = "_1"

    # Calculate OHLC Supertrend directly
    st_result_ohlc = calculate_supertrend(
        df.copy(),
        period=period,
        multiplier=multiplier,
        suffix=suffix
    )
    st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    trending_up_col_name = f"trendingUp_OHLC_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    trending_down_col_name = f"trendingDown_OHLC_{period}_{str(multiplier).replace('.', '_')}{suffix}"

    st_direction_cols.append(st_dir_col_name)

    df = pd.concat([df, st_result_ohlc], axis=1)

    df = df.drop(columns=[trending_up_col_name, trending_down_col_name], errors='ignore')

    # Convert direction column to numpy array for Numba
    st_dir = df[st_direction_cols[0]].to_numpy()
    close_prices = df['Close'].to_numpy()

    buy_entry_prices, buy_exit_prices, sell_entry_prices, sell_exit_prices = _run_strategy_numba(
        st_dir, close_prices
    )

    df['Buy_Entry'] = buy_entry_prices
    df['Buy_Exit'] = buy_exit_prices
    df['Sell_Entry'] = sell_entry_prices
    df['Sell_Exit'] = sell_exit_prices

    trades = []
    current_buy_entry = None
    current_sell_entry = None

    for i in range(len(df)):
        if not pd.isna(df['Buy_Entry'].iloc[i]):
            current_buy_entry = {'entry_price': df['Buy_Entry'].iloc[i], 'entry_date': df.index[i]}
        if not pd.isna(df['Buy_Exit'].iloc[i]) and current_buy_entry is not None:
            profit = df['Buy_Exit'].iloc[i] - current_buy_entry['entry_price']
            trades.append({
                'type': 'buy',
                'entry_date': current_buy_entry['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_buy_entry['entry_price'],
                'exit_price': df['Buy_Exit'].iloc[i],
                'profit': profit
            })
            current_buy_entry = None

        if not pd.isna(df['Sell_Entry'].iloc[i]):
            current_sell_entry = {'entry_price': df['Sell_Entry'].iloc[i], 'entry_date': df.index[i]}
        if not pd.isna(df['Sell_Exit'].iloc[i]) and current_sell_entry is not None:
            profit = current_sell_entry['entry_price'] - df['Sell_Exit'].iloc[i]
            trades.append({
                'type': 'sell',
                'entry_date': current_sell_entry['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_sell_entry['entry_price'],
                'exit_price': df['Sell_Exit'].iloc[i],
                'profit': profit
            })
            current_sell_entry = None

    trades_df = pd.DataFrame(trades)

    metrics = {}
    if not trades_df.empty:
        total_profit = trades_df['profit'].sum()
        winning_trades = trades_df[trades_df['profit'] > 0]
        losing_trades = trades_df[trades_df['profit'] < 0]

        num_total_trades = len(trades_df)
        num_winning_trades = len(winning_trades)
        num_losing_trades = len(losing_trades)

        win_rate = (num_winning_trades / num_total_trades) * 100 if num_total_trades > 0 else 0

        avg_win = winning_trades['profit'].mean() if num_winning_trades > 0 else 0
        avg_loss = losing_trades['profit'].mean() if num_losing_trades > 0 else 0

        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

        gross_profit = winning_trades['profit'].sum() if num_winning_trades > 0 else 0
        gross_loss = abs(losing_trades['profit'].sum()) if num_losing_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

        expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

        returns_series = trades_df.set_index('exit_date')['profit'].cumsum()
        net_profit = returns_series.iloc[-1] if not returns_series.empty else 0

        initial_capital = INITIAL_CAPITAL
        if not returns_series.empty:
            equity_curve = initial_capital + returns_series.fillna(0)
            peak = equity_curve.expanding(min_periods=1).max()
            drawdown = (equity_curve - peak) / peak
            max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0
        else:
            max_drawdown = 0

        net_profit_percent = (net_profit / initial_capital) * 100 if initial_capital != 0 else 0
        return_drawdown_ratio = (net_profit_percent / abs(max_drawdown)) if max_drawdown != 0 else np.nan

        streaks = []
        current_streak_type = None
        current_streak_length = 0

        for profit in trades_df['profit']:
            if profit > 0:
                if current_streak_type == 'win':
                    current_streak_length += 1
                else:
                    if current_streak_type is not None:
                        streaks.append({'type': current_streak_type, 'length': current_streak_length})
                    current_streak_type = 'win'
                    current_streak_length = 1
            elif profit < 0:
                if current_streak_type == 'loss':
                    current_streak_length += 1
                else:
                    if current_streak_type is not None:
                        streaks.append({'type': current_streak_type, 'length': current_streak_length})
                    current_streak_type = 'loss'
                    current_streak_length = 1
            else:
                if current_streak_type is not None:
                    streaks.append({'type': current_streak_type, 'length': current_streak_length})
                current_streak_type = None
                current_streak_length = 0
        if current_streak_type is not None:
            streaks.append({'type': current_streak_type, 'length': current_streak_length})

        winning_streaks = [s['length'] for s in streaks if s['type'] == 'win']
        losing_streaks = [s['length'] for s in streaks if s['type'] == 'loss']

        max_winning_streak = max(winning_streaks) if winning_streaks else 0
        max_losing_streak = max(losing_streaks) if losing_streaks else 0

        daily_returns = trades_df.set_index('exit_date')['profit'].resample('D').sum().fillna(0)
        annualized_returns = daily_returns.mean() * 252
        annualized_std_dev = daily_returns.std() * np.sqrt(252)

        sharpe_ratio = annualized_returns / annualized_std_dev if annualized_std_dev != 0 else np.nan

        downside_returns = daily_returns[daily_returns < 0]
        downside_std_dev = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
        sortino_ratio = annualized_returns / downside_std_dev if downside_std_dev != 0 else np.nan

        metrics = {
            'total_trades': num_total_trades,
            'winning_trades': num_winning_trades,
            'losing_trades': num_losing_trades,
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2) if not np.isnan(risk_reward_ratio) else np.nan,
            'profit_factor': round(profit_factor, 2) if not np.isnan(profit_factor) else np.nan,
            'expectancy': round(expectancy, 2),
            'net_profit': round(net_profit, 2),
            'net_profit_percent': round(net_profit_percent, 2),
            'max_drawdown': round(max_drawdown, 2),
            'return_drawdown_ratio': round(return_drawdown_ratio, 2) if not np.isnan(return_drawdown_ratio) else np.nan,
            'max_winning_streak': max_winning_streak,
            'max_losing_streak': max_losing_streak,
            'sharpe_ratio': round(sharpe_ratio, 2) if not np.isnan(sharpe_ratio) else np.nan,
            'sortino_ratio': round(sortino_ratio, 2) if not np.isnan(sortino_ratio) else np.nan
        }
    else:
        metrics = {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0,
            'avg_win': 0, 'avg_loss': 0, 'risk_reward_ratio': np.nan, 'profit_factor': np.nan,
            'expectancy': 0, 'net_profit': 0, 'net_profit_percent': 0, 'max_drawdown': 0,
            'return_drawdown_ratio': np.nan, 'max_winning_streak': 0, 'max_losing_streak': 0,
            'sharpe_ratio': np.nan, 'sortino_ratio': np.nan
        }
    return metrics

def calculate_atr_alternative(high, low, close, period=14, method='standard'):
    """
    Calculate ATR with different methods

    Parameters:
    high, low, close: Price arrays
    period: ATR period
    method: 'standard', 'wilder', 'simple'
    """
    if method == 'standard':
        # Standard True Range
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
        tr[0] = high[0] - low[0]

        # Simple moving average ATR
        atr = np.zeros_like(tr)
        atr[0] = tr[0]

        for i in range(1, len(tr)):
            atr[i] = (tr[i] + (period - 1) * atr[i-1]) / period
        return atr

    elif method == 'wilder':
        # Wilder's smoothing method (default)
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
        tr[0] = high[0] - low[0]

        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        alpha = 1 / period
        for i in range(1, len(tr)):
            atr[i] = (tr[i] * alpha) + (atr[i-1] * (1 - alpha))
        return atr

    elif method == 'simple':
        # Simple moving average of True Range
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
        tr[0] = high[0] - low[0]

        atr = np.zeros_like(tr)
        for i in range(len(tr)):
            if i < period - 1:
                atr[i] = np.mean(tr[:i+1])
            else:
                atr[i] = np.mean(tr[i-period+1:i+1])
        return atr

    else:
        raise ValueError("Method must be 'standard', 'wilder', or 'simple'")

def calculate_supertrend_optimized(df, period=10, multiplier=3.0, atr_method='wilder', suffix=""):
    """
    Calculate SuperTrend with optimized parameters and ATR methods

    Parameters:
    df: DataFrame with OHLC columns
    period: ATR period
    multiplier: SuperTrend multiplier
    atr_method: ATR calculation method ('standard', 'wilder', 'simple')
    suffix: Column name suffix
    """
    original_index = df.index
    df_reset = df.reset_index(drop=True)

    if not all(col in df_reset.columns for col in ["Open", "High", "Low", "Close"]):
        raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns")

    high = df_reset["High"].to_numpy()
    low = df_reset["Low"].to_numpy()
    close = df_reset["Close"].to_numpy()

    # Calculate ATR with specified method
    atr = calculate_atr_alternative(high, low, close, period, atr_method)

    # Calculate SuperTrend bands
    basic_upper_band = ((high + low) / 2) + (multiplier * atr)
    basic_lower_band = ((high + low) / 2) - (multiplier * atr)

    first_valid_atr_idx = np.where(~np.isnan(atr))[0][0] if np.any(~np.isnan(atr)) else -1

    trending_up_arr, trending_down_arr, direction_arr, supertrend_arr = _calculate_supertrend_numba(
        high, low, close, np.zeros_like(atr), atr, period, multiplier, basic_upper_band, basic_lower_band, first_valid_atr_idx
    )

    st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}_{atr_method}{suffix}"
    st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}_{atr_method}{suffix}"
    trending_up_col_name = f"trendingUp_{period}_{str(multiplier).replace('.', '_')}_{atr_method}{suffix}"
    trending_down_col_name = f"trendingDown_{period}_{str(multiplier).replace('.', '_')}_{atr_method}{suffix}"

    result_df = pd.DataFrame({
        st_col_name: supertrend_arr,
        st_dir_col_name: direction_arr,
        trending_up_col_name: trending_up_arr,
        trending_down_col_name: trending_down_arr
    }, index=original_index)

    return result_df

def optimize_supertrend_parameters(df, periods=None, multipliers=None, atr_methods=None, start_year=2019):
    """
    Optimize SuperTrend parameters through comprehensive backtesting

    Parameters:
    df: DataFrame with OHLC data
    periods: List of ATR periods to test
    multipliers: List of SuperTrend multipliers to test
    atr_methods: List of ATR calculation methods to test
    start_year: Start year for backtesting

    Returns:
    DataFrame with optimization results
    """
    if periods is None:
        periods = [26, 27, 28, 29, 30]

    if multipliers is None:
        multipliers = [0.9, 1.0, 1.2, 1.5, 2.0]

    if atr_methods is None:
        atr_methods = ['wilder', 'standard', 'simple']

    # Filter data for optimization period
    df_opt = df[df.index.year >= start_year].copy()

    optimization_results = []

    total_combinations = len(periods) * len(multipliers) * len(atr_methods)

    print(f"Testing {total_combinations} parameter combinations...")

    for i, period in enumerate(periods):
        for j, multiplier in enumerate(multipliers):
            for k, atr_method in enumerate(atr_methods):
                try:
                    # Calculate SuperTrend with current parameters
                    st_result = calculate_supertrend_optimized(
                        df_opt, period=period, multiplier=multiplier, atr_method=atr_method
                    )

                    # Get direction column name
                    st_dir_col = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}_{atr_method}"

                    df_with_st = pd.concat([df_opt, st_result], axis=1)

                    # Run strategy
                    st_dir = df_with_st[st_dir_col].to_numpy()
                    close_prices = df_with_st['Close'].to_numpy()

                    buy_entry, buy_exit, sell_entry, sell_exit = _run_strategy_numba(st_dir, close_prices)

                    # Calculate basic metrics
                    trades = []
                    current_buy_entry = None
                    current_sell_entry = None

                    for idx in range(len(df_with_st)):
                        if not pd.isna(buy_entry[idx]):
                            current_buy_entry = {'entry_price': buy_entry[idx], 'entry_date': df_with_st.index[idx]}
                        if not pd.isna(buy_exit[idx]) and current_buy_entry is not None:
                            profit = buy_exit[idx] - current_buy_entry['entry_price']
                            trades.append({
                                'type': 'buy',
                                'entry_date': current_buy_entry['entry_date'],
                                'exit_date': df_with_st.index[idx],
                                'entry_price': current_buy_entry['entry_price'],
                                'exit_price': buy_exit[idx],
                                'profit': profit
                            })
                            current_buy_entry = None

                        if not pd.isna(sell_entry[idx]):
                            current_sell_entry = {'entry_price': sell_entry[idx], 'entry_date': df_with_st.index[idx]}
                        if not pd.isna(sell_exit[idx]) and current_sell_entry is not None:
                            profit = current_sell_entry['entry_price'] - sell_exit[idx]
                            trades.append({
                                'type': 'sell',
                                'entry_date': current_sell_entry['entry_date'],
                                'exit_date': df_with_st.index[idx],
                                'entry_price': current_sell_entry['entry_price'],
                                'exit_price': sell_exit[idx],
                                'profit': profit
                            })
                            current_sell_entry = None

                    if trades:
                        trades_df = pd.DataFrame(trades)
                        total_trades = len(trades_df)
                        winning_trades = len(trades_df[trades_df['profit'] > 0])
                        win_rate = (winning_trades / total_trades) * 100
                        net_profit = trades_df['profit'].sum()
                        max_drawdown = 0

                        if total_trades > 0:
                            # Calculate returns for drawdown
                            returns_series = trades_df.set_index('exit_date')['profit'].cumsum()
                            if not returns_series.empty:
                                equity_curve = INITIAL_CAPITAL + returns_series.fillna(0)
                                peak = equity_curve.expanding(min_periods=1).max()
                                drawdown = (equity_curve - peak) / peak
                                max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0

                        optimization_results.append({
                            'period': period,
                            'multiplier': multiplier,
                            'atr_method': atr_method,
                            'total_trades': total_trades,
                            'win_rate': round(win_rate, 2),
                            'net_profit': round(net_profit, 2),
                            'max_drawdown': round(max_drawdown, 2),
                            'profit_factor': round(trades_df[trades_df['profit'] > 0]['profit'].sum() / abs(trades_df[trades_df['profit'] < 0]['profit'].sum()), 2) if len(trades_df[trades_df['profit'] < 0]) > 0 else float('inf')
                        })

                    # Progress indicator
                    combination_num = i * len(multipliers) * len(atr_methods) + j * len(atr_methods) + k + 1
                    print(f"Completed {combination_num}/{total_combinations} combinations")

                except Exception as e:
                    print(f"Error testing Period={period}, Multiplier={multiplier}, ATR={atr_method}: {e}")
                    continue

    if optimization_results:
        results_df = pd.DataFrame(optimization_results)

        # Sort by key metrics (you can customize this)
        results_df = results_df.sort_values(['net_profit', 'win_rate', 'max_drawdown'], ascending=[False, False, True])

        return results_df
    else:
        print("No valid results generated during optimization")
        return pd.DataFrame()

def create_sample_data():
    """
    Create sample OHLC data for testing when CSV file is not available
    """
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    # Generate sample price data with some trend and volatility
    np.random.seed(42)  # For reproducible results
    n = len(dates)

    # Create a trending price series
    trend = np.linspace(100, 150, n)
    noise = np.random.normal(0, 2, n)
    close_prices = trend + noise

    # Generate OHLC from close prices
    high_prices = close_prices + np.abs(np.random.normal(0, 1, n))
    low_prices = close_prices - np.abs(np.random.normal(0, 1, n))
    open_prices = np.random.uniform(low_prices, high_prices)

    sample_df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices
    }, index=dates)

    return sample_df

def load_data(csv_path=None, use_sample_data=False):
    """
    Load data from CSV file or create sample data

    Parameters:
    csv_path (str): Path to CSV file
    use_sample_data (bool): Whether to use sample data instead of CSV

    Returns:
    pd.DataFrame: DataFrame with OHLC data
    """
    if use_sample_data:
        print("Using sample data for demonstration...")
        return create_sample_data()

    try:
        if csv_path is None:
            csv_path = CSV_FILE_PATH

        df_original = pd.read_csv(csv_path)
        df_original = df_original.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
        df_original = df_original.rename(columns={'Date': 'DateTime'})
        df_original = df_original.set_index('DateTime')
        df_original.index = pd.to_datetime(df_original.index)
        df_original = df_original[~df_original.index.duplicated(keep='first')]

        print(f"Loaded data from {csv_path}")
        print(f"Data shape: {df_original.shape}")
        print(f"Date range: {df_original.index.min()} to {df_original.index.max()}")

        return df_original

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        print("Using sample data instead...")
        return create_sample_data()
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        print("Using sample data instead...")
        return create_sample_data()

def analyze_optimization_results(results_df, top_n=10):
    """
    Analyze optimization results and provide recommendations

    Parameters:
    results_df: DataFrame with optimization results
    top_n: Number of top results to display
    """
    if results_df.empty:
        print("No optimization results to analyze")
        return

    print(f"\n=== Top {top_n} Parameter Combinations ===")
    top_results = results_df.head(top_n)

    for i, row in top_results.iterrows():
        print(f"\nRank {i+1}:")
        print(f"  Period: {row['period']}")
        print(f"  Multiplier: {row['multiplier']}")
        print(f"  ATR Method: {row['atr_method']}")
        print(f"  Total Trades: {row['total_trades']}")
        print(f"  Win Rate: {row['win_rate']}%")
        print(f"  Net Profit: ${row['net_profit']:,.2f}")
        print(f"  Max Drawdown: {row['max_drawdown']:.2f}%")
        print(f"  Profit Factor: {row['profit_factor']:.2f}")

    # Find best parameters by different criteria
    print("\n=== Best Parameters by Criteria ===")

    # Best by net profit
    best_profit = results_df.loc[results_df['net_profit'].idxmax()]
    print(f"Best Net Profit: Period={best_profit['period']}, Multiplier={best_profit['multiplier']}, ATR={best_profit['atr_method']} (${best_profit['net_profit']:,.2f})")

    # Best by win rate
    best_winrate = results_df.loc[results_df['win_rate'].idxmax()]
    print(f"Best Win Rate: Period={best_winrate['period']}, Multiplier={best_winrate['multiplier']}, ATR={best_winrate['atr_method']} ({best_winrate['win_rate']:.2f}%)")

    # Best by lowest drawdown
    best_dd = results_df.loc[results_df['max_drawdown'].idxmin()]
    print(f"Best Max Drawdown: Period={best_dd['period']}, Multiplier={best_dd['multiplier']}, ATR={best_dd['atr_method']} ({best_dd['max_drawdown']:.2f}%)")

    # Best by profit factor
    best_pf = results_df.loc[results_df['profit_factor'].idxmax()]
    print(f"Best Profit Factor: Period={best_pf['period']}, Multiplier={best_pf['multiplier']}, ATR={best_pf['atr_method']} ({best_pf['profit_factor']:.2f})")

    # Balanced approach (good profit, decent win rate, reasonable drawdown)
    balanced = results_df[
        (results_df['net_profit'] > results_df['net_profit'].quantile(0.7)) &
        (results_df['win_rate'] > results_df['win_rate'].quantile(0.5)) &
        (results_df['max_drawdown'] < results_df['max_drawdown'].quantile(0.7))
    ]

    if not balanced.empty:
        balanced_pick = balanced.iloc[0]
        print(f"\nBalanced Pick: Period={balanced_pick['period']}, Multiplier={balanced_pick['multiplier']}, ATR={balanced_pick['atr_method']}")
        print(f"  Net Profit: ${balanced_pick['net_profit']:,.2f}")
        print(f"  Win Rate: {balanced_pick['win_rate']:.2f}%")
        print(f"  Max Drawdown: {balanced_pick['max_drawdown']:.2f}%")

    return top_results

if __name__ == "__main__":
    print("=== OHLC Supertrend Strategy & Optimization ===")

    # Load data
    df_original = load_data(use_sample_data=False)

    if df_original.empty:
        print("No data available for optimization")
        exit()

    # Filter for start year
    if df_original.index.year.min() < START_YEAR:
        df_original = df_original[df_original.index.year >= START_YEAR]
        print(f"Using data from {START_YEAR} onwards: {df_original.shape}")

    # Round price columns
    for col in COLUMNS_TO_ROUND:
        if col in df_original.columns:
            df_original[col] = df_original[col].round(DECIMAL_PLACES)

    # Run optimization
    print("\n=== Parameter Optimization ===")
    optimization_results = optimize_supertrend_parameters(
        df_original,
        periods=[21, 22, 23, 24, 25, 26, 27, 28],
        multipliers=[0.9, 1.0, 1.1, 1.2, 1.5],
        atr_methods=['wilder', 'standard'],
        start_year=START_YEAR
    )

    if not optimization_results.empty:
        # Save optimization results
        opt_results_path = 'ohlc_supertrend_optimization_results.csv'
        optimization_results.to_csv(opt_results_path, index=False)
        print(f"\nOptimization results saved to '{opt_results_path}'")

        # Analyze results
        top_params = analyze_optimization_results(optimization_results, top_n=5)

        # Test the best parameters
        if not top_params.empty:
            best_params = top_params.iloc[0]
            print(f"\n=== Testing Best Parameters ===")
            print(f"Best: Period={best_params['period']}, Multiplier={best_params['multiplier']}, ATR={best_params['atr_method']}")

            # Update default parameters
            best_period = int(best_params['period'])
            best_multiplier = best_params['multiplier']
            best_atr_method = best_params['atr_method']

            # Test with best parameters
            st_result = calculate_supertrend_optimized(
                df_original, period=best_period, multiplier=best_multiplier, atr_method=best_atr_method
            )

            st_dir_col = f"ST_Dir_{best_period}_{str(best_multiplier).replace('.', '_')}_{best_atr_method}"
            df_with_st = pd.concat([df_original, st_result], axis=1)

            st_dir = df_with_st[st_dir_col].to_numpy()
            close_prices = df_with_st['Close'].to_numpy()

            buy_entry, buy_exit, sell_entry, sell_exit = _run_strategy_numba(st_dir, close_prices)

            # Calculate comprehensive metrics
            trades = []
            current_buy_entry = None
            current_sell_entry = None

            for i in range(len(df_with_st)):
                if not pd.isna(buy_entry[i]):
                    current_buy_entry = {'entry_price': buy_entry[i], 'entry_date': df_with_st.index[i]}
                if not pd.isna(buy_exit[i]) and current_buy_entry is not None:
                    profit = buy_exit[i] - current_buy_entry['entry_price']
                    trades.append({
                        'type': 'buy',
                        'entry_date': current_buy_entry['entry_date'],
                        'exit_date': df_with_st.index[i],
                        'entry_price': current_buy_entry['entry_price'],
                        'exit_price': buy_exit[i],
                        'profit': profit
                    })
                    current_buy_entry = None

                if not pd.isna(sell_entry[i]):
                    current_sell_entry = {'entry_price': sell_entry[i], 'entry_date': df_with_st.index[i]}
                if not pd.isna(sell_exit[i]) and current_sell_entry is not None:
                    profit = current_sell_entry['entry_price'] - sell_exit[i]
                    trades.append({
                        'type': 'sell',
                        'entry_date': current_sell_entry['entry_date'],
                        'exit_date': df_with_st.index[i],
                        'entry_price': current_sell_entry['entry_price'],
                        'exit_price': sell_exit[i],
                        'profit': profit
                    })
                    current_sell_entry = None

            if trades:
                trades_df = pd.DataFrame(trades)

                # Save all trades to CSV file
                best_trades_csv_path = f'ohlc_supertrend_best_trades_{best_period}_{str(best_multiplier).replace(".", "_")}_{best_atr_method}.csv'
                trades_df.to_csv(best_trades_csv_path, index=False)
                print(f"\nAll trades saved to '{best_trades_csv_path}'")

                print("\n=== Optimized Strategy Results ===")
                total_trades = len(trades_df)
                winning_trades = len(trades_df[trades_df['profit'] > 0])
                win_rate = (winning_trades / total_trades) * 100
                net_profit = trades_df['profit'].sum()

                print(f"Total Trades: {total_trades}")
                print(f"Win Rate: {win_rate:.2f}%")
                print(f"Net Profit: ${net_profit:,.2f}")

                # Calculate additional metrics
                returns_series = trades_df.set_index('exit_date')['profit'].cumsum()
                if not returns_series.empty:
                    equity_curve = INITIAL_CAPITAL + returns_series.fillna(0)
                    peak = equity_curve.expanding(min_periods=1).max()
                    drawdown = (equity_curve - peak) / peak
                    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0
                    print(f"Max Drawdown: {max_drawdown:.2f}%")

                    profit_factor = trades_df[trades_df['profit'] > 0]['profit'].sum() / abs(trades_df[trades_df['profit'] < 0]['profit'].sum()) if len(trades_df[trades_df['profit'] < 0]) > 0 else float('inf')
                    print(f"Profit Factor: {profit_factor:.2f}")

    else:
        print("Optimization completed but no valid results found")

    # Run with original parameters for comparison
    print(f"\n=== Original Parameters (Period={ST_PERIOD}, Multiplier={ST_MULTIPLIER}) ===")
    supertrend_params = [(ST_PERIOD, ST_MULTIPLIER)]
    original_metrics = run_strategy_and_get_metrics(df_original.copy(), supertrend_params)

    print("Original Strategy Metrics:")
    for key, value in original_metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print(f"\nOptimization results saved to 'ohlc_supertrend_optimization_results.csv'")