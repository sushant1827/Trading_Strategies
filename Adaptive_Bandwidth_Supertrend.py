import pandas as pd
import numpy as np
import itertools
from numba import jit

# Define the path to the CSV file
csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'
# csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_60_Min.csv'
# csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_120_Min.csv'

# Define kernel functions as standalone Numba jitted functions
@jit(nopython=True)
def _kernel_func_numba(source: float, bw: float, k_type: str) -> float:
    if k_type == "Gaussian":
        return np.exp(-np.power(source / bw, 2) / 2) / np.sqrt(2 * np.pi)
    elif k_type == "Epanechnikov":
        abs_val = abs(source / bw)
        return (3/4) * (1 - np.power(abs_val, 2)) if abs_val <= 1 else 0.0
    elif k_type == "Laplace":
        return (1 / (2 * bw)) * np.exp(-abs(source / bw))
    else:
        return 0.0  # Default to 0 for unsupported kernels

@jit(nopython=True)
def calculate_adaptive_kernel_atr_numba(tr: np.array, base_bandwidth: int, distance_factor: float, 
                                       supertrend_line: np.array, close_prices: np.array, 
                                       atr_values: np.array, kernel_type: str) -> np.array:
    """
    Calculate ATR using adaptive kernel regression where bandwidth is dynamically adjusted
    based on distance from SuperTrend line
    
    Parameters:
    tr (np.array): True Range values
    base_bandwidth (int): Base bandwidth for kernel smoothing
    distance_factor (float): Factor to multiply distance for bandwidth adjustment
    supertrend_line (np.array): SuperTrend line values
    close_prices (np.array): Close prices
    atr_values (np.array): ATR values for distance calculation
    kernel_type (str): Type of kernel to apply ("Gaussian", "Epanechnikov", "Laplace")
    
    Returns:
    np.array: Adaptive kernel-smoothed ATR values
    """
    n = len(tr)
    kernel_atr = np.zeros(n)
    
    for i in range(n):
        # Calculate distance-based adaptive bandwidth
        if atr_values[i] > 0 and not np.isnan(supertrend_line[i]):
            distance = abs(close_prices[i] - supertrend_line[i]) / atr_values[i]
            adaptive_bandwidth = base_bandwidth * (1 + distance_factor * distance)
            # Ensure bandwidth stays within reasonable bounds
            adaptive_bandwidth = max(5, min(100, adaptive_bandwidth))
        else:
            adaptive_bandwidth = float(base_bandwidth)
        
        weights = []
        weighted_tr = []
        sum_weights = 0.0
        
        # Use standard non-repaint calculation with adaptive bandwidth
        start_idx = max(0, i - int(adaptive_bandwidth * 2))  # Use 2x bandwidth for lookback
        
        # Calculate weights for current position using only past and current data
        for j in range(start_idx, i + 1):
            diff = float(i - j)
            weight = _kernel_func_numba(diff, adaptive_bandwidth, kernel_type)
            weights.append(weight)
            weighted_tr.append(tr[j] * weight)
            sum_weights += weight
        
        # Calculate kernel-smoothed ATR value
        if sum_weights > 0:
            kernel_atr[i] = sum(weighted_tr) / sum_weights
        else:
            kernel_atr[i] = tr[i]  # Fallback to raw TR if no weights
    
    return kernel_atr

@jit(nopython=True)
def _calculate_supertrend_numba(ha_high, ha_low, ha_close, tr, atr, period, multiplier, basic_upper_band, basic_lower_band, first_valid_atr_idx):
    trending_up = np.empty_like(ha_close)
    trending_down = np.empty_like(ha_close)
    direction = np.empty_like(ha_close, dtype=np.int32)
    supertrend = np.empty_like(ha_close)

    if first_valid_atr_idx is None or first_valid_atr_idx >= len(ha_close):
        return trending_up, trending_down, direction, supertrend

    trending_up[first_valid_atr_idx] = basic_lower_band[first_valid_atr_idx]
    trending_down[first_valid_atr_idx] = basic_upper_band[first_valid_atr_idx]

    if ha_close[first_valid_atr_idx] > trending_down[first_valid_atr_idx]:
        direction[first_valid_atr_idx] = 1
    elif ha_close[first_valid_atr_idx] < trending_up[first_valid_atr_idx]:
        direction[first_valid_atr_idx] = -1
    else:
        direction[first_valid_atr_idx] = 1

    supertrend[first_valid_atr_idx] = trending_up[first_valid_atr_idx] if direction[first_valid_atr_idx] == 1 else trending_down[first_valid_atr_idx]

    for i in range(first_valid_atr_idx + 1, len(ha_close)):
        prev_ha_close = ha_close[i - 1]
        prev_trending_up = trending_up[i - 1]
        prev_trending_down = trending_down[i - 1]
        prev_direction = direction[i - 1]

        current_basic_upper_band = basic_upper_band[i]
        current_basic_lower_band = basic_lower_band[i]
        current_ha_close = ha_close[i]

        if prev_ha_close > prev_trending_up:
            trending_up[i] = max(current_basic_lower_band, prev_trending_up)
        else:
            trending_up[i] = current_basic_lower_band

        if prev_ha_close < prev_trending_down:
            trending_down[i] = min(current_basic_upper_band, prev_trending_down)
        else:
            trending_down[i] = current_basic_upper_band

        if current_ha_close > trending_down[i - 1]:
            direction[i] = 1
        elif current_ha_close < trending_up[i - 1]:
            direction[i] = -1
        else:
            direction[i] = prev_direction

        supertrend[i] = trending_up[i] if direction[i] == 1 else trending_down[i]
    
    return trending_up, trending_down, direction, supertrend

def calculate_adaptive_bandwidth_supertrend(df, period=10, multiplier=3.0, base_bandwidth=14, 
                                          distance_factor=1.0, kernel_type="Gaussian", suffix=""):
    """
    Calculate SuperTrend with adaptive bandwidth that adjusts based on distance from SuperTrend line
    
    Parameters:
    df (pd.DataFrame): DataFrame with HA columns
    period (int): ATR period
    multiplier (float): ATR multiplier for SuperTrend bands
    base_bandwidth (int): Base bandwidth for kernel smoothing
    distance_factor (float): Factor to adjust bandwidth based on distance from SuperTrend
    kernel_type (str): Type of kernel to apply
    suffix (str): Suffix for column names
    """
    original_index = df.index
    df_reset = df.reset_index(drop=True)

    if not all(col in df_reset.columns for col in ["HA_Open", "HA_High", "HA_Low", "HA_Close"]):
        raise ValueError("DataFrame must contain 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close' columns for Supertrend calculation.")

    ha_high = df_reset["HA_High"].to_numpy()
    ha_low = df_reset["HA_Low"].to_numpy()
    ha_close = df_reset["HA_Close"].to_numpy()

    # Calculate True Range based on HA candles
    tr = np.maximum(ha_high - ha_low, np.abs(ha_high - np.roll(ha_close, 1)))
    tr = np.maximum(tr, np.abs(ha_low - np.roll(ha_close, 1)))
    tr[0] = ha_high[0] - ha_low[0]  # Correct TR for the first element

    # First calculate regular ATR for distance calculation
    atr_regular = np.zeros_like(tr)
    atr_regular[0] = tr[0]  # Initialize first ATR
    alpha = 1 / period
    for i in range(1, len(tr)):
        atr_regular[i] = (tr[i] * alpha) + (atr_regular[i-1] * (1 - alpha))

    # Initialize with base bandwidth ATR for first iteration
    kernel_atr = calculate_adaptive_kernel_atr_numba(tr, base_bandwidth, distance_factor, 
                                                   np.zeros_like(ha_close), ha_close, 
                                                   atr_regular, kernel_type)

    # Iteratively calculate SuperTrend and adaptive ATR
    basic_upper_band = ((ha_high + ha_low) / 2) + (multiplier * kernel_atr)
    basic_lower_band = ((ha_high + ha_low) / 2) - (multiplier * kernel_atr)

    first_valid_atr_idx = np.where(~np.isnan(kernel_atr))[0][0] if np.any(~np.isnan(kernel_atr)) else -1

    trending_up_arr, trending_down_arr, direction_arr, supertrend_arr = _calculate_supertrend_numba(
        ha_high, ha_low, ha_close, tr, kernel_atr, period, multiplier, basic_upper_band, basic_lower_band, first_valid_atr_idx
    )

    # Recalculate with actual SuperTrend line for better adaptive bandwidth
    kernel_atr = calculate_adaptive_kernel_atr_numba(tr, base_bandwidth, distance_factor, 
                                                   supertrend_arr, ha_close, atr_regular, kernel_type)

    # Final calculation with improved adaptive ATR
    basic_upper_band = ((ha_high + ha_low) / 2) + (multiplier * kernel_atr)
    basic_lower_band = ((ha_high + ha_low) / 2) - (multiplier * kernel_atr)

    trending_up_arr, trending_down_arr, direction_arr, supertrend_arr = _calculate_supertrend_numba(
        ha_high, ha_low, ha_close, tr, kernel_atr, period, multiplier, basic_upper_band, basic_lower_band, first_valid_atr_idx
    )

    st_col_name = f"Adaptive_ST_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_base{base_bandwidth}_df{str(distance_factor).replace('.', '_')}{suffix}"
    st_dir_col_name = f"Adaptive_ST_Dir_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_base{base_bandwidth}_df{str(distance_factor).replace('.', '_')}{suffix}"
    trending_up_col_name = f"adaptive_trendingUp_HA_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_base{base_bandwidth}_df{str(distance_factor).replace('.', '_')}{suffix}"
    trending_down_col_name = f"adaptive_trendingDown_HA_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_base{base_bandwidth}_df{str(distance_factor).replace('.', '_')}{suffix}"

    result_df = pd.DataFrame({
        st_col_name: supertrend_arr,
        st_dir_col_name: direction_arr,
        trending_up_col_name: trending_up_arr,
        trending_down_col_name: trending_down_arr
    }, index=original_index)

    return result_df

@jit(nopython=True)
def _calculate_heikin_ashi_numba(open_prices, high_prices, low_prices, close_prices):
    ha_close = (open_prices + high_prices + low_prices + close_prices) / 4
    ha_open = np.zeros_like(open_prices, dtype=np.float64)
    ha_high = np.zeros_like(high_prices, dtype=np.float64)
    ha_low = np.zeros_like(low_prices, dtype=np.float64)

    if len(open_prices) > 0:
        ha_open[0] = open_prices[0]

    for i in range(1, len(open_prices)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2

    ha_high = np.maximum(high_prices, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(low_prices, np.minimum(ha_open, ha_close))

    return ha_open, ha_high, ha_low, ha_close

def calculate_heikin_ashi(df):
    if not all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns for Heikin Ashi calculation.")

    open_prices = df["Open"].to_numpy()
    high_prices = df["High"].to_numpy()
    low_prices = df["Low"].to_numpy()
    close_prices = df["Close"].to_numpy()

    ha_open, ha_high, ha_low, ha_close = _calculate_heikin_ashi_numba(open_prices, high_prices, low_prices, close_prices)

    ha_df = pd.DataFrame(
        {
            "HA_Open": ha_open,
            "HA_High": ha_high,
            "HA_Low": ha_low,
            "HA_Close": ha_close,
        },
        index=df.index,
    )

    return ha_df

@jit(nopython=True)
def _run_strategy_numba(st_dir, close_prices):
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

def run_strategy_and_get_metrics(df_original, supertrend_params_list, adaptive_params=None):
    df = df_original.copy()

    # Calculate Heikin Ashi
    ha_df = calculate_heikin_ashi(df.copy())
    df = pd.concat([df, ha_df], axis=1)

    st_direction_cols = []

    # Only use the first (and only) Supertrend parameter
    period, multiplier = supertrend_params_list[0]
    suffix = "_1"
    
    # Use adaptive bandwidth version if adaptive_params provided, otherwise use regular
    if adaptive_params:
        base_bandwidth, distance_factor, kernel_type = adaptive_params
        st_result_ha = calculate_adaptive_bandwidth_supertrend(
            df.copy(),
            period=period,
            multiplier=multiplier,
            base_bandwidth=base_bandwidth,
            distance_factor=distance_factor,
            kernel_type=kernel_type,
            suffix=suffix
        )
        st_col_name = f"Adaptive_ST_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_base{base_bandwidth}_df{str(distance_factor).replace('.', '_')}{suffix}"
        st_dir_col_name = f"Adaptive_ST_Dir_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_base{base_bandwidth}_df{str(distance_factor).replace('.', '_')}{suffix}"
        trending_up_col_name = f"adaptive_trendingUp_HA_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_base{base_bandwidth}_df{str(distance_factor).replace('.', '_')}{suffix}"
        trending_down_col_name = f"adaptive_trendingDown_HA_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_base{base_bandwidth}_df{str(distance_factor).replace('.', '_')}{suffix}"
    else:
        # Fallback to regular supertrend for comparison
        st_result_ha = calculate_supertrend(
            df.copy(),
            period=period,
            multiplier=multiplier,
            suffix=suffix
        )
        st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        trending_up_col_name = f"trendingUp_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        trending_down_col_name = f"trendingDown_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"

    st_direction_cols.append(st_dir_col_name)

    df = pd.concat([df, st_result_ha], axis=1)

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

        initial_capital = 100000
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

# Copy the regular calculate_supertrend function from One_Supertrend.py for comparison
def calculate_supertrend(df, period=10, multiplier=3.0, suffix=""):
    original_index = df.index
    df_reset = df.reset_index(drop=True)

    if not all(col in df_reset.columns for col in ["HA_Open", "HA_High", "HA_Low", "HA_Close"]):
        raise ValueError("DataFrame must contain 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close' columns for Supertrend calculation.")

    ha_high = df_reset["HA_High"].to_numpy()
    ha_low = df_reset["HA_Low"].to_numpy()
    ha_close = df_reset["HA_Close"].to_numpy()

    # Calculate True Range based on HA candles
    tr = np.maximum(ha_high - ha_low, np.abs(ha_high - np.roll(ha_close, 1)))
    tr = np.maximum(tr, np.abs(ha_low - np.roll(ha_close, 1)))
    tr[0] = ha_high[0] - ha_low[0]  # Correct TR for the first element

    # ATR using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]  # Initialize first ATR
    alpha = 1 / period
    for i in range(1, len(tr)):
        atr[i] = (tr[i] * alpha) + (atr[i-1] * (1 - alpha))

    basic_upper_band = ((ha_high + ha_low) / 2) + (multiplier * atr)
    basic_lower_band = ((ha_high + ha_low) / 2) - (multiplier * atr)

    first_valid_atr_idx = np.where(~np.isnan(atr))[0][0] if np.any(~np.isnan(atr)) else -1  # Use -1 if no valid index

    trending_up_arr, trending_down_arr, direction_arr, supertrend_arr = _calculate_supertrend_numba(
        ha_high, ha_low, ha_close, tr, atr, period, multiplier, basic_upper_band, basic_lower_band, first_valid_atr_idx
    )

    st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    trending_up_col_name = f"trendingUp_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    trending_down_col_name = f"trendingDown_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"

    result_df = pd.DataFrame({
        st_col_name: supertrend_arr,
        st_dir_col_name: direction_arr,
        trending_up_col_name: trending_up_arr,
        trending_down_col_name: trending_down_arr
    }, index=original_index)

    return result_df

try:
    df_original = pd.read_csv(csv_file_path)
    df_original = df_original.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
    df_original = df_original.rename(columns={'Date': 'DateTime'})
    df_original = df_original.set_index('DateTime')
    df_original.index = pd.to_datetime(df_original.index)
    df_original = df_original[~df_original.index.duplicated(keep='first')]
    print(f"After dropping duplicates, df_original index is unique: {df_original.index.is_unique}")
    df_original = df_original[df_original.index.year >= 2021]
    print(f"After filtering for 2021 onwards, df_original shape: {df_original.shape}")

    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df_original.columns:
            df_original[col] = df_original[col].round(2)

    # Test different adaptive bandwidth parameters
    adaptive_configs = [
        (30, 1.0, "Gaussian"),
        (40, 1.5, "Epanechnikov"), 
        (35, 0.8, "Laplace")
    ]
    
    results_list = []
    
    # Use fixed Supertrend parameter: Period=21, Multiplier=0.9 for single indicator
    fixed_supertrend_params = [(26, 0.9)]
    print("Using fixed Supertrend parameter: ST=(21, 0.9)")
    
    # Test regular Supertrend for comparison
    print("\n=== Regular Supertrend ===")
    metrics_regular = run_strategy_and_get_metrics(df_original.copy(), fixed_supertrend_params)
    print("Regular Supertrend Metrics:")
    for key, value in metrics_regular.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    results_list.append({
        'Type': 'Regular',
        'ST_Period': 26,
        'ST_Multiplier': 0.9,
        'Kernel_Type': 'None',
        'Base_Bandwidth': 0,
        'Distance_Factor': 0,
        **metrics_regular
    })
    
    # Test Adaptive Bandwidth Supertrend with different configurations
    for base_bandwidth, distance_factor, kernel_type in adaptive_configs:
        print(f"\n=== Adaptive Bandwidth Supertrend: {kernel_type} (base_bw={base_bandwidth}, df={distance_factor}) ===")
        adaptive_params = (base_bandwidth, distance_factor, kernel_type)
        metrics_adaptive = run_strategy_and_get_metrics(df_original.copy(), fixed_supertrend_params, adaptive_params)
        print(f"Adaptive Bandwidth Supertrend Metrics ({kernel_type}, base_bw={base_bandwidth}, df={distance_factor}):")
        for key, value in metrics_adaptive.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        results_list.append({
            'Type': 'Adaptive',
            'ST_Period': 21,
            'ST_Multiplier': 0.9,
            'Kernel_Type': kernel_type,
            'Base_Bandwidth': base_bandwidth,
            'Distance_Factor': distance_factor,
            **metrics_adaptive
        })
    
    # Save all results to CSV
    results_df = pd.DataFrame(results_list)
    output_csv_path = 'adaptive_bandwidth_supertrend_results.csv'
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nResults saved to '{output_csv_path}'")

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
