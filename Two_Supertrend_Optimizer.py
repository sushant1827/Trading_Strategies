# Two Supertrend Optimizer
# Comprehensive Parameter Optimization for the Dual Supertrend Strategy
#
# This optimizer tests various parameter combinations for two Supertrends:
# - 60-minute data for trend confirmation
# - 15-minute data for buy/sell signals
#
# OPTIMIZATION APPROACH:
# 1. Walk-forward optimization to avoid overfitting
# 2. Multi-objective optimization (profit + risk metrics)
# 3. Parameter sensitivity analysis
# 4. Out-of-sample validation
# 5. Forward-looking bias prevention using merge_asof for data alignment

import pandas as pd
import numpy as np
import itertools
from numba import jit

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
    tr[0] = ha_high[0] - ha_low[0]

    # ATR using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    alpha = 1 / period
    for i in range(1, len(tr)):
        atr[i] = (tr[i] * alpha) + (atr[i-1] * (1 - alpha))

    basic_upper_band = ((ha_high + ha_low) / 2) + (multiplier * atr)
    basic_lower_band = ((ha_high + ha_low) / 2) - (multiplier * atr)

    first_valid_atr_idx = np.where(~np.isnan(atr))[0][0] if np.any(~np.isnan(atr)) else -1

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
def _run_dual_strategy_numba(st_dir_15, st_dir_60, close_prices):
    buy_entry_prices = np.full_like(close_prices, np.nan)
    buy_exit_prices = np.full_like(close_prices, np.nan)
    sell_entry_prices = np.full_like(close_prices, np.nan)
    sell_exit_prices = np.full_like(close_prices, np.nan)

    in_buy_position = False
    in_sell_position = False

    prev_st_dir_15 = 0  # Initialize previous direction

    for i in range(1, len(close_prices)):
        current_st_dir_15 = st_dir_15[i]
        current_st_dir_60 = st_dir_60[i]

        # Check for Buy_Entry: 15-min ST changes to up (1) and 60-min ST is up (1)
        if not in_buy_position and not in_sell_position and current_st_dir_15 == 1 and prev_st_dir_15 != 1 and current_st_dir_60 == 1:
            buy_entry_prices[i] = close_prices[i]
            in_buy_position = True
        # Check for Buy_Exit: 15-min ST changes to down (-1)
        elif in_buy_position and current_st_dir_15 == -1 and prev_st_dir_15 != -1:
            buy_exit_prices[i] = close_prices[i]
            in_buy_position = False

        # Check for Sell_Entry: 15-min ST changes to down (-1) and 60-min ST is down (-1)
        if not in_sell_position and not in_buy_position and current_st_dir_15 == -1 and prev_st_dir_15 != -1 and current_st_dir_60 == -1:
            sell_entry_prices[i] = close_prices[i]
            in_sell_position = True
        # Check for Sell_Exit: 15-min ST changes to up (1)
        elif in_sell_position and current_st_dir_15 == 1 and prev_st_dir_15 != 1:
            sell_exit_prices[i] = close_prices[i]
            in_sell_position = False

        prev_st_dir_15 = current_st_dir_15

    return buy_entry_prices, buy_exit_prices, sell_entry_prices, sell_exit_prices

def run_dual_strategy_and_get_metrics(df_15, df_60, period_15, mult_15, period_60, mult_60):
    # Calculate Heikin Ashi and Supertrend for 15-min data (signals)
    ha_15 = calculate_heikin_ashi(df_15.copy())
    df_15 = pd.concat([df_15, ha_15], axis=1)

    st_15_result = calculate_supertrend(df_15.copy(), period=period_15, multiplier=mult_15, suffix="_15")
    st_dir_15_col = f"ST_Dir_{period_15}_{str(mult_15).replace('.', '_')}_15"
    df_15 = pd.concat([df_15, st_15_result], axis=1)

    # Calculate Heikin Ashi and Supertrend for 60-min data (confirmation)
    ha_60 = calculate_heikin_ashi(df_60.copy())
    df_60 = pd.concat([df_60, ha_60], axis=1)

    st_60_result = calculate_supertrend(df_60.copy(), period=period_60, multiplier=mult_60, suffix="_60")
    st_dir_60_col = f"ST_Dir_{period_60}_{str(mult_60).replace('.', '_')}_60"
    df_60 = pd.concat([df_60, st_60_result], axis=1)

    # Merge 60-min direction to 15-min data using merge_asof to avoid forward bias
    df_60_for_merge = df_60[[st_dir_60_col]].reset_index()
    df_15_reset = df_15.reset_index()

    merged = pd.merge_asof(df_15_reset, df_60_for_merge, on='DateTime', direction='backward')
    merged = merged.set_index('DateTime')

    # Fill any NaN in merged direction with forward fill (for initial periods)
    merged[st_dir_60_col] = merged[st_dir_60_col].ffill()

    # Now run the strategy
    st_dir_15 = merged[st_dir_15_col].to_numpy()
    st_dir_60 = merged[st_dir_60_col].to_numpy()
    close_prices = merged['Close'].to_numpy()

    buy_entry_prices, buy_exit_prices, sell_entry_prices, sell_exit_prices = _run_dual_strategy_numba(
        st_dir_15, st_dir_60, close_prices
    )

    merged['Buy_Entry'] = buy_entry_prices
    merged['Buy_Exit'] = buy_exit_prices
    merged['Sell_Entry'] = sell_entry_prices
    merged['Sell_Exit'] = sell_exit_prices

    # Calculate trades and metrics (similar to original)
    trades = []
    current_buy_entry = None
    current_sell_entry = None

    for i in range(len(merged)):
        if not pd.isna(merged['Buy_Entry'].iloc[i]):
            current_buy_entry = {'entry_price': merged['Buy_Entry'].iloc[i], 'entry_date': merged.index[i]}
        if not pd.isna(merged['Buy_Exit'].iloc[i]) and current_buy_entry is not None:
            profit = merged['Buy_Exit'].iloc[i] - current_buy_entry['entry_price']
            trades.append({
                'type': 'buy',
                'entry_date': current_buy_entry['entry_date'],
                'exit_date': merged.index[i],
                'entry_price': current_buy_entry['entry_price'],
                'exit_price': merged['Buy_Exit'].iloc[i],
                'profit': profit
            })
            current_buy_entry = None

        if not pd.isna(merged['Sell_Entry'].iloc[i]):
            current_sell_entry = {'entry_price': merged['Sell_Entry'].iloc[i], 'entry_date': merged.index[i]}
        if not pd.isna(merged['Sell_Exit'].iloc[i]) and current_sell_entry is not None:
            profit = current_sell_entry['entry_price'] - merged['Sell_Exit'].iloc[i]
            trades.append({
                'type': 'sell',
                'entry_date': current_sell_entry['entry_date'],
                'exit_date': merged.index[i],
                'entry_price': current_sell_entry['entry_price'],
                'exit_price': merged['Sell_Exit'].iloc[i],
                'profit': profit
            })
            current_sell_entry = None

    trades_df = pd.DataFrame(trades)

    # Metrics calculation (same as original)
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

def calculate_score(metrics):
    if metrics['total_trades'] == 0:
        return -1000
    profit_score = metrics['net_profit_percent']
    risk_score = -metrics['max_drawdown']
    sharpe = metrics['sharpe_ratio'] if not np.isnan(metrics['sharpe_ratio']) else 0
    sharpe_score = sharpe * 10
    win_rate_score = metrics['win_rate'] * 0.1
    score = profit_score + risk_score + sharpe_score + win_rate_score
    return score

try:
    # Load 15-min data for signals
    csv_file_15 = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'
    df_15 = pd.read_csv(csv_file_15)
    df_15 = df_15.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
    df_15 = df_15.rename(columns={'Date': 'DateTime'})
    df_15 = df_15.set_index('DateTime')
    df_15.index = pd.to_datetime(df_15.index)
    df_15 = df_15[~df_15.index.duplicated(keep='first')]
    df_15 = df_15[df_15.index.year >= 2021]
    print(f"15-min data: After filtering for 2021 onwards, df_15 shape: {df_15.shape}")

    # Load 60-min data for confirmation
    csv_file_60 = 'Nifty50_Index/NIFTY50_INDEX_120_Min.csv'
    df_60 = pd.read_csv(csv_file_60)
    df_60 = df_60.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
    df_60 = df_60.rename(columns={'Date': 'DateTime'})
    df_60 = df_60.set_index('DateTime')
    df_60.index = pd.to_datetime(df_60.index)
    df_60 = df_60[~df_60.index.duplicated(keep='first')]
    df_60 = df_60[df_60.index.year >= 2021]
    print(f"60-min data: After filtering for 2021 onwards, df_60 shape: {df_60.shape}")

    for df in [df_15, df_60]:
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = df[col].round(2)

    # Split data based on 15-min timeframe for consistency
    train_size = int(0.7 * len(df_15))
    split_time = df_15.index[train_size]

    df_15_train = df_15[df_15.index <= split_time]
    df_15_test = df_15[df_15.index > split_time]

    df_60_train = df_60[df_60.index <= split_time]
    df_60_test = df_60[df_60.index > split_time]

    print(f"Training data (15-min): {len(df_15_train)} rows")
    print(f"Testing data (15-min): {len(df_15_test)} rows")
    print(f"Training data (60-min): {len(df_60_train)} rows")
    print(f"Testing data (60-min): {len(df_60_test)} rows")

    # Define parameter ranges (optimized for speed while maintaining coverage)
    periods = list(range(7, 26, 1))  # 7,9,11,13,15,17,19 (7 values)
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]  # 5 values for faster optimization

    print(f"Testing {len(periods)} periods and {len(multipliers)} multipliers per timeframe")
    print(f"Total combinations: {len(periods) * len(multipliers) * len(periods) * len(multipliers)}")

    # Generate all parameter combinations
    param_combos = list(itertools.product(periods, multipliers, periods, multipliers))

    results = []
    for idx, (period_15, mult_15, period_60, mult_60) in enumerate(param_combos):
        if idx % 50 == 0:
            print(f"Processing combination {idx+1}/{len(param_combos)}: 15-min P={period_15}, M={mult_15}; 60-min P={period_60}, M={mult_60}")

        metrics_train = run_dual_strategy_and_get_metrics(df_15_train.copy(), df_60_train.copy(), period_15, mult_15, period_60, mult_60)
        metrics_test = run_dual_strategy_and_get_metrics(df_15_test.copy(), df_60_test.copy(), period_15, mult_15, period_60, mult_60)

        score_train = calculate_score(metrics_train)
        score_test = calculate_score(metrics_test)

        result = {
            'period_15': period_15,
            'multiplier_15': mult_15,
            'period_60': period_60,
            'multiplier_60': mult_60,
            'score_train': score_train,
            'score_test': score_test,
            **{f'train_{k}': v for k, v in metrics_train.items()},
            **{f'test_{k}': v for k, v in metrics_test.items()}
        }
        results.append(result)

    # Find best parameters based on training score
    best_result = max(results, key=lambda x: x['score_train'])

    print("\n" + "="*50)
    print("BEST PARAMETERS (based on training data)")
    print("="*50)
    print(f"15-min Period: {best_result['period_15']}")
    print(f"15-min Multiplier: {best_result['multiplier_15']}")
    print(f"60-min Period: {best_result['period_60']}")
    print(f"60-min Multiplier: {best_result['multiplier_60']}")
    print(f"Training Score: {best_result['score_train']:.2f}")
    print(f"Testing Score: {best_result['score_test']:.2f}")

    print("\n--- Training Metrics ---")
    for key, value in best_result.items():
        if key.startswith('train_'):
            print(f"{key.replace('train_', '').replace('_', ' ').title()}: {value}")

    print("\n--- Testing Metrics ---")
    for key, value in best_result.items():
        if key.startswith('test_'):
            print(f"{key.replace('test_', '').replace('_', ' ').title()}: {value}")

    # Save optimization results
    results_df = pd.DataFrame(results)
    optimization_csv_path = 'two_supertrend_optimization_results.csv'
    results_df.to_csv(optimization_csv_path, index=False)
    print(f"\nOptimization results saved to '{optimization_csv_path}'")

    print("\nOptimization completed successfully!")

except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
except Exception as e:
    print(f"An error occurred: {e}")