# McGinley Dynamic Supertrend Optimizer
# ---------------------------------------------------------------------------
# IMPORTANT (live realism notes - prefer McGinley_Live_Realistic_Backtest.py):
# 1) This file historically filled on the SAME bar close as the signal (look-ahead
#    relative to live: you only know ST_Dir after the close, so fill next open).
# 2) sell_pe / sell_ce PnL both used (exit - entry) on the INDEX. That is NOT
#    option premium PnL. CE short on a falling index was economically wrong as modeled.
# 3) No brokerage / STT / fees were applied; optimistic vs live India costs.
# 4) Single 70/30 split + max train score = in-sample cherry pick, weak OOS design.
# 5) ATR threshold compared raw ATR points to 0/10/20/30 without % scaling.
#
# Use PyScripts/McGinley_Live_Realistic_Backtest.py for:
#   next-open fills, Rs 250 RT costs, vectorbt trades, walk-forward OOS,
#   and train-only mistake memory.
# ---------------------------------------------------------------------------
# Parameters: ATR period, base multiplier, rolling period, McGinley length, ATR threshold

import pandas as pd
import numpy as np
import itertools
from numba import jit
import concurrent.futures
import multiprocessing
import time

# Define the path to the CSV file
csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_60_Min.csv'

# Live-cost floor on Rs 1L capital (brokerage+taxes+modest slippage bundled)
ROUND_TRIP_COST_INR = 250.0
# If True: signal on bar i close, fill at bar i+1 open (live-safe). If False: legacy same-bar close fill.
USE_NEXT_OPEN_FILL = True

@jit(nopython=True)
def _calculate_supertrend_numba(ha_high, ha_low, mg, tr, atr, period, multipliers, basic_upper_band, basic_lower_band, first_valid_atr_idx):
    trending_up = np.empty_like(mg)
    trending_down = np.empty_like(mg)
    direction = np.empty_like(mg, dtype=np.int32)
    supertrend = np.empty_like(mg)

    if first_valid_atr_idx is None or first_valid_atr_idx >= len(mg):
        return trending_up, trending_down, direction, supertrend

    trending_up[first_valid_atr_idx] = basic_lower_band[first_valid_atr_idx]
    trending_down[first_valid_atr_idx] = basic_upper_band[first_valid_atr_idx]

    if mg[first_valid_atr_idx] > trending_down[first_valid_atr_idx]:
        direction[first_valid_atr_idx] = 1
    elif mg[first_valid_atr_idx] < trending_up[first_valid_atr_idx]:
        direction[first_valid_atr_idx] = -1
    else:
        direction[first_valid_atr_idx] = 1

    supertrend[first_valid_atr_idx] = trending_up[first_valid_atr_idx] if direction[first_valid_atr_idx] == 1 else trending_down[first_valid_atr_idx]

    for i in range(first_valid_atr_idx + 1, len(mg)):
        prev_mg = mg[i - 1]
        prev_trending_up = trending_up[i - 1]
        prev_trending_down = trending_down[i - 1]
        prev_direction = direction[i - 1]

        current_basic_upper_band = basic_upper_band[i]
        current_basic_lower_band = basic_lower_band[i]
        current_mg = mg[i]

        if prev_mg > prev_trending_up:
            trending_up[i] = max(current_basic_lower_band, prev_trending_up)
        else:
            trending_up[i] = current_basic_lower_band

        if prev_mg < prev_trending_down:
            trending_down[i] = min(current_basic_upper_band, prev_trending_down)
        else:
            trending_down[i] = current_basic_upper_band

        if current_mg > trending_down[i - 1]:
            direction[i] = 1
        elif current_mg < trending_up[i - 1]:
            direction[i] = -1
        else:
            direction[i] = prev_direction

        supertrend[i] = trending_up[i] if direction[i] == 1 else trending_down[i]

    return trending_up, trending_down, direction, supertrend

def calculate_supertrend(df, period=14, base_multiplier=2.0, rolling_period=20, mcginley_length=14, suffix=""):
    original_index = df.index
    df_reset = df.reset_index(drop=True)

    if not all(col in df_reset.columns for col in ["HA_Open", "HA_High", "HA_Low", "HA_Close"]):
        raise ValueError("DataFrame must contain 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close' columns for Supertrend calculation.")

    ha_high = df_reset["HA_High"].to_numpy()
    ha_low = df_reset["HA_Low"].to_numpy()
    ha_close = df_reset["HA_Close"].to_numpy()

    # Calculate McGinley Dynamic
    mg = _calculate_mcginley_numba(ha_close, mcginley_length)

    # Calculate True Range based on HA candles
    tr = np.maximum(ha_high - ha_low, np.abs(ha_high - np.roll(ha_close, 1)))
    tr = np.maximum(tr, np.abs(ha_low - np.roll(ha_close, 1)))
    tr[0] = ha_high[0] - ha_low[0]

    # ATR using Wilder's smoothing
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    alpha = 1 / period
    for i in range(1, len(tr)):
        atr[i] = (tr[i] * alpha) + (atr[i-1] * (1 - alpha))

    # Rolling ATR for normalization
    rolling_atr = np.zeros_like(atr)
    rolling_atr[0] = atr[0]
    beta = 2 / (rolling_period + 1)
    for i in range(1, len(atr)):
        rolling_atr[i] = (atr[i] * beta) + (rolling_atr[i-1] * (1 - beta))

    # Dynamic multiplier
    multipliers = base_multiplier * (atr / rolling_atr)
    multipliers = np.clip(multipliers, 0.5, 5.0)

    basic_upper_band = ((ha_high + ha_low) / 2) + (multipliers * atr)
    basic_lower_band = ((ha_high + ha_low) / 2) - (multipliers * atr)

    first_valid_atr_idx = np.where(~np.isnan(atr))[0][0] if np.any(~np.isnan(atr)) else -1

    trending_up_arr, trending_down_arr, direction_arr, supertrend_arr = _calculate_supertrend_numba(
        ha_high, ha_low, mg, tr, atr, period, multipliers, basic_upper_band, basic_lower_band, first_valid_atr_idx
    )

    st_col_name = f"ST_{period}_{str(base_multiplier).replace('.', '_')}_{rolling_period}_{mcginley_length}{suffix}"
    st_dir_col_name = f"ST_Dir_{period}_{str(base_multiplier).replace('.', '_')}_{rolling_period}_{mcginley_length}{suffix}"
    trending_up_col_name = f"trendingUp_HA_{period}_{str(base_multiplier).replace('.', '_')}_{rolling_period}_{mcginley_length}{suffix}"
    trending_down_col_name = f"trendingDown_HA_{period}_{str(base_multiplier).replace('.', '_')}_{rolling_period}_{mcginley_length}{suffix}"

    result_df = pd.DataFrame({
        st_col_name: supertrend_arr,
        st_dir_col_name: direction_arr,
        trending_up_col_name: trending_up_arr,
        trending_down_col_name: trending_down_arr
    }, index=original_index)

    return result_df, atr

@jit(nopython=True)
def _calculate_mcginley_numba(source, length):
    mg = np.zeros_like(source)
    if len(source) > 0:
        mg[0] = source[0]
    for i in range(1, len(source)):
        ratio = source[i] / mg[i-1] if mg[i-1] != 0 else 1
        mg[i] = mg[i-1] + (source[i] - mg[i-1]) / (length * ratio ** 4)
    return mg

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
def _calculate_streaks_numba(profits):
    winning_streaks = []
    losing_streaks = []
    current_streak = 0
    current_type = 0  # 0: none, 1: win, -1: loss

    for profit in profits:
        if profit > 0:
            if current_type == 1:
                current_streak += 1
            else:
                if current_type != 0:
                    if current_type == 1:
                        winning_streaks.append(current_streak)
                    else:
                        losing_streaks.append(current_streak)
                current_type = 1
                current_streak = 1
        elif profit < 0:
            if current_type == -1:
                current_streak += 1
            else:
                if current_type != 0:
                    if current_type == 1:
                        winning_streaks.append(current_streak)
                    else:
                        losing_streaks.append(current_streak)
                current_type = -1
                current_streak = 1
        else:
            if current_type != 0:
                if current_type == 1:
                    winning_streaks.append(current_streak)
                else:
                    losing_streaks.append(current_streak)
            current_type = 0
            current_streak = 0

    if current_type != 0:
        if current_type == 1:
            winning_streaks.append(current_streak)
        else:
            losing_streaks.append(current_streak)

    max_winning_streak = max(winning_streaks) if winning_streaks else 0
    max_losing_streak = max(losing_streaks) if losing_streaks else 0

    return max_winning_streak, max_losing_streak

@jit(nopython=True)
def _run_strategy_numba(st_dir, fill_prices, atr, atr_threshold, use_next_open_fill):
    """
    Underlying long/short flip model (NOT option premium).
    If use_next_open_fill: signal on bar i-1, fill on bar i (pass fill=Open).
    Else legacy: same-bar close fill (look-ahead vs live).
    long  -> old sell_pe naming kept for CSV compatibility
    short -> old sell_ce naming kept for CSV compatibility
    """
    n = len(fill_prices)
    long_entry = np.full(n, np.nan)
    long_exit = np.full(n, np.nan)
    short_entry = np.full(n, np.nan)
    short_exit = np.full(n, np.nan)

    in_long = False
    in_short = False

    start = 2 if use_next_open_fill else 1
    for i in range(start, n):
        # Signal bar: previous bar when next-open; current bar when legacy close fill
        sig_i = i - 1 if use_next_open_fill else i
        d = st_dir[sig_i]
        a = atr[sig_i]

        if (not in_long) and (not in_short) and d == 1 and a > atr_threshold:
            long_entry[i] = fill_prices[i]
            in_long = True
        elif in_long and d == -1:
            long_exit[i] = fill_prices[i]
            in_long = False

        if (not in_short) and (not in_long) and d == -1 and a > atr_threshold:
            short_entry[i] = fill_prices[i]
            in_short = True
        elif in_short and d == 1:
            short_exit[i] = fill_prices[i]
            in_short = False

    return long_entry, long_exit, short_entry, short_exit

def run_strategy_and_get_metrics(df_original, period, base_multiplier, rolling_period, mcginley_length, atr_threshold):
    df = df_original.copy()

    ha_df = calculate_heikin_ashi(df.copy())
    df = pd.concat([df, ha_df], axis=1)

    st_result_ha, atr = calculate_supertrend(
        df.copy(),
        period=period,
        base_multiplier=base_multiplier,
        rolling_period=rolling_period,
        mcginley_length=mcginley_length,
        suffix=""
    )
    st_col_name = f"ST_{period}_{str(base_multiplier).replace('.', '_')}_{rolling_period}_{mcginley_length}"
    st_dir_col_name = f"ST_Dir_{period}_{str(base_multiplier).replace('.', '_')}_{rolling_period}_{mcginley_length}"
    trending_up_col_name = f"trendingUp_HA_{period}_{str(base_multiplier).replace('.', '_')}_{rolling_period}_{mcginley_length}"
    trending_down_col_name = f"trendingDown_HA_{period}_{str(base_multiplier).replace('.', '_')}_{rolling_period}_{mcginley_length}"

    df = pd.concat([df, st_result_ha], axis=1)

    df = df.drop(columns=[trending_up_col_name, trending_down_col_name], errors='ignore')

    st_dir = df[st_dir_col_name].to_numpy()
    close_prices = df['Close'].to_numpy()
    open_prices = df['Open'].to_numpy() if 'Open' in df.columns else close_prices
    fill_prices = open_prices if USE_NEXT_OPEN_FILL else close_prices

    long_entry, long_exit, short_entry, short_exit = _run_strategy_numba(
        st_dir, fill_prices, atr, atr_threshold, USE_NEXT_OPEN_FILL
    )

    # Keep old column names so downstream CSVs still parse
    df['Sell_PE_Entry'] = long_entry
    df['Sell_PE_Exit'] = long_exit
    df['Sell_CE_Entry'] = short_entry
    df['Sell_CE_Exit'] = short_exit

    trades = []
    current_long = None
    current_short = None

    for i in range(len(df)):
        if not pd.isna(df['Sell_PE_Entry'].iloc[i]):
            current_long = {'entry_price': df['Sell_PE_Entry'].iloc[i], 'entry_date': df.index[i]}
        if not pd.isna(df['Sell_PE_Exit'].iloc[i]) and current_long is not None:
            # Long underlying: exit - entry, minus round-trip costs
            profit = (df['Sell_PE_Exit'].iloc[i] - current_long['entry_price']) - ROUND_TRIP_COST_INR
            trades.append({
                'type': 'long',
                'entry_date': current_long['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_long['entry_price'],
                'exit_price': df['Sell_PE_Exit'].iloc[i],
                'profit': profit,
                'costs': ROUND_TRIP_COST_INR,
            })
            current_long = None

        if not pd.isna(df['Sell_CE_Entry'].iloc[i]):
            current_short = {'entry_price': df['Sell_CE_Entry'].iloc[i], 'entry_date': df.index[i]}
        if not pd.isna(df['Sell_CE_Exit'].iloc[i]) and current_short is not None:
            # Short underlying: entry - exit, minus round-trip costs
            # (legacy sell_ce used exit-entry which was wrong for short/CE)
            profit = (current_short['entry_price'] - df['Sell_CE_Exit'].iloc[i]) - ROUND_TRIP_COST_INR
            trades.append({
                'type': 'short',
                'entry_date': current_short['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_short['entry_price'],
                'exit_price': df['Sell_CE_Exit'].iloc[i],
                'profit': profit,
                'costs': ROUND_TRIP_COST_INR,
            })
            current_short = None

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

        profits_array = trades_df['profit'].to_numpy()
        max_winning_streak, max_losing_streak = _calculate_streaks_numba(profits_array)

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

def process_param_combo(args):
    df_train, df_test, period, base_multiplier, rolling_period, mcginley_length, atr_threshold = args
    metrics_train = run_strategy_and_get_metrics(df_train.copy(), period, base_multiplier, rolling_period, mcginley_length, atr_threshold)
    metrics_test = run_strategy_and_get_metrics(df_test.copy(), period, base_multiplier, rolling_period, mcginley_length, atr_threshold)

    score_train = calculate_score(metrics_train)
    score_test = calculate_score(metrics_test)

    result = {
        'period': period,
        'base_multiplier': base_multiplier,
        'rolling_period': rolling_period,
        'mcginley_length': mcginley_length,
        'atr_threshold': atr_threshold,
        'score_train': score_train,
        'score_test': score_test,
        **{f'train_{k}': v for k, v in metrics_train.items()},
        **{f'test_{k}': v for k, v in metrics_test.items()}
    }
    return result

def calculate_score(metrics):
    if metrics['total_trades'] == 0:
        return -1000
    profit_score = metrics['net_profit_percent']
    risk_score = -metrics['max_drawdown']
    sharpe = metrics['sharpe_ratio'] if not np.isnan(metrics['sharpe_ratio']) else 0
    sharpe_score = sharpe * 10
    # win_rate_score = metrics['win_rate'] * 0.5  # Increased weight for win rate
    win_rate_score = metrics['win_rate']
    score = profit_score + risk_score + sharpe_score + win_rate_score
    return score

if __name__ == '__main__':
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

        # Walk-forward optimization
        train_size = int(0.7 * len(df_original))
        df_train = df_original[:train_size]
        df_test = df_original[train_size:]

        print(f"Training data: {len(df_train)} rows")
        print(f"Testing data: {len(df_test)} rows")

        # Parameter ranges
        periods = list(range(20, 26))
        base_multipliers = [1.0, 1.2, 1.5]
        rolling_periods = list(range(30, 51))
        mcginley_lengths = list(range(10, 21))
        atr_thresholds = [0, 10, 20, 30]

        print(f"Testing {len(periods)} periods, {len(base_multipliers)} multipliers, {len(rolling_periods)} rolling periods, {len(mcginley_lengths)} mcginley lengths, {len(atr_thresholds)} atr thresholds")
        print(f"Total combinations: {len(periods) * len(base_multipliers) * len(rolling_periods) * len(mcginley_lengths) * len(atr_thresholds)}")

        param_combos = list(itertools.product(periods, base_multipliers, rolling_periods, mcginley_lengths, atr_thresholds))

        # Prepare arguments for parallel processing
        args = [(df_train, df_test, period, base_multiplier, rolling_period, mcginley_length, atr_threshold) for period, base_multiplier, rolling_period, mcginley_length, atr_threshold in param_combos]

        # Use parallel processing
        num_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 workers to avoid memory issues
        print(f"Starting parallel processing with {num_workers} workers...")
        start_time = time.time()
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for idx, result in enumerate(executor.map(process_param_combo, args)):
                results.append(result)
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx+1}/{len(param_combos)} combinations")
        end_time = time.time()
        print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")

        best_result = max(results, key=lambda x: x['score_train'])

        print("\n" + "="*50)
        print("BEST PARAMETERS (based on training data)")
        print("="*50)
        print(f"Period: {best_result['period']}")
        print(f"Base Multiplier: {best_result['base_multiplier']}")
        print(f"Rolling Period: {best_result['rolling_period']}")
        print(f"McGinley Length: {best_result['mcginley_length']}")
        print(f"ATR Threshold: {best_result['atr_threshold']}")
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

        results_df = pd.DataFrame(results)
        optimization_csv_path = 'mcginley_dynamic_supertrend_optimization_results.csv'
        results_df.to_csv(optimization_csv_path, index=False)
        print(f"\nOptimization results saved to '{optimization_csv_path}'")

        print("\nOptimization completed successfully!")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")



# ==================================================
# BEST PARAMETERS (based on training data)
# ==================================================
# Period: 24
# Base Multiplier: 1.2
# Rolling Period: 40
# McGinley Length: 19
# ATR Threshold: 0
# Training Score: 55.56
# Testing Score: 33.80

# --- Training Metrics ---
# Total Trades: 319
# Winning Trades: 203
# Losing Trades: 116
# Win Rate: 63.64
# Avg Win: 159.86
# Avg Loss: -191.76
# Risk Reward Ratio: 0.83
# Profit Factor: 1.46
# Expectancy: 32.0
# Net Profit: 10208.15
# Net Profit Percent: 10.21
# Max Drawdown: -2.33
# Return Drawdown Ratio: 4.38
# Max Winning Streak: 10
# Max Losing Streak: 5
# Sharpe Ratio: 1.12
# Sortino Ratio: 0.83

# --- Testing Metrics ---
# Total Trades: 124
# Winning Trades: 69
# Losing Trades: 55
# Win Rate: 55.65
# Avg Win: 203.66
# Avg Loss: -242.23
# Risk Reward Ratio: 0.84
# Profit Factor: 1.05
# Expectancy: 5.89
# Net Profit: 730.2
# Net Profit Percent: 0.73
# Max Drawdown: -3.74
# Return Drawdown Ratio: 0.2
# Max Winning Streak: 8
# Max Losing Streak: 6
# Sharpe Ratio: 0.15
# Sortino Ratio: 0.09