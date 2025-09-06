# McGinley Dynamic Supertrend Strategy
# Runs the McGinley Dynamic Supertrend strategy with fixed parameters
# Parameters: ATR period, base multiplier, rolling period, McGinley length, ATR threshold

import pandas as pd
import numpy as np
from numba import jit

# Define the path to the CSV file
csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_60_Min.csv'

# Best parameters (from optimization)
period = 24
base_multiplier = 1.2
rolling_period = 40
mcginley_length = 19
atr_threshold = 0

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
def _run_strategy_numba(st_dir, close_prices, atr, atr_threshold):
    sell_pe_entry_prices = np.full_like(close_prices, np.nan)
    sell_pe_exit_prices = np.full_like(close_prices, np.nan)
    sell_ce_entry_prices = np.full_like(close_prices, np.nan)
    sell_ce_exit_prices = np.full_like(close_prices, np.nan)

    in_sell_pe_position = False
    in_sell_ce_position = False

    for i in range(1, len(close_prices)):
        if not in_sell_pe_position and not in_sell_ce_position and st_dir[i] == 1 and atr[i] > atr_threshold:
            sell_pe_entry_prices[i] = close_prices[i]
            in_sell_pe_position = True
        elif in_sell_pe_position and st_dir[i] == -1:
            sell_pe_exit_prices[i] = close_prices[i]
            in_sell_pe_position = False

        if not in_sell_ce_position and not in_sell_pe_position and st_dir[i] == -1 and atr[i] > atr_threshold:
            sell_ce_entry_prices[i] = close_prices[i]
            in_sell_ce_position = True
        elif in_sell_ce_position and st_dir[i] == 1:
            sell_ce_exit_prices[i] = close_prices[i]
            in_sell_ce_position = False

    return sell_pe_entry_prices, sell_pe_exit_prices, sell_ce_entry_prices, sell_ce_exit_prices

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

    sell_pe_entry_prices, sell_pe_exit_prices, sell_ce_entry_prices, sell_ce_exit_prices = _run_strategy_numba(
        st_dir, close_prices, atr, atr_threshold
    )

    df['Sell_PE_Entry'] = sell_pe_entry_prices
    df['Sell_PE_Exit'] = sell_pe_exit_prices
    df['Sell_CE_Entry'] = sell_ce_entry_prices
    df['Sell_CE_Exit'] = sell_ce_exit_prices

    trades = []
    current_sell_pe_entry = None
    current_sell_ce_entry = None

    for i in range(len(df)):
        if not pd.isna(df['Sell_PE_Entry'].iloc[i]):
            current_sell_pe_entry = {'entry_price': df['Sell_PE_Entry'].iloc[i], 'entry_date': df.index[i]}
        if not pd.isna(df['Sell_PE_Exit'].iloc[i]) and current_sell_pe_entry is not None:
            profit = df['Sell_PE_Exit'].iloc[i] - current_sell_pe_entry['entry_price']
            trades.append({
                'type': 'sell_pe',
                'entry_date': current_sell_pe_entry['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_sell_pe_entry['entry_price'],
                'exit_price': df['Sell_PE_Exit'].iloc[i],
                'profit': profit
            })
            current_sell_pe_entry = None

        if not pd.isna(df['Sell_CE_Entry'].iloc[i]):
            current_sell_ce_entry = {'entry_price': df['Sell_CE_Entry'].iloc[i], 'entry_date': df.index[i]}
        if not pd.isna(df['Sell_CE_Exit'].iloc[i]) and current_sell_ce_entry is not None:
            profit = df['Sell_CE_Exit'].iloc[i] - current_sell_ce_entry['entry_price']
            trades.append({
                'type': 'sell_ce',
                'entry_date': current_sell_ce_entry['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_sell_ce_entry['entry_price'],
                'exit_price': df['Sell_CE_Exit'].iloc[i],
                'profit': profit
            })
            current_sell_ce_entry = None

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
    return metrics, trades_df

if __name__ == '__main__':
    try:
        df_original = pd.read_csv(csv_file_path)
        df_original = df_original.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
        df_original = df_original.rename(columns={'Date': 'DateTime'})
        df_original = df_original.set_index('DateTime')
        df_original.index = pd.to_datetime(df_original.index)
        df_original = df_original[~df_original.index.duplicated(keep='first')]

        df_original = df_original[df_original.index.year >= 2021]

        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df_original.columns:
                df_original[col] = df_original[col].round(2)

        print(f"Data loaded: {len(df_original)} rows")

        # Run strategy with best parameters
        metrics, trades_df = run_strategy_and_get_metrics(df_original, period, base_multiplier, rolling_period, mcginley_length, atr_threshold)

        print("\n" + "="*50)
        print("STRATEGY RESULTS")
        print("="*50)
        print(f"Parameters: Period={period}, Base Multiplier={base_multiplier}, Rolling Period={rolling_period}, McGinley Length={mcginley_length}, ATR Threshold={atr_threshold}")
        print("\n--- Metrics ---")
        for key, value in metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        # Save trades to CSV
        trades_csv_path = 'mcginley_dynamic_supertrend_trades.csv'
        trades_df.to_csv(trades_csv_path, index=False)
        print(f"\nTrades saved to '{trades_csv_path}'")

        print("\nStrategy execution completed successfully!")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")