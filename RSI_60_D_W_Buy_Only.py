# Trading strategy backtest for: 60-min entries with daily+weekly confirmation (BUY ONLY)
# Features:
# - Uses 60-min OHLC file and daily OHLC file (resamples daily -> weekly)
# - Avoids lookahead by merging daily/weekly values as of the last closed daily/weekly bar <= 60m timestamp
# - RSI thresholds for daily, weekly, 60m are exposed as variables for optimization
# - Enters on next 60-min OPEN after signal (no lookahead into the current bar's future)
# - Exits: configurable to use daily/60m/weekly RSI crossover condition (prev bar above, current below threshold)
# - Clean execution with no slippage or transaction costs (simplified)
# - Produces trade list and summary metrics (net P/L, CAGR, Sharpe, max drawdown, win rate, avg trade, etc.)
# - BUY ONLY strategy - no short selling
#
# Requirements: pandas, numpy, scipy (for stats), matplotlib (optional for plot)
#
# Usage: set CSV paths then run. CSVs must have a datetime column named 'datetime' or change variable below.
#  - 60m CSV should be continuous 60-minute candles with columns: datetime, open, high, low, close, volume (volume optional)
#  - Daily CSV should be daily OHLC with same column names.
#
# IMPORTANT: Read comments about lookback/merge behavior to ensure it matches your data timezone / timestamps.

import pandas as pd
import numpy as np
from math import isnan
from dataclasses import dataclass
from datetime import timedelta
from scipy import stats

# -------------------------
# USER CONFIG (change these)
# -------------------------
path_60m_csv = "D:\\Sushant\\Fyers_AlgoTrade\\Fyers_Data\\Nifty50_Index\\NIFTY50_INDEX_60_Min.csv"      # <-- set your 60-min CSV path
path_daily_csv = "D:\\Sushant\\Fyers_AlgoTrade\\Fyers_Data\\Nifty50_Index\\NIFTY50_INDEX_D_Min.csv"    # <-- set your daily CSV path
output_folder = "."                  # where to save outputs (optional)

# column names (change if your csv uses different names)
DT_COL = "Date"
OPEN = "Open"
HIGH = "High"
LOW = "Low"
CLOSE = "Close"
VOL = "Volume"

# Strategy parameters (exposed for optimization) - BUY ONLY (Multi-timeframe oversold bounce)
weekly_rsi_threshold_buy = 30.0  # need weekly RSI > this (oversold bounce from < 35)
daily_rsi_threshold_buy = 28.0   # need daily RSI > this (oversold bounce from < 30)
rsi_60m_entry_buy = 35.0         # 60m RSI > this to enter buy (oversold bounce from < 25)

# exit mode: 'daily' (exit when daily RSI crosses back), '60m', 'weekly', or 'any' (any of the selected triggers)
exit_mode = 'daily'

# Exit thresholds (used when exit_mode selects 60m/weekly)
daily_exit_threshold = 72.0  # for buy exit when daily RSI goes below this (exit when bounce loses momentum)
weekly_exit_threshold = 60.0
rsi_60m_exit_threshold = 60.0

# Money & execution settings
initial_capital = 100000.0
position_size = 1   # fixed position size for all trades (1 share/contract)

# Indicator settings
rsi_length_60m = 12
rsi_length_daily = 2
rsi_length_weekly = 5

# -------------------------
# Helper functions
# -------------------------
def compute_rsi(series: pd.Series, length: int):
    """
    Classic Wilder RSI (smoothed) computed progressively to avoid forward-looking bias.
    Returns same length Series with NaNs for initial periods.
    """
    if length < 1:
        raise ValueError("RSI length must be >= 1")

    # Initialize result series
    rsi_values = pd.Series(index=series.index, dtype=float)

    # Need minimum 2 * length + 1 data points for proper RSI calculation
    min_periods = length + 1

    if len(series) < min_periods:
        rsi_values.iloc[:] = np.nan
        return rsi_values

    # Calculate initial average gain and loss for the first RSI value
    initial_gains = []
    initial_losses = []

    for i in range(1, min_periods):
        delta = series.iloc[i] - series.iloc[i-1]
        if delta > 0:
            initial_gains.append(delta)
            initial_losses.append(0)
        else:
            initial_gains.append(0)
            initial_losses.append(abs(delta))

    avg_gain = sum(initial_gains) / length
    avg_loss = sum(initial_losses) / length

    # Calculate first RSI value
    if avg_loss == 0:
        rsi_values.iloc[length] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_values.iloc[length] = 100 - (100 / (1 + rs))

    # Calculate subsequent RSI values using Wilder's smoothing
    alpha = 1.0 / length
    current_gain = avg_gain
    current_loss = avg_loss

    for i in range(length + 1, len(series)):
        delta = series.iloc[i] - series.iloc[i-1]
        if delta > 0:
            gain = delta
            loss = 0
        else:
            gain = 0
            loss = abs(delta)

        # Wilder's smoothing
        current_gain = (current_gain * (length - 1) + gain) / length
        current_loss = (current_loss * (length - 1) + loss) / length

        if current_loss == 0:
            rsi_values.iloc[i] = 100.0
        else:
            rs = current_gain / current_loss
            rsi_values.iloc[i] = 100 - (100 / (1 + rs))

    # Fill initial NaN values
    for i in range(length):
        rsi_values.iloc[i] = np.nan

    return rsi_values

def resample_daily_to_weekly(df_daily: pd.DataFrame):
    """
    Accepts daily OHLC indexed by datetime (date of close). Returns weekly OHLC with same index as weekly close (Timestamp of last day in week).
    We'll use 'W-FRI' by default: weekly periods ending on Friday. But to be robust, use pandas' default 'W' which ends Sun.
    We will resample to calendar-week ending Sunday; user can adjust if they need 'W-FRI' or custom weeks.
    """
    # Ensure datetime index and sorted
    d = df_daily.copy()
    d = d.sort_index()
    # Resample OHLC: open = first, high=max, low=min, close=last, volume=sum
    o = d[OPEN].resample('W').first()
    h = d[HIGH].resample('W').max()
    l = d[LOW].resample('W').min()
    c = d[CLOSE].resample('W').last()
    try:
        v = d[VOL].resample('W').sum()
        weekly = pd.concat([o,h,l,c,v], axis=1)
    except Exception:
        weekly = pd.concat([o,h,l,c], axis=1)
    weekly.columns = [OPEN, HIGH, LOW, CLOSE] + ([VOL] if VOL in d.columns else [])
    weekly = weekly.dropna(subset=[CLOSE])
    return weekly

def align_daily_weekly_to_60m(df_60m, df_daily, df_weekly):
    """
    Merge the latest daily and weekly values that are *available* as of each 60-min timestamp.
    For a 60-min bar with timestamp t (bar close time), the latest available daily/weekly is the last daily/weekly bar with timestamp <= t.
    We'll do merge_asof with 'backward' direction meaning take last row with index <= t.
    Input dataframes must have DatetimeIndex.

    FIXED: Calculates RSI progressively to avoid forward-looking bias.
    Daily RSI for day N is calculated using ONLY data up to day N, not future days.
    """
    # For 60m, compute RSI progressively to avoid forward-looking bias
    df = df_60m.copy()
    df = df.sort_index()
    df['rsi_60m'] = compute_rsi(df[CLOSE], rsi_length_60m)

    # Calculate daily RSI progressively to avoid forward-looking bias
    # This ensures RSI for day N only uses data up to day N
    daily_rsi_values = []
    daily_dates = df_daily.index.sort_values()

    for i, current_date in enumerate(daily_dates):
        # Get historical data up to current date (inclusive)
        historical_data = df_daily[df_daily.index <= current_date][CLOSE]

        if len(historical_data) >= rsi_length_daily:
            # Calculate RSI using only historical data up to current date
            rsi_series = compute_rsi(historical_data, rsi_length_daily)
            # Get the RSI value for the current date
            current_rsi = rsi_series.iloc[-1] if len(rsi_series) > 0 else np.nan
            daily_rsi_values.append((current_date, current_rsi))
        else:
            daily_rsi_values.append((current_date, np.nan))

    # Create daily RSI series without forward-looking bias
    daily_rsi_df = pd.DataFrame(daily_rsi_values, columns=['DateTime', 'daily_rsi'])
    daily_rsi_df = daily_rsi_df.set_index('DateTime')

    # Shift daily RSI by one day to avoid future bias - for day N, use RSI from day N-1
    daily_rsi_df['daily_rsi'] = daily_rsi_df['daily_rsi'].shift(1)

    # Calculate weekly RSI progressively to avoid forward-looking bias
    weekly_rsi_values = []
    weekly_dates = df_weekly.index.sort_values()

    for i, current_date in enumerate(weekly_dates):
        # Get historical data up to current date (inclusive)
        historical_data = df_weekly[df_weekly.index <= current_date][CLOSE]

        if len(historical_data) >= rsi_length_weekly:
            # Calculate RSI using only historical data up to current date
            rsi_series = compute_rsi(historical_data, rsi_length_weekly)
            # Get the RSI value for the current date
            current_rsi = rsi_series.iloc[-1] if len(rsi_series) > 0 else np.nan
            weekly_rsi_values.append((current_date, current_rsi))
        else:
            weekly_rsi_values.append((current_date, np.nan))

    # Create weekly RSI series without forward-looking bias
    weekly_rsi_df = pd.DataFrame(weekly_rsi_values, columns=['DateTime', 'weekly_rsi'])
    weekly_rsi_df = weekly_rsi_df.set_index('DateTime')

    # merge_asof: indexes as column
    df_reset = df.reset_index()
    if 'index' in df_reset.columns:
        df_reset = df_reset.rename(columns={'index':'ts'})
    else:
        df_reset = df_reset.rename(columns={df_reset.columns[0]:'ts'})

    daily_reset = daily_rsi_df.reset_index()
    if 'index' in daily_reset.columns:
        daily_reset = daily_reset.rename(columns={'index':'ts'})
    else:
        daily_reset = daily_reset.rename(columns={daily_reset.columns[0]:'ts'})

    weekly_reset = weekly_rsi_df.reset_index()
    if 'index' in weekly_reset.columns:
        weekly_reset = weekly_reset.rename(columns={'index':'ts'})
    else:
        weekly_reset = weekly_reset.rename(columns={weekly_reset.columns[0]:'ts'})

    # make sure ts are sorted
    df_reset = df_reset.sort_values('ts')
    daily_reset = daily_reset.sort_values('ts')
    weekly_reset = weekly_reset.sort_values('ts')

    # merge last daily value <= ts
    df_merged = pd.merge_asof(df_reset, daily_reset, on='ts', direction='backward')
    df_merged = pd.merge_asof(df_merged, weekly_reset, on='ts', direction='backward')

    df_final = df_merged.set_index('ts')

    # Remove any duplicate indices that may have been created during merging
    df_final = df_final[~df_final.index.duplicated(keep='first')]

    return df_final

# -------------------------
# Load data
# -------------------------
def load_and_prepare(path_60m_csv, path_daily_csv, dt_col=DT_COL):
    # Load 60-min
    df60 = pd.read_csv(path_60m_csv, parse_dates=[dt_col])
    df60 = df60.drop(columns=['Unnamed: 0'], errors='ignore')
    df60 = df60.rename(columns={'Date': 'DateTime'})
    df60 = df60[~df60.index.duplicated(keep='first')]
    df60 = df60.set_index('DateTime').sort_index()
    # Ensure numeric columns
    for c in [OPEN,HIGH,LOW,CLOSE]:
        df60[c] = pd.to_numeric(df60[c], errors='coerce')
    # Drop rows with NaN close
    df60 = df60[~df60[CLOSE].isna()]
    # Additional deduplication after all processing
    df60 = df60[~df60.index.duplicated(keep='first')]

    # Load daily
    dfd = pd.read_csv(path_daily_csv, parse_dates=[dt_col])
    dfd = dfd.drop(columns=['Unnamed: 0'], errors='ignore')
    dfd = dfd.rename(columns={'Date': 'DateTime'})
    dfd = dfd[~dfd.index.duplicated(keep='first')]
    dfd = dfd.set_index('DateTime').sort_index()
    for c in [OPEN,HIGH,LOW,CLOSE]:
        dfd[c] = pd.to_numeric(dfd[c], errors='coerce')
    dfd = dfd[~dfd[CLOSE].isna()]
    # Additional deduplication after all processing
    dfd = dfd[~dfd.index.duplicated(keep='first')]

    return df60, dfd

# -------------------------
# Strategy: signal generation avoiding lookahead (BUY ONLY)
# -------------------------
def generate_signals(df60, daily, weekly):
    """
    For each 60-min index we have:
      - daily_rsi (most recent closed daily bar <= ts)
      - weekly_rsi (most recent closed weekly bar <= ts)
      - rsi_60m (computed for 60m series)
    Signal rules (BUY ONLY):
      BUY signal on 60m bar if:
        daily_rsi > daily_rsi_threshold_buy and weekly_rsi > weekly_rsi_threshold_buy and rsi_60m > rsi_60m_entry_buy
    We'll create a 'signal' column with 1 = buy, 0 = no signal.
    Implementation detail: We'll only mark signal on the 60-min bar where conditions hold, but execution will be at the next 60-min bar's open.
    """
    df = align_daily_weekly_to_60m(df60, daily, weekly)
    df = df.copy()

    # ensure we do not use the rsi for the current bar's close to create a signal AND execute at next open.
    # compute 'signal' using rsi_60m shifted by 0 because rsi_60m is computed using past data up to bar close.
    # But most conservative approach: treat the rsi_60m at bar t as available only at bar close; so execution at next bar open uses this rsi.
    # So signal at timestamp t means "enter at next bar open".
    df['signal'] = 0

    # buy condition only
    buy_cond = (
        (df['daily_rsi'] > daily_rsi_threshold_buy) &
        (df['weekly_rsi'] > weekly_rsi_threshold_buy) &
        (df['rsi_60m'] > rsi_60m_entry_buy)
    )
    df.loc[buy_cond, 'signal'] = 1

    # We'll add a column 'signal_exec_price' which will be the next bar OPEN price where trade would actually execute.
    df['next_open'] = df[OPEN].shift(-1)  # execution price at next bar open (NaN for last row)
    # Add execution timestamp (next bar ts) - use to_series().shift() to avoid frequency issues
    df['exec_ts'] = df.index.to_series().shift(-1)  # next bar timestamp where order executes

    # NOTE: For maximum realism, consider using next bar's OPEN for signal execution
    # Current implementation executes at next open, which is correct for avoiding lookahead bias

    return df

# -------------------------
# Backtest engine (simple: one position at a time) - BUY ONLY
# -------------------------
@dataclass
class Trade:
    entry_ts: pd.Timestamp
    entry_price: float
    side: int                    # 1=long, -1=short
    size: float                  # number of contracts/shares
    exit_ts: pd.Timestamp = None
    exit_price: float = None
    pnl: float = None
    return_pct: float = None

def run_backtest(df_signals):
    """
    Single-position backtest (BUY ONLY):
      - At a signal (1) and if no open position, we enter at next bar open (no slippage or fees).
      - Fixed position size for all trades.
      - Exit is triggered depending on exit_mode:
           if exit_mode == 'daily': exit when daily_rsi crosses below daily_exit_threshold
           if exit_mode == '60m': exit when rsi_60m crosses below rsi_60m_exit_threshold
           if exit_mode == 'weekly': exit when weekly_rsi crosses below weekly_exit_threshold
           if exit_mode == 'any': exit when any selected condition hits
      - Exit executed at next bar open after the condition is *observed* (same logic for fairness).
      - Crossover logic: For daily exits, checks if previous bar was above threshold and current is below
    """
    capital = initial_capital
    trades = []
    position = None  # current trade object
    df = df_signals.copy()
    df = df.sort_index()

    for idx in df.index:
        row = df.loc[idx]

        # 1) If there's an open position, check exits based on the rsi values available at the current timestamp.
        if position is not None:
            # check exit conditions using the current row's available RSI values (these are computed up to previous bars)
            should_exit = False

            # Long position exit logic (only buy positions in this strategy)
            if exit_mode == 'daily':
                daily_rsi_val = row['daily_rsi']
                # Handle both scalar and Series cases
                if hasattr(daily_rsi_val, 'iloc'):
                    daily_rsi_scalar = daily_rsi_val.iloc[0]
                else:
                    daily_rsi_scalar = daily_rsi_val

                # Get previous bar's daily RSI for crossover check
                try:
                    current_idx = df.index.get_loc(idx)
                    # Handle case where get_loc returns a slice (duplicate indices)
                    if isinstance(current_idx, slice):
                        current_idx = current_idx.start
                    prev_daily_rsi = None

                    if current_idx > 0:
                        prev_row = df.iloc[current_idx - 1]
                        prev_daily_rsi_val = prev_row['daily_rsi']
                        if hasattr(prev_daily_rsi_val, 'iloc'):
                            prev_daily_rsi = prev_daily_rsi_val.iloc[0]
                        else:
                            prev_daily_rsi = prev_daily_rsi_val
                except KeyError:
                    # If index not found, skip crossover check
                    prev_daily_rsi = None

                # Crossover exit: Previous bar > threshold AND current bar < threshold
                if (not pd.isna(daily_rsi_scalar) and prev_daily_rsi is not None and not pd.isna(prev_daily_rsi) and
                    daily_rsi_scalar < daily_exit_threshold and prev_daily_rsi >= daily_exit_threshold):
                    should_exit = True
            elif exit_mode == '60m':
                rsi_60m_val = row['rsi_60m']
                if hasattr(rsi_60m_val, 'iloc'):
                    rsi_60m_scalar = rsi_60m_val.iloc[0]
                else:
                    rsi_60m_scalar = rsi_60m_val
                if not pd.isna(rsi_60m_scalar) and rsi_60m_scalar < rsi_60m_exit_threshold:
                    should_exit = True
            elif exit_mode == 'weekly':
                weekly_rsi_val = row['weekly_rsi']
                if hasattr(weekly_rsi_val, 'iloc'):
                    weekly_rsi_scalar = weekly_rsi_val.iloc[0]
                else:
                    weekly_rsi_scalar = weekly_rsi_val
                if not pd.isna(weekly_rsi_scalar) and weekly_rsi_scalar < weekly_exit_threshold:
                    should_exit = True
            elif exit_mode == 'any':
                conds = []
                daily_rsi_val = row.get('daily_rsi', np.nan)
                rsi_60m_val = row.get('rsi_60m', np.nan)
                weekly_rsi_val = row.get('weekly_rsi', np.nan)
                # Handle scalar conversion for each value
                for val, threshold, name in [(daily_rsi_val, daily_exit_threshold, 'daily'),
                                            (rsi_60m_val, rsi_60m_exit_threshold, '60m'),
                                            (weekly_rsi_val, weekly_exit_threshold, 'weekly')]:
                    if not pd.isna(val):
                        if hasattr(val, 'iloc'):
                            scalar_val = val.iloc[0]
                        else:
                            scalar_val = val
                        conds.append(scalar_val < threshold)
                if any(conds):
                    should_exit = True

            if should_exit:
                # exit at next bar open (ensure there is a next bar)
                try:
                    exec_idx = df.index.get_loc(idx)
                    if isinstance(exec_idx, slice):
                        exec_idx = exec_idx.start
                    exec_idx = exec_idx + 1
                    if exec_idx < len(df.index):
                        exit_ts = df.index[exec_idx]
                        exit_price = df.iloc[exec_idx][OPEN]
                        # Calculate P&L (no fees or slippage)
                        pnl = position.size * (exit_price - position.entry_price) * position.side
                        position.exit_ts = exit_ts
                        position.exit_price = exit_price
                        position.pnl = pnl
                        position.return_pct = pnl / (position.entry_price * position_size)
                        trades.append(position)
                        position = None
                        # continue loop after exit
                    else:
                        # no further bars to execute exit; skip
                        pass
                except:
                    # If get_loc fails, skip this exit
                    pass

        # 2) If no position, check for entry signal at this index; if signal exists, enter at next open
        if position is None:
            signal_val = row['signal']
            if hasattr(signal_val, 'iloc'):
                signal_scalar = signal_val.iloc[0]
            else:
                signal_scalar = signal_val
            if signal_scalar == 1:  # Only buy signals (signal == 1)
                # execute at next bar open
                try:
                    exec_idx = df.index.get_loc(idx)
                    if isinstance(exec_idx, slice):
                        exec_idx = exec_idx.start
                    exec_idx = exec_idx + 1
                    if exec_idx < len(df.index):
                        entry_ts = df.index[exec_idx]
                        entry_price = df.iloc[exec_idx][OPEN]
                        # use fixed position size (no fees or slippage)
                        size = position_size
                        # create trade
                        position = Trade(entry_ts=entry_ts, entry_price=entry_price, side=1, size=size)
                    else:
                        pass  # no further bars to execute entry
                except:
                    pass  # if get_loc fails, skip this entry

    # if position still open at end, close at last close price (or last open if prefer)
    if position is not None:
        last_idx = df.index[-1]
        # exit at last close (conservative)
        exit_price = df.iloc[-1][CLOSE]
        # Calculate P&L (no fees or slippage)
        pnl = position.size * (exit_price - position.entry_price) * position.side
        position.exit_ts = last_idx
        position.exit_price = exit_price
        position.pnl = pnl
        position.return_pct = pnl / (position.entry_price * position_size)
        trades.append(position)
        position = None

    return trades, capital

# -------------------------
# Performance metrics
# -------------------------
def trades_to_df(trades):
    rows = []
    for t in trades:
        rows.append({
            'entry_ts': t.entry_ts,
            'exit_ts': t.exit_ts,
            'side': t.side,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'size': t.size,
            'pnl': t.pnl,
            'return_pct': t.return_pct
        })
    return pd.DataFrame(rows)

def compute_metrics(trades_df, initial_capital=initial_capital):
    if trades_df.empty:
        return {}
    total_pnl = trades_df['pnl'].sum()
    net_profit = total_pnl
    # equity curve: approximate incremental by accumulating trade pnl in chronological order on trade exit_ts
    df = trades_df.sort_values('exit_ts').copy()
    df['cum_pnl'] = df['pnl'].cumsum()
    # build equity over time series at trade exits
    equity = pd.Series(df['cum_pnl'].values + initial_capital, index=pd.to_datetime(df['exit_ts']).values)

    # CAGR: from first trade entry to last exit
    start = pd.to_datetime(df['entry_ts'].iloc[0])
    end = pd.to_datetime(df['exit_ts'].iloc[-1])
    years = (end - start).days / 365.25 if (end - start).days > 0 else 1/252
    final_equity = equity.iloc[-1]
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else np.nan

    # daily returns approximation (spread P&L across trade duration?) We'll compute returns on the series of trade exits (not perfect).
    rets = df['pnl'] / initial_capital
    if len(rets) > 1:
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() != 0 else np.nan
    else:
        sharpe = np.nan

    # max drawdown on equity series
    eq = equity.copy()
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max
    max_dd = drawdown.min()

    win_trades = df[df['pnl'] > 0]
    loss_trades = df[df['pnl'] <= 0]

    metrics = {
        'total_trades': len(df),
        'net_profit': net_profit,
        'final_equity': final_equity,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': len(win_trades) / len(df) if len(df)>0 else np.nan,
        'avg_win': win_trades['pnl'].mean() if len(win_trades)>0 else np.nan,
        'avg_loss': loss_trades['pnl'].mean() if len(loss_trades)>0 else np.nan,
        'avg_trade_pnl': df['pnl'].mean(),
        'median_trade_pnl': df['pnl'].median()
    }
    return metrics, equity, df

# -------------------------
# Main routine
# -------------------------
def main():
    # Load
    df60, dfdaily = load_and_prepare(path_60m_csv, path_daily_csv)
    # Build weekly from daily
    dfweekly = resample_daily_to_weekly(dfdaily)

    # Generate signals (aligned) - BUY ONLY
    df_signals = generate_signals(df60, dfdaily, dfweekly)

    # Run backtest
    trades, final_capital = run_backtest(df_signals)

    # Convert trades to df, metrics
    trades_df = trades_to_df(trades)
    metrics, equity_series, trades_summary = compute_metrics(trades_df)

    # Save outputs
    trades_df.to_csv(f"{output_folder}/trades_list_buy_only_simplified.csv", index=False)
    if isinstance(equity_series, pd.Series):
        equity_series.to_csv(f"{output_folder}/equity_curve_buy_only_simplified.csv")
    # Save cleaned signal data if helpful
    # Remove any duplicates before saving
    df_signals_deduped = df_signals[~df_signals.index.duplicated(keep='first')]

    # Round all numeric columns to 2 decimal places for cleaner output
    numeric_columns = df_signals_deduped.select_dtypes(include=[np.number]).columns
    df_signals_deduped[numeric_columns] = df_signals_deduped[numeric_columns].round(2)

    df_signals_deduped.to_csv(f"{output_folder}/signals_buy_only_simplified.csv")

    # Print summary
    print("=== Buy-Only Backtest Summary ===")
    print(f"Initial capital: {initial_capital:.2f}")
    print(f"Final capital (approx): {final_capital:.2f}")

    print("\nStrategy Parameters:")
    print(f"  60m RSI Period (buy): {rsi_length_60m}")
    print(f"  Daily RSI Period (buy): {rsi_length_daily}")
    print(f"  Weekly RSI Period (buy): {rsi_length_weekly}")
    print(f"  Daily RSI threshold (buy): {daily_rsi_threshold_buy}")
    print(f"  Weekly RSI threshold (buy): {weekly_rsi_threshold_buy}")
    print(f"  60m RSI entry threshold: {rsi_60m_entry_buy}")
    print(f"  Exit mode: {exit_mode}")
    print(f"  Daily exit threshold: {daily_exit_threshold}")
    print(f"  Weekly exit threshold: {weekly_exit_threshold}")
    print(f"  60m exit threshold: {rsi_60m_exit_threshold}")

    print("\nMetrics:")
    for k,v in metrics.items():
        if pd.isna(v):
            print(f"  {k}: N/A")
        elif k in ['cagr', 'win_rate']:
            print(f"  {k}: {v*100:.2f}%")
        elif k in ['max_drawdown']:
            print(f"  {k}: {v*100:.2f}%")
        elif k in ['sharpe']:
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v:.2f}")
    print(f"Total trades: {len(trades_df)}")
    display_cols = ['entry_ts','exit_ts','side','entry_price','exit_price','pnl','return_pct']
    if not trades_df.empty:
        print("\nSample trades:")
        print(trades_df[display_cols].head(10).to_string(index=False))

    return {
        'df_signals': df_signals,
        'trades_df': trades_df,
        'metrics': metrics,
        'equity': equity_series
    }

# If run as script
if __name__ == "__main__":
    out = main()


# === Buy-Only Backtest Summary ===

# Initial capital: 100000.00

# Strategy Parameters:
#   60m RSI Period (buy): 11
#   Daily RSI Period (buy): 2
#   Weekly RSI Period (buy): 5
#   Daily RSI threshold (buy): 28.0
#   Weekly RSI threshold (buy): 30.0
#   60m RSI entry threshold: 35.0
#   Exit mode: daily
#   Daily exit threshold: 72.0
#   Weekly exit threshold: 60.0
#   60m exit threshold: 60.0

# Metrics:
#   total_trades: 255.00
#   net_profit: 18545.55
#   final_equity: 118545.55
#   cagr: 2.10%
#   sharpe: 3.35
#   max_drawdown: -2.34%
#   win_rate: 64.31%
#   avg_win: 247.11
#   avg_loss: -241.54
#   avg_trade_pnl: 72.73
#   median_trade_pnl: 78.40
# Total trades: 255