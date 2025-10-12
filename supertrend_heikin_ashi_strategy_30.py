# supertrend_heikin_ashi_fixed.py
import pandas as pd
import numpy as np
import glob
import os

# Optional: use TA-Lib ATR if available for standard ATR calculation
try:
    import talib as ta
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False

# -------------------------
# Parameters / config
# -------------------------
DATA_DIR = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'  # path to CSV file
ATR_PERIOD = 24
ATR_MULTIPLIER = 0.9
USE_TALIB_ATR = TALIB_AVAILABLE  # set False to use rolling SMA of True Range

# Risk Management Parameters
TRAILING_STOP_MULTIPLIER = 1.0  # 1-2x ATR for trailing stop
RISK_PERCENTAGE = 0.01  # 1% of capital per trade
INITIAL_CAPITAL = 100000  # starting capital
MAX_LOSS_STREAK = 15  # allow more losses to continue trading
# -------------------------

def load_and_prepare(data_file):
    try:
        df = pd.read_csv(data_file)
        # Keep expected columns and normalize names (adapts to file having those columns)
        if all(col in df.columns for col in ['Date', 'Open', 'High', 'Low', 'Close']):
            df = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
            df.columns = ['date', 'open', 'high', 'low', 'close']
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            df = df.drop_duplicates()
            return df
        else:
            raise ValueError("CSV file does not have required columns: Date, Open, High, Low, Close")
    except Exception as e:
        print(f"Error loading {data_file}: {e}")
        raise

def calculate_heikin_ashi(df):
    """
    Returns ha_df with columns: HA_Open, HA_High, HA_Low, HA_Close
    HA_Close is vectorized. HA_Open computed iteratively:
      HA_Open[t] = (HA_Open[t-1] + HA_Close[t-1]) / 2
    First HA_Open initialized to (raw_open + raw_close)/2 (common choice).
    """
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0

    # prepare series with same index
    ha_open = pd.Series(index=df.index, dtype='float64')
    # initialize
    ha_open.iat[0] = (df['open'].iat[0] + df['close'].iat[0]) / 2.0

    # iterate to get HA_Open from previous HA values
    for i in range(1, len(df)):
        ha_open.iat[i] = (ha_open.iat[i-1] + ha_close.iat[i-1]) / 2.0

    ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([df['low'], ha_open, ha_close], axis=1).min(axis=1)

    ha_df = pd.DataFrame({
        'HA_Open' : ha_open,
        'HA_High' : ha_high,
        'HA_Low'  : ha_low,
        'HA_Close': ha_close
    }, index=df.index)

    return ha_df

def calculate_atr(ha_df, period=14, use_talib=True):
    """
    ATR computed on HA_High, HA_Low, HA_Close (consistent with earlier code).
    If TA-Lib unavailable or use_talib=False, a rolling SMA of TR with min_periods=1 is used.
    """
    high = ha_df['HA_High']
    low  = ha_df['HA_Low']
    close = ha_df['HA_Close']

    if use_talib and TALIB_AVAILABLE:
        atr_values = ta.ATR(high.values, low.values, close.values, timeperiod=period)
        atr = pd.Series(atr_values, index=ha_df.index)
    else:
        # True Range
        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def calculate_supertrend(ha_df, period=10, multiplier=3.0, use_talib_atr=False):
    """
    Robust Supertrend:
      - fills ATR NaNs by backfilling so bands start having numeric values
      - initializes final bands at first valid index
    Returns: trend, final_upper, final_lower, atr
    """
    ha_close = ha_df['HA_Close']
    ha_high = ha_df['HA_High']
    ha_low  = ha_df['HA_Low']

    hl2 = (ha_high + ha_low) / 2.0
    atr = calculate_atr(ha_df, period=period, use_talib=use_talib_atr)

    # --- Make ATR numeric from the start to avoid NaN propagation ---
    # Option: backfill (preferred) so band math has numbers early.
    if atr.isna().any():
        atr = atr.bfill()
        # If still NaNs (all NaN), fallback to zeros
        atr = atr.fillna(0.0)

    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    final_upper = pd.Series(index=ha_df.index, dtype='float64')
    final_lower = pd.Series(index=ha_df.index, dtype='float64')
    trend = pd.Series(index=ha_df.index, dtype='int8')

    # find first index where basic bands are finite
    valid_idx = basic_upper.first_valid_index()
    if valid_idx is None:
        # Nothing valid — return empty-ish series (avoid crashing)
        final_upper[:] = np.nan
        final_lower[:] = np.nan
        trend[:] = 1
        return trend, final_upper, final_lower, atr

    start_pos = ha_df.index.get_loc(valid_idx)

    # initialize at start_pos instead of always at 0
    final_upper.iat[start_pos] = basic_upper.iat[start_pos]
    final_lower.iat[start_pos] = basic_lower.iat[start_pos]
    trend.iat[start_pos] = 1  # arbitrary

    # For rows before start_pos, keep as NaN / default (or set to trend[ start_pos ] if desired)
    # Iterate from next position
    for i in range(start_pos + 1, len(ha_df)):
        fu_prev = final_upper.iat[i-1]
        fl_prev = final_lower.iat[i-1]
        close_prev = ha_close.iat[i-1]

        # If previous final bands are NaN (shouldn't be after start_pos), protect against it
        if np.isnan(fu_prev):
            fu_prev = basic_upper.iat[i-1]
        if np.isnan(fl_prev):
            fl_prev = basic_lower.iat[i-1]

        # final upper
        if (basic_upper.iat[i] < fu_prev) or (close_prev > fu_prev):
            final_upper.iat[i] = basic_upper.iat[i]
        else:
            final_upper.iat[i] = fu_prev

        # final lower
        if (basic_lower.iat[i] > fl_prev) or (close_prev < fl_prev):
            final_lower.iat[i] = basic_lower.iat[i]
        else:
            final_lower.iat[i] = fl_prev

        # determine trend
        if ha_close.iat[i] > final_upper.iat[i]:
            trend.iat[i] = 1
        elif ha_close.iat[i] < final_lower.iat[i]:
            trend.iat[i] = -1
        else:
            trend.iat[i] = trend.iat[i-1]

    # Optionally, forward-fill trend/bands for earlier rows (if you want numeric values from start)
    # final_upper = final_upper.ffill()
    # final_lower = final_lower.ffill()
    # trend = trend.ffill().astype('int8')

    return trend, final_upper, final_lower, atr

def generate_signals_from_trend(trend):
    """
    Buy when trend flips from -1 -> 1, Sell when trend flips from 1 -> -1
    Returns buy_signal, sell_signal boolean Series aligned with index
    """
    buy = (trend == 1) & (trend.shift(1) == -1)
    sell = (trend == -1) & (trend.shift(1) == 1)
    # fill NaN as False
    buy = buy.fillna(False)
    sell = sell.fillna(False)
    return buy, sell

def backtest_signals(df, buy_signal, sell_signal, atr):
    """
    Backtest with risk management:
      - Trailing stop-loss based on ATR
      - Position sizing based on risk percentage
      - Maximum loss streak limit
      - Enter long on buy, short on sell
      - Close on opposite signal or stop-loss
    Returns trades DataFrame with PnL per trade.
    """
    capital = INITIAL_CAPITAL
    loss_streak = 0
    position = 0
    entry_price = 0.0
    stop_loss = 0.0
    position_size = 0
    trades = []

    for i in range(len(df)):
        price = df['close'].iat[i]
        ts = df.index[i]
        current_atr = atr.iat[i] if not np.isnan(atr.iat[i]) else 0

        # Check for stop-loss hit
        if position == 1 and price <= stop_loss:
            pnl = (price - entry_price) * position_size
            capital += pnl
            if pnl < 0:
                loss_streak += 1
            else:
                loss_streak = 0
            trades.append({'type': 'stop_loss_exit', 'price': price, 'pnl': pnl, 'position_size': position_size, 'datetime': ts})
            position = 0
            position_size = 0
        elif position == -1 and price >= stop_loss:
            pnl = (entry_price - price) * position_size
            capital += pnl
            if pnl < 0:
                loss_streak += 1
            else:
                loss_streak = 0
            trades.append({'type': 'stop_loss_exit', 'price': price, 'pnl': pnl, 'position_size': position_size, 'datetime': ts})
            position = 0
            position_size = 0

        # Check for signals
        if buy_signal.iat[i] and position <= 0 and loss_streak < MAX_LOSS_STREAK:
            # close short if open
            if position == -1:
                pnl = (entry_price - price) * position_size
                capital += pnl
                if pnl < 0:
                    loss_streak += 1
                else:
                    loss_streak = 0
                trades.append({'type': 'close_short', 'price': price, 'pnl': pnl, 'position_size': position_size, 'datetime': ts})
            # calculate position size
            risk_distance = TRAILING_STOP_MULTIPLIER * current_atr
            if risk_distance > 0:
                position_size = (capital * RISK_PERCENTAGE) / risk_distance
            else:
                position_size = 1
            # open long
            position = 1
            entry_price = price
            stop_loss = price - risk_distance
            trades.append({'type': 'buy', 'price': price, 'position_size': position_size, 'datetime': ts})

        elif sell_signal.iat[i] and position >= 0 and loss_streak < MAX_LOSS_STREAK:
            if position == 1:
                pnl = (price - entry_price) * position_size
                capital += pnl
                if pnl < 0:
                    loss_streak += 1
                else:
                    loss_streak = 0
                trades.append({'type': 'close_long', 'price': price, 'pnl': pnl, 'position_size': position_size, 'datetime': ts})
            # calculate position size
            risk_distance = TRAILING_STOP_MULTIPLIER * current_atr
            if risk_distance > 0:
                position_size = (capital * RISK_PERCENTAGE) / risk_distance
            else:
                position_size = 1
            # open short
            position = -1
            entry_price = price
            stop_loss = price + risk_distance
            trades.append({'type': 'sell', 'price': price, 'position_size': position_size, 'datetime': ts})

        # Update trailing stop
        if position == 1:
            new_stop = price - TRAILING_STOP_MULTIPLIER * current_atr
            if new_stop > stop_loss:
                stop_loss = new_stop
        elif position == -1:
            new_stop = price + TRAILING_STOP_MULTIPLIER * current_atr
            if new_stop < stop_loss:
                stop_loss = new_stop

    # close any open position at last price
    if position == 1:
        pnl = (df['close'].iat[-1] - entry_price) * position_size
        capital += pnl
        if pnl < 0:
            loss_streak += 1
        else:
            loss_streak = 0
        trades.append({'type': 'close_long', 'price': df['close'].iat[-1], 'pnl': pnl, 'position_size': position_size, 'datetime': df.index[-1]})
    elif position == -1:
        pnl = (entry_price - df['close'].iat[-1]) * position_size
        capital += pnl
        if pnl < 0:
            loss_streak += 1
        else:
            loss_streak = 0
        trades.append({'type': 'close_short', 'price': df['close'].iat[-1], 'pnl': pnl, 'position_size': position_size, 'datetime': df.index[-1]})

    trades_df = pd.DataFrame(trades)
    return trades_df, capital

def compute_basic_metrics(trades_df):
    if trades_df.empty:
        return {}
    trades_df = trades_df.sort_values('datetime').reset_index(drop=True)
    # Filter to completed trades (closes and stop-loss exits have pnl)
    completed_trades = trades_df[trades_df['type'].str.contains('close|stop_loss')].copy()
    if completed_trades.empty:
        return {}
    cum_pnl = completed_trades['pnl'].cumsum()
    returns = cum_pnl.pct_change(fill_method=None).dropna()
    if returns.empty or returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    neg_returns = returns[returns < 0]
    if neg_returns.empty or neg_returns.std() == 0:
        sortino = 0.0
    else:
        sortino = returns.mean() / neg_returns.std() * np.sqrt(252)
    max_dd = (cum_pnl - cum_pnl.expanding().max()).min()
    pnl_sign = completed_trades['pnl'] > 0
    win_streak = 0
    loss_streak = 0
    current_win = 0
    current_loss = 0
    for sign in pnl_sign:
        if sign:
            current_win += 1
            current_loss = 0
            win_streak = max(win_streak, current_win)
        else:
            current_loss += 1
            current_win = 0
            loss_streak = max(loss_streak, current_loss)
    total_pnl = completed_trades['pnl'].sum()
    wins = completed_trades[completed_trades['pnl'] > 0]
    losses = completed_trades[completed_trades['pnl'] < 0]
    total_trades = len(completed_trades)
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0
    profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else float('inf')
    if len(completed_trades) > 1:
        days = (completed_trades['datetime'].max() - completed_trades['datetime'].min()).days
        ann_return = total_pnl / days * 365 if days > 0 else 0
    else:
        ann_return = 0
    metrics = {
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'max_win_streak': win_streak,
        'max_loss_streak': loss_streak,
        'annualized_return': ann_return
    }
    return metrics


# -------------------------
# run everything
# -------------------------
if __name__ == "__main__":
    df = load_and_prepare(DATA_DIR)
    ha_df = calculate_heikin_ashi(df)
    trend, upper, lower, atr = calculate_supertrend(ha_df, period=ATR_PERIOD, multiplier=ATR_MULTIPLIER, use_talib_atr=USE_TALIB_ATR)

    buy_signal, sell_signal = generate_signals_from_trend(trend)

    # attach to dataframe and save
    combined = df.join(ha_df)
    combined['atr'] = atr
    combined['supertrend_up'] = upper
    combined['supertrend_dn'] = lower
    combined['trend'] = trend
    combined['buy_signal'] = buy_signal.astype(int)
    combined['sell_signal'] = sell_signal.astype(int)
    combined.to_csv('supertrend_heikin_ashi_fixed_output.csv')
    print("Saved calculations to supertrend_heikin_ashi_fixed_output.csv")

    # backtest
    trades_df, final_capital = backtest_signals(df, buy_signal, sell_signal, atr)

    # Group by year and print metrics per year
    trades_df['year'] = trades_df['datetime'].dt.year
    years = sorted(trades_df['year'].unique())
    print(f"Years with trades: {years}")
    for year in years:
        year_trades = trades_df[trades_df['year'] == year]
        if not year_trades.empty:
            year_metrics = compute_basic_metrics(year_trades)
            print(f"\nMetrics for {year}:")
            print("-" * 50)
            for k, v in year_metrics.items():
                if isinstance(v, float):
                    print(f"{k:20}: {v:12.4f}")
                else:
                    print(f"{k:20}: {v:12}")

    # Overall metrics
    metrics = compute_basic_metrics(trades_df)
    metrics['final_capital'] = final_capital
    print("\nOverall Backtest Metrics:")
    print("-" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20}: {v:12.4f}")
        else:
            print(f"{k:20}: {v:12}")

