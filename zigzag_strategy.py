# zigzag_strategy.py (non-repainting version)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

FILEPATH = None

def compute_atr(df, length=14):
    h, l, c = df['high'], df['low'], df['close']
    tr1 = (h - l).abs()
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def compute_zigzag_nonrepainting(df, atr_len=14, s_med_len=50, alpha=1.0, beta=1.0):
    """
    Non-repainting ZigZag pivot detection.
    - D_t = max(alpha * ATR_t, beta * median_abs_return_t)
    - Maintain a candidate extreme (high or low). Confirm pivot when the current bar moves away
      from the candidate by >= D_t. The pivot is considered 'confirmed' on the bar that triggers it.
    - Writes:
        df['pivot_price_confirmed']  -> most recent confirmed pivot price (ffill)
        df['pivot_type_confirmed']   -> 'high'/'low' for the most recent confirmed pivot (ffill)
        df['D'] and df['atr']        -> diagnostic/used by backtester
    """
    df = df.copy()
    if len(df) == 0:
        df['pivot_price_confirmed'] = pd.Series(dtype=float)
        df['pivot_type_confirmed'] = pd.Series(dtype=object)
        df['D'] = pd.Series(dtype=float)
        df['atr'] = pd.Series(dtype=float)
        return df
    # compute ATR and median absolute returns
    high = df['high']; low = df['low']; close = df['close']
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_len, min_periods=1).mean().fillna(method='bfill')
    s_med = close.diff().abs().rolling(s_med_len, min_periods=1).median().fillna(method='bfill')
    D = np.maximum(alpha * atr, beta * s_med).clip(lower=1e-9)

    n = len(df)
    # arrays to store confirmed pivot info at the bar where it becomes known
    confirmed_price_at_bar = np.full(n, np.nan)
    confirmed_type_at_bar = np.full(n, None, dtype=object)

    # initialize candidate extreme using the first close
    last_confirmed_type = None  # None / 'high' / 'low'
    candidate_price = close.iat[0]
    candidate_idx = 0

    for t in range(1, n):
        price = close.iat[t]
        d_t = D.iat[t]

        if (last_confirmed_type is None) or (last_confirmed_type == 'low'):
            # searching for HIGH pivot
            if price > candidate_price:
                candidate_price = price
                candidate_idx = t
            # confirm high if price drops enough from candidate high
            if candidate_price - price >= d_t:
                # candidate_price at candidate_idx is confirmed as a HIGH pivot and becomes known at bar t
                confirmed_price_at_bar[t] = candidate_price
                confirmed_type_at_bar[t] = 'high'
                last_confirmed_type = 'high'
                # after confirming high, start candidate low from current price
                candidate_price = price
                candidate_idx = t
                continue

        if last_confirmed_type == 'high':
            # searching for LOW pivot
            if price < candidate_price:
                candidate_price = price
                candidate_idx = t
            # confirm low if price rises enough from candidate low
            if price - candidate_price >= d_t:
                confirmed_price_at_bar[t] = candidate_price
                confirmed_type_at_bar[t] = 'low'
                last_confirmed_type = 'low'
                # after confirming low, start candidate high from current price
                candidate_price = price
                candidate_idx = t
                continue

        # otherwise, no confirmation on this bar (leave NaN)

    # Forward fill confirmed pivots so each bar knows the last CONFIRMED pivot (if any).
    # A pivot that is confirmed on bar t is available at bar t and can be used to create signals
    # that will be executed at t+1 open (no future look-ahead).
    df['pivot_price_confirmed'] = pd.Series(confirmed_price_at_bar, index=df.index).ffill()
    df['pivot_type_confirmed'] = pd.Series(confirmed_type_at_bar, index=df.index).ffill()
    df['D'] = D
    df['atr'] = atr
    return df

def generate_signals_zigzag(df, pivot_buffer_pct: float = 0.0, use_intrabar_confirmation: bool = False):
    """
    Generate long/exit signals using only confirmed pivots (non-repainting).
    - pivot_buffer_pct: small buffer (e.g., 0.001 for 0.1%) to avoid micro-cross noise
    - If use_intrabar_confirmation True, will use high/low intrabar values for detecting crosses;
      otherwise it uses close-based comparisons (safe for next-bar execution).
    Output columns:
      df['long_signal']  -> True on bar t when entry should be scheduled for t+1 open
      df['exit_signal']  -> True on bar t when exit should be scheduled for t+1 open
    """
    df = df.copy()
    if 'pivot_price_confirmed' not in df.columns:
        raise ValueError("DataFrame must contain 'pivot_price_confirmed' (run compute_zigzag_nonrepainting first)")

    pivot = df['pivot_price_confirmed']
    pivot_prev = pivot.shift(1)

    buf = pivot * pivot_buffer_pct

    if use_intrabar_confirmation:
        # Use today's high/low to detect breakout intrabar (still non-repainting)
        above = df['high'] > (pivot + buf)
        above_prev = df['high'].shift(1) > (pivot_prev + buf)
        cross_up = (~above_prev) & (above)
        below = df['low'] < (pivot - buf)
        below_prev = df['low'].shift(1) < (pivot_prev - buf)
        cross_down = (~below_prev) & (below)
    else:
        # Use closes to detect cross; signals created on bar t are executed on t+1 open
        above = df['close'] > (pivot + buf)
        above_prev = df['close'].shift(1) > (pivot_prev + buf)
        cross_up = (~above_prev) & (above)
        below = df['close'] < (pivot - buf)
        below_prev = df['close'].shift(1) < (pivot_prev - buf)
        cross_down = (~below_prev) & (below)

    # Only consider long entries when the most recent confirmed pivot is a HIGH (breakout of pivot high)
    long_signal = cross_up & (df['pivot_type_confirmed'] == 'high')
    # Exit when price crosses back below pivot (or if cross_down happens)
    exit_signal = cross_down

    df['recent_pivot'] = pivot
    df['long_signal'] = long_signal.fillna(False)
    df['exit_signal'] = exit_signal.fillna(False)
    return df

def calculate_metrics(trades, eq):
    if trades.empty:
        return {}
    eq_series = eq['equity']
    wins = trades[trades['pnl'] > 0]
    losses = trades[trades['pnl'] < 0]
    win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0
    profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty and losses['pnl'].sum() != 0 else float('inf')
    total_return = (eq_series.iloc[-1] / eq_series.iloc[0] - 1) * 100
    # resample to daily for Sharpe/Sortino; if your data is intraday change resample accordingly
    daily_eq = eq_series.resample('D').last().dropna()
    daily_returns = daily_eq.pct_change().dropna()
    if not daily_returns.empty:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        sortino = daily_returns.mean() / daily_returns[daily_returns < 0].std() * np.sqrt(252) if len(daily_returns[daily_returns < 0]) > 0 and daily_returns[daily_returns < 0].std() > 0 else 0
    else:
        sharpe = 0
        sortino = 0
    peak = eq_series.expanding().max()
    drawdown = (eq_series - peak) / peak
    max_dd = drawdown.min() * 100
    days = (eq_series.index[-1] - eq_series.index[0]).days
    ann_return = ((eq_series.iloc[-1] / eq_series.iloc[0]) ** (365 / days) - 1) * 100 if days > 0 else 0
    metrics = {
        'Total Trades': len(trades),
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Average Win': avg_win,
        'Average Loss': avg_loss,
        'Total Return (%)': total_return,
        'Annualized Return (%)': ann_return,
        'Max Drawdown (%)': max_dd,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino
    }
    return metrics

def backtest(df, start_equity=100000, risk_pct=0.01, m_init=2.0, warmup=60):
    """
    Next-bar execution backtester.
    - signals at bar t are executed at t+1 open
    - intrabar stops are checked using bar low (stop hit) on the same bar
    """
    df = df.copy().reset_index()
    trades = []
    cash = start_equity
    pos = 0
    ep = ei = stop = None
    eq = []
    for i in range(warmup, len(df)-1):
        # mark-to-market at close of bar i
        eq.append({'dt': df.at[i, df.columns[0]], 'equity': cash + pos * df.at[i, 'close']})
        # If in position, check stop hit using low of bar i
        if pos > 0:
            if df.at[i, 'low'] <= stop:
                exit_price = stop
                pnl = (exit_price - ep) * pos
                cash += exit_price * pos
                trades.append({'entry_dt': df.at[ei, df.columns[0]],
                               'exit_dt': df.at[i, df.columns[0]],
                               'entry_price': ep,
                               'exit_price': exit_price,
                               'size': pos,
                               'pnl': pnl})
                pos = 0; ep = ei = stop = None
                continue
            # exit signal executed at open of this bar if it was signaled on previous bar
            if i-1 >= 0 and df.at[i-1, 'exit_signal']:
                exit_price = df.at[i, 'open']
                pnl = (exit_price - ep) * pos
                cash += exit_price * pos
                trades.append({'entry_dt': df.at[ei, df.columns[0]],
                               'exit_dt': df.at[i, df.columns[0]],
                               'entry_price': ep,
                               'exit_price': exit_price,
                               'size': pos,
                               'pnl': pnl})
                pos = 0; ep = ei = stop = None
                continue
        # If flat, check entry signal at bar i and execute at i+1 open
        if pos == 0 and df.at[i, 'long_signal'] and not pd.isna(df.at[i, 'recent_pivot']):
            exec_idx = i + 1
            if exec_idx >= len(df):
                break
            ep = df.at[exec_idx, 'open']
            ei = exec_idx
            atr = df.at[i, 'atr'] if 'atr' in df.columns else compute_atr(df.set_index(df.columns[0])).iloc[i]
            stop = ep - m_init * atr
            risk_unit = ep - stop
            size = np.floor((cash * risk_pct) / risk_unit) if risk_unit > 0 else 0
            if size <= 0:
                continue
            pos = size
            cash -= pos * ep
            continue
    # final mark
    final_total = cash + pos * df.at[len(df)-1, 'close']
    eq.append({'dt': df.at[len(df)-1, df.columns[0]], 'equity': final_total})
    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(eq).set_index('dt')
    return trades_df, eq_df

# ---------------------------
# Main execution
# ---------------------------
if __name__ == '__main__':
    # Example CSV path you used — adjust if necessary
    FILEPATH = "D:/Sushant/Fyers_AlgoTrade/Fyers_Data/Nifty50_Index/NIFTY50_INDEX_60_Min.csv"

    if FILEPATH:
        # load CSV with a Date column or index; adapt parsing if structure differs
        try:
            df = pd.read_csv(FILEPATH, parse_dates=True, index_col=0)
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns=lambda x: x.strip().lower())
            # sanitize typical variations
            if 'date' in df.columns and 'datetime' in df.columns:
                # if both exist, assume index is fine
                pass
            # drop nuisance columns
            df = df.drop(columns=['unnamed: 0', 'volume'], errors='ignore')
            # ensure index is datetime
            if df.index.dtype == 'int64':
                df.index = pd.to_datetime(df.index)
            else:
                df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep='first')]
            # optionally filter years - keep as you had originally if needed
            try:
                df = df[df.index.year >= 2021]
            except Exception:
                pass
        except Exception as e:
            print("Failed to load CSV; falling back to synthetic:", e)
            FILEPATH = None

    if 'df' in locals() and df.empty:
        print("Loaded DataFrame is empty; falling back to synthetic data")
        FILEPATH = None

    if not FILEPATH:
        rng = np.random.default_rng(5)
        n = 1000
        dt = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='D')
        steps = rng.normal(0.0003, 0.01, n)
        close = 100 + np.cumsum(steps)
        open_ = np.roll(close, 1); open_[0] = close[0]
        high = np.maximum(open_, close) + abs(rng.normal(0, 0.25, n))
        low = np.minimum(open_, close) - abs(rng.normal(0, 0.25, n))
        df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close}, index=dt)

    # Compute non-repainting pivots and signals
    df = compute_zigzag_nonrepainting(df, atr_len=14, s_med_len=50, alpha=1.0, beta=1.0)
    df = generate_signals_zigzag(df, pivot_buffer_pct=0.001, use_intrabar_confirmation=False)

    trades, eq = backtest(df, start_equity=100000, risk_pct=0.01, m_init=2.0, warmup=60)
    metrics = calculate_metrics(trades, eq)

    print("Initial Capital:", 100000)
    print("Trades:", len(trades))
    if not trades.empty:
        print(trades.head())
    else:
        print("No trades")
    print("Final equity:", eq['equity'].iloc[-1])
    print("\nTrading Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    # quick plots
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df['close'], label='close')
    # plot confirmed pivot points as markers (only at bars where pivot was confirmed)
    confirmed_mask = ~pd.isna(df['pivot_price_confirmed']) & (df['pivot_type_confirmed'] == 'high')
    plt.scatter(df.index[confirmed_mask], df['pivot_price_confirmed'][confirmed_mask], color='green', s=20, label='confirmed high pivots')
    confirmed_mask_low = ~pd.isna(df['pivot_price_confirmed']) & (df['pivot_type_confirmed'] == 'low')
    plt.scatter(df.index[confirmed_mask_low], df['pivot_price_confirmed'][confirmed_mask_low], color='red', s=20, label='confirmed low pivots')
    plt.legend(); plt.show()

    plt.figure(figsize=(12,3))
    eq['equity'].plot(title='Equity Curve')
    plt.show()
