"""
Modular backtester with 6 adaptive indicators.
Swap indicators by passing the desired compute_indicator_* function to `run_backtest`.

Dependencies: pandas, numpy, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple

# -----------------------
# Helpers
# -----------------------
def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def load_csv(filepath: str, datetime_col: str = None) -> pd.DataFrame:
    """
    Load CSV into standardized OHLCV DataFrame with datetime index.
    If datetime_col is None assumes first column is datetime or file already has index.
    """
    df = pd.read_csv(filepath, parse_dates=True)
    df = df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
    df = df.rename(columns={'Date': 'DateTime'})
    df = df.set_index('DateTime')
    if df.index.dtype == 'int64':
        # Assume DateTime is day of year for 2021
        df.index = pd.to_datetime(df.index.astype(str) + '-2021', format='%j-%Y')
    else:
        df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]
    df = df[df.index.year >= 2021]
    # Round all numerical columns to 2 decimal places
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].round(2)

    # DateTime should already be parsed by parse_dates=True
    # lowercase columns
    df.columns = [c.lower() for c in df.columns]
    required = {'open','high','low','close'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")
    # ensure sorted
    df = df.sort_index()
    return df[['open','high','low','close'] + ([c for c in df.columns if c not in ['open','high','low','close']])]

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - 100/(1+rs)

# -----------------------
# Indicator 1: VEMA
# -----------------------
def compute_indicator_vema(df: pd.DataFrame,
                           N_base: int = 50,
                           N_min: int = 8,
                           atr_len: int = 14,
                           atr_ma_len: int = 50,
                           k: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    df['atr'] = compute_atr(df, atr_len)
    df['atr_ma'] = df['atr'].rolling(atr_ma_len, min_periods=1).mean().replace(0, np.nan).fillna(method='bfill')
    v = (df['atr'] / df['atr_ma']).fillna(1.0)
    w_v = ((v - 1.0) * k).clip(0.0, 1.0)
    N_eff = (N_base * (1 - w_v) + N_min * w_v).clip(N_min, N_base)
    alpha = 2.0 / (N_eff + 1.0)

    prices = df['close'].values
    vema = np.full(len(prices), np.nan)
    vema[0] = prices[0]
    for t in range(1, len(prices)):
        a = alpha.iloc[t]
        vema[t] = a * prices[t] + (1.0 - a) * vema[t-1]

    df['indicator'] = vema
    df['indicator_period'] = N_eff
    df['indicator_slope'] = df['indicator'].diff()
    return df

# -----------------------
# Indicator 2: Regime-Switching (Volatility x Momentum)
# Produces 'indicator' = VEMA when regime is trend, else rolling median (mean-level) for mean-revert
# Also exposes 'regime' column and momentum for diagnostics.
# -----------------------
def compute_indicator_regime(df: pd.DataFrame,
                             atr_short: int = 14,
                             atr_long: int = 50,
                             ema_short: int = 12,
                             ema_long: int = 26,
                             V_low: float = 0.9,
                             V_high: float = 1.2,
                             mom_mult: float = 0.1,
                             vema_params: Dict = None) -> pd.DataFrame:
    df = df.copy()
    if vema_params is None: vema_params = {}
    df['atr_s'] = compute_atr(df, atr_short)
    df['atr_l'] = compute_atr(df, atr_long)
    df['V'] = (df['atr_s'] / df['atr_l']).fillna(1.0)
    df['ema_s'] = ema(df['close'], ema_short)
    df['ema_l'] = ema(df['close'], ema_long)
    df['momentum'] = df['ema_s'] - df['ema_l']
    df['momentum_norm'] = df['momentum'] / (df['atr_s'] + 1e-12)

    def decide_regime(v, m_norm):
        if v < V_low and abs(m_norm) < mom_mult:
            return 'LowVol_Sideways'
        if v < V_low and abs(m_norm) >= mom_mult:
            return 'LowVol_Trend'
        if v >= V_high and abs(m_norm) >= mom_mult:
            return 'HighVol_Trend'
        return 'HighVol_Sideways'

    df['regime'] = [decide_regime(v,m) for v,m in zip(df['V'], df['momentum_norm'])]

    # compute vema for trend behavior
    df = compute_indicator_vema(df, **({'N_base':50,'N_min':8,'atr_len':atr_short,'atr_ma_len':atr_long,'k':1.5} if vema_params==None else vema_params))
    # rolling central tendency for sideways (use median)
    df['sideways_lvl'] = df['close'].rolling(window=20, min_periods=1).median()

    # final indicator: choose vema for trend regimes, sideways_lvl for sideways regimes
    df['indicator'] = df['vema'].where(df['regime'].str.contains('Trend'), df['sideways_lvl'])
    df['indicator_slope'] = df['indicator'].diff()
    # keep debugging columns
    df['indicator_period'] = df.get('vema_period', np.nan)
    return df

# -----------------------
# Indicator 3: VATR - Breakout level + adaptive trailing helper series
# We produce 'indicator' as breakout level (break threshold) to be used by signal generator.
# Also produce trailing stop helper column 'vatr_trail' (but backtester uses generic stop computing).
# -----------------------
def compute_indicator_vatr(df: pd.DataFrame,
                           atr_len: int = 14,
                           atr_ma_len: int = 50,
                           lookback: int = 20,
                           m_b_base: float = 1.0,
                           s: float = 1.0,
                           m_ts_base: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    df['atr'] = compute_atr(df, atr_len)
    df['atr_ma'] = df['atr'].rolling(atr_ma_len, min_periods=1).mean().fillna(method='bfill')
    V = (df['atr'] / df['atr_ma']).fillna(1.0)
    df['m_b'] = (m_b_base * (1 + s * (V - 1.0))).clip(0.3, 5.0)
    df['break_high'] = df['high'].rolling(lookback, min_periods=1).max()
    df['indicator'] = df['break_high'] + df['m_b'] * df['atr']  # breakout level
    # trailing stop helper: initial trail = NaN; we will compute a conservative trail column: close - m_ts * atr
    df['vatr_trail'] = df['close'] - (m_ts_base * (1 + s*(V-1.0))) * df['atr']
    df['indicator_slope'] = df['indicator'].diff()
    df['indicator_period'] = lookback
    return df

# -----------------------
# Indicator 4: Kalman filter trend (adaptive Q/R)
# We'll implement a simple 2-state Kalman and return filter level as 'indicator'
# -----------------------
def compute_indicator_kalman(df: pd.DataFrame,
                             atr_len: int = 14,
                             atr_ma_len: int = 50,
                             Q0_scale: float = 1e-4,
                             vel_scale: float = 1e-6) -> pd.DataFrame:
    df = df.copy()
    price = df['close'].values
    n = len(price)
    df['atr'] = compute_atr(df, atr_len)
    df['atr_ma'] = df['atr'].rolling(atr_ma_len, min_periods=1).mean().fillna(method='bfill')
    V = (df['atr'] / df['atr_ma']).fillna(1.0).ewm(span=5).mean().clip(0.5, 3.0).values

    # initialize
    x = np.zeros((2,))  # [level, vel]
    x[0] = price[0]; x[1] = 0.0
    P = np.eye(2) * 0.1
    F = np.array([[1,1],[0,1]])
    H = np.array([[1.0, 0.0]])
    mu = np.zeros(n); vel = np.zeros(n)
    R0 = max(1e-6, np.nanstd(np.diff(price))**2)
    for t in range(n):
        Q = np.diag([Q0_scale, vel_scale]) * V[t]
        R = R0 / V[t]
        # predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        z = price[t]
        y = z - (H @ x_pred)[0]
        S = (H @ P_pred @ H.T)[0,0] + R
        K = (P_pred @ H.T).flatten() / S
        x = x_pred + K * y
        P = (np.eye(2) - np.outer(K, H[0])) @ P_pred
        mu[t] = x[0]; vel[t] = x[1]
    df['indicator'] = mu
    df['kf_vel'] = vel
    df['indicator_slope'] = df['indicator'].diff()
    df['indicator_period'] = np.nan
    return df

# -----------------------
# Indicator 5: Adaptive RSI (volatility-normalized)
# We'll produce 'indicator' as adaptive RSI values (0-100); signals treat it as oscillator (crosses)
# -----------------------
def compute_indicator_adaptive_rsi(df: pd.DataFrame,
                                   n_base: int = 14,
                                   n_min: int = 7,
                                   n_max: int = 40,
                                   atr_len: int = 14,
                                   atr_ma_len: int = 50,
                                   s: float = 0.5,
                                   a: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    df['atr'] = compute_atr(df, atr_len)
    df['atr_ma'] = df['atr'].rolling(atr_ma_len, min_periods=1).mean().fillna(method='bfill')
    V = (df['atr'] / df['atr_ma']).fillna(1.0)
    n_t = (n_base * (1 + s*(V - 1.0))).round().clip(n_min, n_max).astype(int)
    # Precompute RSIs for candidate windows (to avoid dynamically recomputing expensive smoothing)
    candidates = sorted(list(set([n_min, n_base, n_max, 21, 28])))
    for n in candidates:
        df[f'rsi_{n}'] = rsi(df['close'], n)
    # select per-row
    rsi_adapt = []
    for idx, n in zip(df.index, n_t):
        rsi_adapt.append(df.at[idx, f"rsi_{n}"])
    df['indicator'] = pd.Series(rsi_adapt, index=df.index)
    # indicator_slope is simply diff of RSI
    df['indicator_slope'] = df['indicator'].diff()
    # adaptive bands (for reference)
    df['rsi_lower'] = 30 - a * (V - 1.0) * 10
    df['rsi_upper'] = 70 + a * (V - 1.0) * 10
    df['indicator_period'] = n_t
    return df

# -----------------------
# Indicator 6: Dynamic ZigZag / Pivot Detector
# We'll return pivot level as 'indicator' (last confirmed pivot price), and slope as diff
# -----------------------
def compute_indicator_zigzag(df: pd.DataFrame,
                             atr_len: int = 14,
                             s_med_len: int = 50,
                             alpha: float = 1.0,
                             beta: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    df['atr'] = compute_atr(df, atr_len)
    df['s_med'] = df['close'].diff().abs().rolling(s_med_len, min_periods=1).median().fillna(method='bfill')
    D = np.maximum(alpha * df['atr'], beta * df['s_med']).clip(lower=1e-6)
    df['D'] = D
    n = len(df)
    pivots_idx = []  # (idx, 'high'/'low', price)

    last_high = df['close'].iloc[0]; last_high_idx = 0
    last_low = df['close'].iloc[0]; last_low_idx = 0

    for t in range(1, n):
        price = df['close'].iat[t]
        d = df['D'].iat[t]
        if price > last_high:
            last_high = price; last_high_idx = t
        if price < last_low:
            last_low = price; last_low_idx = t
        # detect confirmed high pivot (price drops by >= d from last_high)
        if last_high - price >= d:
            pivots_idx.append((last_high_idx, 'high', last_high))
            # reset after pivot
            last_low = price; last_low_idx = t
            last_high = price; last_high_idx = t
        # detect confirmed low pivot (price rises by >= d from last_low)
        if price - last_low >= d:
            pivots_idx.append((last_low_idx, 'low', last_low))
            last_high = price; last_high_idx = t
            last_low = price; last_low_idx = t

    # build pivot series: for each bar, indicator=most recent pivot price
    pivot_price = np.full(n, np.nan)
    pivot_type = np.full(n, np.nan)
    last_p = np.nan
    last_t = np.nan
    pivot_map = {idx: (typ, price) for idx, typ, price in pivots_idx}
    for t in range(n):
        if t in pivot_map:
            typ, price = pivot_map[t]
            last_p = price
            last_t = typ
        pivot_price[t] = last_p
        pivot_type[t] = last_t
    df['indicator'] = pivot_price
    df['pivot_type'] = pivot_type
    df['indicator_slope'] = pd.Series(df['indicator']).fillna(method='ffill').diff()
    df['indicator_period'] = df['D']
    return df

# -----------------------
# Generic signal generator (indicator-agnostic)
# - For oscillators (like RSI) you might want different logic; we provide base behavior:
#   - Use crossing above indicator for long entries (works if indicator is MA/level/breakout)
#   - For RSI-like indicator, user may override or use threshold-based signals externally.
# -----------------------
def generate_signals_from_indicator(df: pd.DataFrame, slope_eps: float = 1e-6) -> pd.DataFrame:
    df = df.copy()
    if 'indicator' not in df.columns:
        raise ValueError("df must contain 'indicator' column")
    df['above_ind'] = df['close'] > df['indicator']
    df['above_ind_prev'] = df['above_ind'].shift(1).fillna(False)
    df['cross_up'] = (~df['above_ind_prev']) & (df['above_ind'])
    df['cross_down'] = (df['above_ind_prev']) & (~df['above_ind'])
    df['ind_slope_pos'] = df.get('indicator_slope', df['indicator'].diff()) > (slope_eps * df['close'])
    df['ind_slope_neg'] = df.get('indicator_slope', df['indicator'].diff()) < -(slope_eps * df['close'])
    # Default long/exit
    df['long_signal'] = df['cross_up'] & df['ind_slope_pos']
    df['exit_signal'] = df['cross_down'] | df['ind_slope_neg']
    # Special-case: if indicator is RSI (0-100), use adaptive thresholds if columns exist (rsi_lower)
    if 'rsi_lower' in df.columns and 'rsi_upper' in df.columns:
        # override: in sideways, long when RSI < lower; exit when RSI > 50
        df['long_signal'] = (df['indicator'] < df['rsi_lower'])
        df['exit_signal'] = (df['indicator'] > 50)
    return df

# -----------------------
# Backtester (next-bar execution) - unchanged but accepts indicator function externally via precomputed df
# -----------------------
def backtest_next_bar(df: pd.DataFrame,
                      start_equity: float = 100000.0,
                      risk_pct: float = 0.01,
                      m_init: float = 2.0,
                      commission_per_trade: float = 0.0,
                      slippage_pct: float = 0.0,
                      warmup: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    df = df.copy().reset_index()
    n = len(df)
    trades = []
    equity = start_equity
    cash = equity
    position = 0.0
    entry_price = None
    entry_idx = None
    entry_atr = None
    stop_price = None

    equity_curve = []

    # Main backtesting loop: iterate over each bar starting from warmup period
    for i in range(warmup, n-1):
        # Mark-to-market equity at close of bar i
        mkt_val = position * df.at[i, 'close']
        total_equity = cash + mkt_val
        equity_curve.append({'dt': df.at[i, df.columns[0]], 'equity': total_equity})

        # Check for intrabar stop loss if in position
        if position > 0:
            if df.at[i, 'low'] <= stop_price:
                exit_price = stop_price
                pnl = (exit_price - entry_price) * position - commission_per_trade
                cash += exit_price * position
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_dt': df.at[entry_idx, df.columns[0]],
                    'exit_dt': df.at[i, df.columns[0]],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': position,
                    'pnl': pnl
                })
                # Continue to next bar if stop was hit
                position = 0.0; entry_price = entry_idx = stop_price = entry_atr = None
                continue
            # Check for exit signal from previous bar and execute at current open
            if i-1 >= 0 and df.at[i-1, 'exit_signal']:
                exit_price = df.at[i, 'open'] * (1.0 - slippage_pct)
                pnl = (exit_price - entry_price) * position - commission_per_trade
                cash += exit_price * position
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_dt': df.at[entry_idx, df.columns[0]],
                    'exit_dt': df.at[i, df.columns[0]],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': position,
                    'pnl': pnl
                })
                position = 0.0; entry_price = entry_idx = stop_price = entry_atr = None
                continue

        # Check for entry signal if flat and execute at next bar's open
        if position == 0 and df.at[i, 'long_signal']:
            exec_idx = i+1
            if exec_idx >= n: break
            entry_price = df.at[exec_idx, 'open'] * (1.0 + slippage_pct)
            entry_idx = exec_idx
            entry_atr = df.at[i, 'atr'] if 'atr' in df.columns else compute_atr(df.set_index(df.columns[0])).iloc[i]
            stop_price = entry_price - m_init * entry_atr
            # Calculate position sizing based on risk per unit
            risk_per_unit = entry_price - stop_price
            if risk_per_unit <= 0:
                continue
            size = (equity * risk_pct) / risk_per_unit
            size = np.floor(size)
            if size <= 0:
                continue
            position = size
            # Continue to next bar (we'll track exits on subsequent bars)
            cash -= position * entry_price + commission_per_trade
            continue

    # Final mark-to-market for remaining position
    final_mkt_val = position * df.at[n-1, 'close']
    final_total = cash + final_mkt_val
    equity_curve.append({'dt': df.at[n-1, df.columns[0]], 'equity': final_total})
    eq_df = pd.DataFrame(equity_curve).set_index('dt')
    trades_df = pd.DataFrame(trades)

    # Compute performance statistics
    returns = eq_df['equity'].pct_change().fillna(0)
    periods_per_year = 252
    cum_return = (eq_df['equity'].iloc[-1] / eq_df['equity'].iloc[0]) - 1.0
    try:
        cagr = (eq_df['equity'].iloc[-1] / eq_df['equity'].iloc[0]) ** (periods_per_year / len(eq_df)) - 1.0
    except:
        cagr = np.nan
    max_dd = (eq_df['equity'].cummax() - eq_df['equity']).max()
    max_dd_pct = ((eq_df['equity'].cummax() - eq_df['equity']).max() / eq_df['equity'].cummax().max()) if not eq_df.empty else np.nan
    total_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0.0
    n_trades = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean() if not trades_df.empty else np.nan
    n_winning_trades = (trades_df['pnl'] > 0).sum() if not trades_df.empty else 0
    n_losing_trades = (trades_df['pnl'] <= 0).sum() if not trades_df.empty else 0
    avg_win = trades_df.loc[trades_df['pnl']>0, 'pnl'].mean() if not trades_df.empty else np.nan
    avg_loss = trades_df.loc[trades_df['pnl']<=0, 'pnl'].mean() if not trades_df.empty else np.nan
    profit_factor = trades_df.loc[trades_df['pnl']>0, 'pnl'].sum() / (-trades_df.loc[trades_df['pnl']<=0, 'pnl'].sum()) if not trades_df.empty and (trades_df.loc[trades_df['pnl']<=0, 'pnl'].sum()!=0) else np.nan
    sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() != 0 else np.nan

    # Calculate max winning and losing streaks
    if not trades_df.empty:
        pnl_sign = (trades_df['pnl'] > 0).astype(int)
        win_streak = pnl_sign.groupby((pnl_sign != pnl_sign.shift()).cumsum()).cumsum()
        max_win_streak = win_streak.max() if (pnl_sign == 1).any() else 0
        loss_streak = (1 - pnl_sign).groupby((pnl_sign != pnl_sign.shift()).cumsum()).cumsum()
        max_loss_streak = loss_streak.max() if (pnl_sign == 0).any() else 0
    else:
        max_win_streak = 0
        max_loss_streak = 0

    perf = {
        'starting_equity': start_equity,
        'ending_equity': float(eq_df['equity'].iloc[-1]),
        'cumulative_return': float(cum_return),
        'cagr_approx': float(cagr) if not np.isnan(cagr) else np.nan,
        'max_drawdown': float(max_dd),
        'max_drawdown_pct': float(max_dd_pct) if not np.isnan(max_dd_pct) else np.nan,
        'total_pnl': float(total_pnl),
        'n_trades': int(n_trades),
        'n_winning_trades': int(n_winning_trades),
        'n_losing_trades': int(n_losing_trades),
        'max_win_streak': int(max_win_streak),
        'max_loss_streak': int(max_loss_streak),
        'win_rate': float(win_rate) if not np.isnan(win_rate) else np.nan,
        'avg_win': float(avg_win) if not np.isnan(avg_win) else np.nan,
        'avg_loss': float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        'profit_factor': float(profit_factor) if not np.isnan(profit_factor) else np.nan,
        'sharpe_approx': float(sharpe) if not np.isnan(sharpe) else np.nan
    }
    return trades_df, eq_df, perf

# -----------------------
# Runner that accepts any indicator function
# -----------------------
def run_backtest(df: pd.DataFrame,
                 indicator_fn: Callable[[pd.DataFrame, Dict], pd.DataFrame],
                 indicator_params: Dict = None,
                 backtest_params: Dict = None,
                 plot: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, pd.DataFrame]:
    if indicator_params is None: indicator_params = {}
    if backtest_params is None: backtest_params = {}
    df2 = indicator_fn(df, **indicator_params)
    df2 = generate_signals_from_indicator(df2)
    trades, eq, perf = backtest_next_bar(df2, **backtest_params)
    if plot:
        plt.figure(figsize=(13,6))
        plt.plot(df2.index, df2['close'], label='close')
        if 'indicator' in df2.columns:
            plt.plot(df2.index, df2['indicator'], label='indicator')
        if not trades.empty:
            for _, r in trades.iterrows():
                plt.scatter(r['entry_dt'], r['entry_price'], marker='^', color='g')
                plt.scatter(r['exit_dt'], r['exit_price'], marker='v', color='r')
        plt.legend(); plt.title(f"Indicator: {indicator_fn.__name__}")
        plt.show()

        plt.figure(figsize=(13,4))
        eq['equity'].plot(title='Equity Curve')
        plt.show()
    return trades, eq, perf, df2

# -----------------------
# Demo helper (synthetic data)
# -----------------------
# def synthetic_ohlc(n: int = 1000, seed: int = 42) -> pd.DataFrame:
#     rng = np.random.default_rng(seed)
#     dt_index = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='D')
#     steps = rng.normal(loc=0.0003, scale=0.01, size=n)
#     close = 100 + np.cumsum(steps)
#     open_ = np.roll(close,1); open_[0] = close[0]
#     high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, n))
#     low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, n))
#     vol = rng.integers(100,1000,size=n)
#     df = pd.DataFrame({'open':open_, 'high':high, 'low':low, 'close':close, 'volume':vol}, index=dt_index)
#     return df

# -----------------------
# Example usage (run as script or import functions)
# -----------------------
if __name__ == "__main__":
    # generate synthetic data or replace with pd.read_csv
    # df = synthetic_ohlc(1000)
    csv_file_path = "D:/Sushant/Fyers_AlgoTrade/Fyers_Data/Nifty50_Index/NIFTY50_INDEX_60_Min.csv"
    df = load_csv(csv_file_path)

    # select indicator function and params
    indicator_fn = compute_indicator_regime   # swap to compute_indicator_regime, compute_indicator_vatr, compute_indicator_kalman, compute_indicator_adaptive_rsi, compute_indicator_zigzag
    indicator_params = {}
    backtest_params = dict(start_equity=100000.0, risk_pct=0.01, m_init=2.0, commission_per_trade=0.0, slippage_pct=0.0, warmup=80)

    trades, eq, perf, df_ind = run_backtest(df, indicator_fn, indicator_params, backtest_params, plot=True)

    percentage_keys = {'cumulative_return', 'cagr_approx', 'max_drawdown_pct', 'win_rate'}
    print("Performance summary:")
    for k, v in perf.items():
        if k in percentage_keys:
            if pd.isna(v):
                formatted = "NaN"
            else:
                formatted = f"{v * 100:.2f}%"
        else:
            if isinstance(v, float):
                formatted = f"{v:.2f}"
            else:
                formatted = str(v)
        print(f"  {k}: {formatted}")
    print(f"\nTrades: {len(trades)}")
    if not trades.empty:
        print(trades.head())
