"""
Modular backtester + VEMA indicator example.

Usage:
- Replace compute_indicator(...) with any indicator that writes these columns to df:
    df['indicator']         -> primary indicator series (e.g., VEMA)
    df['indicator_period']  -> (optional) effective period if adaptive (useful for debugging)
    df['indicator_slope']   -> slope (diff) of indicator or trend measure

- Then call run_backtest_from_csv(filepath) or run_demo().

Dependencies: pandas, numpy, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# ------------------------------
# Helper utilities
# ------------------------------
def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Simple ATR (rolling average of true range).
    Expects df has 'high','low','close'
    """
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
    
    # try to detect datetime
    if df.index.dtype.kind in 'OM':
        # index is non-datetime: maybe first column
        try:
            df = pd.read_csv(filepath, parse_dates=[0], index_col=0)
        except Exception:
            pass
    # lowercase columns
    df.columns = [c.lower() for c in df.columns]
    required = {'open','high','low','close'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")
    # ensure sorted
    df = df.sort_index()
    return df[['open','high','low','close'] + ([c for c in df.columns if c not in ['open','high','low','close']])]

def synthetic_ohlc(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt_index = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='D')
    steps = rng.normal(loc=0.0003, scale=0.01, size=n)
    close = 100 + np.cumsum(steps)
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, n))
    vol = rng.integers(100,1000,size=n)
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': vol}, index=dt_index)
    return df

# ------------------------------
# Indicator: VEMA (Volatility-Scaled EMA)
# ------------------------------
def compute_indicator_vema(df: pd.DataFrame,
                           N_base: int = 50,
                           N_min: int = 8,
                           atr_len: int = 14,
                           atr_ma_len: int = 50,
                           k: float = 1.5) -> pd.DataFrame:
    """
    Compute VEMA and store into df with standardized keys:
      df['indicator'] -> VEMA values
      df['indicator_period'] -> effective N per bar (float)
      df['indicator_slope'] -> vema.diff()
    This function can be swapped with any other compute_indicator_* that writes same column names.
    """
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
    # recursive EMA with per-bar alpha
    for t in range(1, len(prices)):
        a = alpha.iloc[t]
        vema[t] = a * prices[t] + (1.0 - a) * vema[t-1]

    df['indicator'] = vema
    df['indicator_period'] = N_eff
    df['indicator_slope'] = df['indicator'].diff()
    return df

# ------------------------------
# Generic signals generator (works with any indicator function that wrote 'indicator' & 'indicator_slope')
# ------------------------------
def generate_signals_from_indicator(df: pd.DataFrame, slope_eps: float = 1e-6) -> pd.DataFrame:
    """
    Builds basic long/exit signals based on indicator crossing and slope.
    - long_signal: detected on bar t (use for execution at t+1 open)
    - exit_signal: detected on bar t (execute at t+1 open)
    You can change logic here once for all indicators.
    """
    df = df.copy()
    if 'indicator' not in df.columns or 'indicator_slope' not in df.columns:
        raise ValueError("df must have 'indicator' and 'indicator_slope' columns")
    # crossing up on close
    df['above_ind'] = df['close'] > df['indicator']
    df['above_ind_prev'] = df['close'].shift(1) > df['indicator'].shift(1)
    df['cross_up'] = (~df['above_ind_prev']) & (df['above_ind'])
    df['cross_down'] = (df['above_ind_prev']) & (~df['above_ind'])
    df['ind_trending_up'] = df['indicator_slope'] > (slope_eps * df['close'])
    df['ind_trending_down'] = df['indicator_slope'] < -(slope_eps * df['close'])
    # Long only example
    df['long_signal'] = df['cross_up'] & df['ind_trending_up']
    df['exit_signal'] = df['cross_down'] | df['ind_trending_down']
    return df

# ------------------------------
# Next-bar backtester (long-only simple)
# ------------------------------
def backtest_next_bar(df: pd.DataFrame,
                      start_equity: float = 100000.0,
                      risk_pct: float = 0.01,
                      m_init: float = 2.0,
                      commission_per_trade: float = 0.0,
                      slippage_pct: float = 0.0,
                      warmup: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Next-bar execution backtester:
      - detects signals on bar t data and executes at bar t+1 open
      - position sizing uses ATR-based stop distance: size = equity * risk_pct / (entry_price - stop)
      - stop = entry_price - m_init * atr_at_entry
      - exits on stop hit (intrabar low) or exit_signal executed at next open
    Returns: trades_df, equity_curve_df, perf_stats
    """
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
    for i in range(warmup, n-1):  # we will potentially execute at i+1
        # Mark-to-market equity at close of bar i
        mkt_val = position * df.at[i, 'close']
        total_equity = cash + mkt_val
        equity_curve.append({'dt': df.at[i, df.columns[0]], 'equity': total_equity})

        # Check for intrabar stop loss if in position
        if position > 0:
            if df.at[i, 'low'] <= stop_price:
                # assume stop hit at stop_price
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
                # reset position
                position = 0.0; entry_price = None; entry_idx = None; stop_price = None; entry_atr = None
                # Continue to next bar if stop was hit
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
                position = 0.0; entry_price = None; entry_idx = None; stop_price = None; entry_atr = None
                continue

        # Check for entry signal if flat and execute at next bar's open
        if position == 0 and df.at[i, 'long_signal']:
            exec_idx = i+1
            if exec_idx >= n:
                break
            entry_price = df.at[exec_idx, 'open'] * (1.0 + slippage_pct)
            entry_idx = exec_idx
            entry_atr = df.at[i, 'atr'] if 'atr' in df.columns else compute_atr(df.set_index(df.columns[0])).iloc[i]
            stop_price = entry_price - m_init * entry_atr
            # Calculate position sizing based on risk per unit
            risk_per_unit = entry_price - stop_price
            if risk_per_unit <= 0:
                continue
            size = (equity * risk_pct) / risk_per_unit
            # floor to integer shares/units if appropriate
            size = np.floor(size)
            if size <= 0:
                continue
            position = size
            cash -= position * entry_price + commission_per_trade
            # Continue to next bar (we'll track exits on subsequent bars)
            continue

    # Final mark-to-market for remaining position
    final_mkt_val = position * df.at[n-1, 'close']
    final_total = cash + final_mkt_val
    equity_curve.append({'dt': df.at[n-1, df.columns[0]], 'equity': final_total})
    eq_df = pd.DataFrame(equity_curve).set_index('dt')
    trades_df = pd.DataFrame(trades)

    # Compute performance statistics
    returns = eq_df['equity'].pct_change().fillna(0)
    periods_per_year = 252  # assume daily - change if intraday
    cum_return = (eq_df['equity'].iloc[-1] / eq_df['equity'].iloc[0]) - 1.0
    try:
        cagr = (eq_df['equity'].iloc[-1] / eq_df['equity'].iloc[0]) ** (periods_per_year / len(eq_df)) - 1.0
    except Exception:
        cagr = np.nan
    max_dd = (eq_df['equity'].cummax() - eq_df['equity']).max()
    max_dd_pct = ((eq_df['equity'].cummax() - eq_df['equity']).max() / eq_df['equity'].cummax().max()) if not eq_df.empty else np.nan
    total_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0.0
    n_trades = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean() if not trades_df.empty else np.nan
    avg_win = trades_df.loc[trades_df['pnl']>0, 'pnl'].mean() if not trades_df.empty else np.nan
    avg_loss = trades_df.loc[trades_df['pnl']<=0, 'pnl'].mean() if not trades_df.empty else np.nan
    profit_factor = trades_df.loc[trades_df['pnl']>0, 'pnl'].sum() / (-trades_df.loc[trades_df['pnl']<=0, 'pnl'].sum()) if not trades_df.empty and (trades_df.loc[trades_df['pnl']<=0, 'pnl'].sum()!=0) else np.nan
    sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() != 0 else np.nan

    perf = {
        'starting_equity': start_equity,
        'ending_equity': eq_df['equity'].iloc[-1],
        'cumulative_return': cum_return,
        'cagr_approx': cagr,
        'max_drawdown': float(max_dd),
        'max_drawdown_pct': float(max_dd_pct),
        'total_pnl': float(total_pnl),
        'n_trades': n_trades,
        'win_rate': float(win_rate) if not np.isnan(win_rate) else np.nan,
        'avg_win': float(avg_win) if avg_win is not None else np.nan,
        'avg_loss': float(avg_loss) if avg_loss is not None else np.nan,
        'profit_factor': float(profit_factor) if profit_factor is not None else np.nan,
        'sharpe_approx': float(sharpe) if sharpe is not None else np.nan
    }
    return trades_df, eq_df, perf

# ------------------------------
# Convenience runner / demo
# ------------------------------
def run_vema_backtest(df: pd.DataFrame,
                      vema_params: Dict = None,
                      backtest_params: Dict = None,
                      plot: bool = True):
    """
    Compute indicator, signals and run backtest. Shows plots if plot=True.
    """
    if vema_params is None: vema_params = {}
    if backtest_params is None: backtest_params = {}

    df2 = compute_indicator_vema(df, **vema_params)
    df2 = generate_signals_from_indicator(df2)
    trades_df, eq_df, perf = backtest_next_bar(df2, **backtest_params)

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
    print(f"\nTrades: {len(trades_df)}")
    if not trades_df.empty:
        print(trades_df.head())

    if plot:
        plt.figure(figsize=(14,6))
        plt.plot(df2.index, df2['close'], label='close')
        plt.plot(df2.index, df2['indicator'], label='VEMA (indicator)')
        # mark trades
        if not trades_df.empty:
            for _, r in trades_df.iterrows():
                plt.scatter(r['entry_dt'], r['entry_price'], marker='^', color='g')
                plt.scatter(r['exit_dt'], r['exit_price'], marker='v', color='r')
        plt.legend(); plt.title('Price and Indicator (entries ^, exits v)')
        plt.show()

        plt.figure(figsize=(14,4))
        eq_df['equity'].plot(title='Equity Curve')
        plt.show()

    return trades_df, eq_df, perf, df2

def run_demo(use_csv: str = None):
    """
    Demo runner. If use_csv is path -> loads CSV; else runs synthetic sample.
    Adjust parameters here as desired.
    """
    if use_csv:
        df = load_csv(use_csv)
    # else:
    #     df = synthetic_ohlc(n=1000)

    vema_params = dict(N_base=50, N_min=8, atr_len=14, atr_ma_len=50, k=1.5)
    backtest_params = dict(start_equity=100000.0, risk_pct=0.01, m_init=2.0, commission_per_trade=0.0, slippage_pct=0.0, warmup=80)
    return run_vema_backtest(df, vema_params=vema_params, backtest_params=backtest_params, plot=True)

# ------------------------------
# If run as script, execute demo
# ------------------------------
if __name__ == "__main__":
    # Example: run_demo('path/to/your.csv') or run_demo() for synthetic
    trades, eq, perf, df_with_ind = run_demo(use_csv="D:/Sushant/Fyers_AlgoTrade/Fyers_Data/Nifty50_Index/NIFTY50_INDEX_60_Min.csv")
