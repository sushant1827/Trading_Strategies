#!/usr/bin/env python3
"""
adaptive_supertrend_possize_full.py

Single train/test adaptive SuperTrend using KMeans regimes.
Parameter selection on TRAIN uses a realistic position-sized ATR-stop backtester,
and the selected per-regime params are applied row-wise to the TEST set.
Final backtest on TEST uses position sizing as well.

Outputs saved to OUTDIR:
 - final_equity.csv
 - final_trades.csv
 - final_summary_extended.csv

Author: ChatGPT
"""

import os, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------- USER CONFIG ----------------
FILEPATH = "Nifty50_Index\\NIFTY50_INDEX_30_Min.csv"   # change if needed
OUTDIR = "Lorentzian_Classification\\adaptive_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# adaptive grid (keep small when testing)
ADAPT_GRID_ATR_PERIODS = [24]
ADAPT_GRID_MULTIPLIERS = [0.9]

# position-sizing / backtest defaults
INITIAL_CAPITAL = 100000.0
RISK_PCT = 0.02            # fraction of equity risked per trade (2%)
ATR_STOP_MULT = 2.0        # stop distance = ATR * ATR_STOP_MULT
MAX_POS_FRACTION = 0.2     # maximum fraction of equity allocated to a single trade
SLIPPAGE_PCT = 0.0005
COMMISSION_PCT = 0.0005

KM_REGIMES = 3

# frequency approximation for 30-min bars (used for annualization)
FREQ_PER_YEAR = 252 * 13   # approx 13 half-hour bars per trading day
# ---------------------------------------------

def read_ohlc(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # common rename mapping
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("date") or lc in ("datetime","timestamp"):
            rename[c] = "datetime"
        elif lc == "open":
            rename[c] = "open"
        elif lc == "high":
            rename[c] = "high"
        elif lc == "low":
            rename[c] = "low"
        elif lc in ("close","last"):
            rename[c] = "close"
        elif lc in ("volume","vol"):
            rename[c] = "volume"
    df = df.rename(columns=rename)
    required = ["datetime","open","high","low","close"]
    if not all(k in df.columns for k in required):
        raise ValueError(f"CSV must contain columns: {required}. Found: {list(df.columns)}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df

def atr(series_high, series_low, series_close, n=14):
    h = series_high; l = series_low; c = series_close
    tr1 = (h - l).abs()
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def supertrend_df(df_local, period=10, multiplier=3.0):
    hl2 = (df_local['high'] + df_local['low']) / 2.0
    atr_s = atr(df_local['high'], df_local['low'], df_local['close'], n=period)
    upper = hl2 + multiplier * atr_s
    lower = hl2 - multiplier * atr_s
    final_upper = upper.copy(); final_lower = lower.copy()
    st = np.zeros(len(df_local)); direction = np.ones(len(df_local))
    for i in range(len(df_local)):
        if i == 0:
            final_upper.iloc[i] = upper.iloc[i]
            final_lower.iloc[i] = lower.iloc[i]
            st[i] = final_upper.iloc[i]; direction[i] = -1
            continue
        if (upper.iloc[i] < final_upper.iloc[i-1]) or (df_local['close'].iloc[i-1] > final_upper.iloc[i-1]):
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
        if (lower.iloc[i] > final_lower.iloc[i-1]) or (df_local['close'].iloc[i-1] < final_lower.iloc[i-1]):
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
        if st[i-1] == final_upper.iloc[i-1]:
            if df_local['close'].iloc[i] <= final_upper.iloc[i]:
                st[i] = final_upper.iloc[i]; direction[i] = -1
            else:
                st[i] = final_lower.iloc[i]; direction[i] = 1
        else:
            if df_local['close'].iloc[i] >= final_lower.iloc[i]:
                st[i] = final_lower.iloc[i]; direction[i] = 1
            else:
                st[i] = final_upper.iloc[i]; direction[i] = -1
    return pd.DataFrame({'st': st, 'dir': direction, 'atr': atr_s})

def backtest_position_sizing(df_local, st_df_local,
                             initial_capital=INITIAL_CAPITAL,
                             risk_pct=RISK_PCT,
                             atr_stop_mult=ATR_STOP_MULT,
                             max_pos_fraction=MAX_POS_FRACTION,
                             slippage_pct=SLIPPAGE_PCT,
                             commission_pct=COMMISSION_PCT,
                             absolute_units_cap=100000):
    """
    Robust position-sized backtest with:
     - minimum stop distance guard to avoid tiny denominators,
     - exposure cap (max_pos_fraction in $) and absolute units cap,
     - records entry_units for each trade,
     - closes any open position at the end (so trades P&L sums to equity delta).
    Returns: (equity_series, trades_list)
    """
    cash = float(initial_capital)
    pos = 0.0
    entry_price = None
    entry_units = 0.0
    stop_price = None
    trades = []
    equity_vals = []

    # try use atr from st_df_local if present
    if 'atr' in st_df_local.columns:
        atr_series = st_df_local['atr']
    else:
        atr_series = atr(df_local['high'], df_local['low'], df_local['close'], n=14)

    for i in range(1, len(df_local)):
        prev = st_df_local['dir'].iloc[i-1]
        curr = st_df_local['dir'].iloc[i]
        openp = float(df_local['open'].iloc[i])
        lowp  = float(df_local['low'].iloc[i])
        closep= float(df_local['close'].iloc[i])

        # ENTRY: prev -1 -> curr 1 (no existing pos)
        if prev == -1 and curr == 1 and pos == 0:
            atr_val = float(atr_series.iloc[i-1]) if i-1 >= 0 else float(atr_series.iloc[i])
            stop_dist = atr_stop_mult * atr_val

            # enforce a minimum stop distance to avoid tiny denominators
            min_stop_dist = max( max(0.001 * openp, 0.5), 0.0001 * openp )  # at least 0.1% or 0.5 absolute
            stop_dist = max(stop_dist, min_stop_dist)

            entry_px = openp * (1.0 + slippage_pct + commission_pct)
            stop_px = entry_px - stop_dist
            if stop_px <= 0 or entry_px <= stop_px:
                # invalid stop — skip entry
                pass
            else:
                dollar_risk = cash * risk_pct
                # candidate units from risk formula
                units_from_risk = dollar_risk / max(1e-9, (entry_px - stop_px))
                # cap exposure by max_pos_fraction
                current_equity = cash + pos * closep
                max_dollars = current_equity * max_pos_fraction
                max_units_from_exposure = max_dollars / entry_px if entry_px > 0 else 0.0
                # enforce an absolute unit cap
                units = min(units_from_risk, max_units_from_exposure, absolute_units_cap)
                # floor to integer if trading discrete contracts (optional); keep float for continuous
                if units > 0:
                    pos = units
                    entry_price = entry_px
                    entry_units = units
                    stop_price = stop_px
                    cash -= units * entry_px  # pay for position

        # If we have a position, check for intrabar stop hit and flip exit
        if pos > 0:
            # intrabar stop
            if lowp <= stop_price:
                exit_px = stop_price * (1.0 - slippage_pct - commission_pct)
                proceeds = pos * exit_px
                profit = proceeds - (entry_units * entry_price)
                trades.append({
                    'entry_idx': None, 'exit_idx': i,
                    'entry_price': entry_price, 'exit_price': exit_px,
                    'units': entry_units, 'profit': profit, 'exit_type': 'stop'
                })
                cash += proceeds
                pos = 0.0; entry_price = None; entry_units = 0.0; stop_price = None

            # flip exit at open
            elif prev == 1 and curr == -1 and pos > 0:
                exit_px = openp * (1.0 - slippage_pct - commission_pct)
                proceeds = pos * exit_px
                profit = proceeds - (entry_units * entry_price)
                trades.append({
                    'entry_idx': None, 'exit_idx': i,
                    'entry_price': entry_price, 'exit_price': exit_px,
                    'units': entry_units, 'profit': profit, 'exit_type': 'flip'
                })
                cash += proceeds
                pos = 0.0; entry_price = None; entry_units = 0.0; stop_price = None

        # mark-to-market equity
        equity_val = pos * closep if pos > 0 else cash
        equity_vals.append(equity_val)

    # After loop: if a position remains open, close it at the last available close price and log the trade
    if pos > 0:
        final_close = float(df_local['close'].iloc[-1])
        exit_px = final_close * (1.0 - slippage_pct - commission_pct)
        proceeds = pos * exit_px
        profit = proceeds - (entry_units * entry_price)
        trades.append({
            'entry_idx': None, 'exit_idx': len(df_local)-1,
            'entry_price': entry_price, 'exit_price': exit_px,
            'units': entry_units, 'profit': profit, 'exit_type': 'final_close'
        })
        cash += proceeds
        pos = 0.0
        equity_vals[-1] = cash  # last equity reflects closure

    # build equity series aligned to df_local index
    equity_series = pd.Series([initial_capital] + equity_vals, index=df_local.index[:len(equity_vals)+1]).ffill()

    # Sanity check: trades P&L sum should equal equity delta (within tiny tol)
    trades_sum = sum([float(t.get('profit', 0.0)) for t in trades]) if len(trades)>0 else 0.0
    equity_delta = float(equity_series.iloc[-1] - equity_series.iloc[0])
    # small tolerance
    if abs(trades_sum - equity_delta) > max(1.0, abs(equity_delta)*1e-6):
        # We print a helpful warning (do not raise) — you can change to raise if you prefer strict enforcement
        print("WARNING: P&L mismatch in backtest_position_sizing: sum(trade.profit) = {:.2f}, equity delta = {:.2f}".format(trades_sum, equity_delta))

    return equity_series, trades


def compute_metrics_and_diagnostics(equity_series, trades, freq_per_year=FREQ_PER_YEAR):
    """
    Robust metrics + diagnostics.
    equity_series: pd.Series (index should be datetimes)
    trades: list of dicts with 'profit' numeric
    """
    # Ensure equity index is datetime for time-based calcs
    try:
        eq_index = pd.to_datetime(equity_series.index)
        eq = pd.Series(equity_series.values, index=eq_index).astype(float)
    except Exception:
        eq = pd.Series(equity_series.values).astype(float)

    start_val = float(eq.iloc[0])
    end_val = float(eq.iloc[-1])
    equity_delta = end_val - start_val
    total_return = (end_val / start_val) - 1.0

    # periodic simple returns
    pct = eq.pct_change().fillna(0)

    # Robust annualized return: use CAGR (multiplicative) and also arithmetic annualized mean for sharpe
    days = (eq.index[-1] - eq.index[0]).days if isinstance(eq.index[0], pd.Timestamp) else None
    if days is None or days <= 0:
        years = max(1.0 / freq_per_year, 1.0/252)
    else:
        years = max(days / 365.25, 1.0 / freq_per_year)
    cagr = (end_val / start_val) ** (1.0 / years) - 1.0 if start_val > 0 else float('nan')

    # annualized arithmetic return & vol for Sharpe
    arith_ann_return = float(pct.mean() * freq_per_year)
    ann_vol = float(pct.std() * math.sqrt(freq_per_year))

    sharpe = float(arith_ann_return / ann_vol) if ann_vol > 0 else float('nan')

    # Sortino: use downside deviation (only negative returns)
    downside = pct.copy()
    downside[downside > 0] = 0.0
    downside_std = float(downside.std() * math.sqrt(freq_per_year))
    sortino = float(arith_ann_return / downside_std) if downside_std > 0 else float('nan')

    # Max drawdown (fraction)
    running_max = eq.cummax()
    drawdowns = eq / running_max - 1.0
    max_dd = float(drawdowns.min())

    # Trades stats
    profits = np.array([float(t.get('profit', 0.0)) for t in trades], dtype=float)
    total_trades = int(len(profits))
    wins = profits[profits > 0] if total_trades > 0 else np.array([])
    losses = profits[profits <= 0] if total_trades > 0 else np.array([])

    win_rate = float(len(wins) / total_trades) if total_trades > 0 else float('nan')
    avg_win = float(wins.mean()) if wins.size > 0 else 0.0
    avg_loss = float(losses.mean()) if losses.size > 0 else 0.0
    sum_wins = float(wins.sum()) if wins.size > 0 else 0.0
    sum_losses = float(losses.sum()) if losses.size > 0 else 0.0
    profit_factor = float(sum_wins / abs(sum_losses)) if sum_losses != 0 else (float('inf') if sum_wins>0 else float('nan'))

    # Streaks
    max_win_streak = 0; max_loss_streak = 0
    cur_win = 0; cur_loss = 0
    for p in profits:
        if p > 0:
            cur_win += 1; cur_loss = 0
        elif p < 0:
            cur_loss += 1; cur_win = 0
        else:
            cur_win = 0; cur_loss = 0
        if cur_win > max_win_streak: max_win_streak = cur_win
        if cur_loss > max_loss_streak: max_loss_streak = cur_loss

    metrics = {
        'total_trades': total_trades,
        'total_pnl': equity_delta,           # use equity delta as truth
        'sum_trades_pnl': float(profits.sum()) if profits.size>0 else 0.0,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'max_win_streak': int(max_win_streak),
        'max_loss_streak': int(max_loss_streak),
        'annualized_return_cagr': cagr,
        'annualized_return_arith': arith_ann_return,
        'annualized_vol': ann_vol,
        'total_return': total_return
    }

    # Diagnostics: check trade P&L sum vs equity delta
    trades_sum = metrics['sum_trades_pnl']
    equity_diff = metrics['total_pnl']
    tol = max(1.0, abs(equity_diff)*1e-6)  # tolerance
    consistent = abs(trades_sum - equity_diff) <= tol

    diagnostics = {
        'equity_start': start_val,
        'equity_end': end_val,
        'equity_delta': equity_delta,
        'sum_trades_pnl': trades_sum,
        'pnl_difference': trades_sum - equity_diff,
        'pnl_consistent': consistent,
        'num_periods': len(pct),
        'mean_period_return': float(pct.mean()),
        'std_period_return': float(pct.std())
    }

    return metrics, diagnostics

def main():
    t0 = time.time()
    print("Reading CSV:", FILEPATH)
    df = read_ohlc(FILEPATH)
    n = len(df)
    print(f"Rows loaded: {n}  Range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

    # train/test split 70/30
    split = int(n * 0.7)
    df_train = df.iloc[:split].reset_index(drop=True)
    df_test = df.iloc[split:].reset_index(drop=True)
    print(f"Train rows: {len(df_train)}  Test rows: {len(df_test)}")

    # regime detection on ATR(14)
    train_atr14 = atr(df_train['high'], df_train['low'], df_train['close'], n=14).values.reshape(-1,1)
    test_atr14  = atr(df_test['high'], df_test['low'], df_test['close'], n=14).values.reshape(-1,1)

    km_clusters = min(KM_REGIMES, max(1, len(train_atr14)//10))
    print("Using", km_clusters, "KMeans regimes for adaptation")
    kmeans = KMeans(n_clusters=km_clusters, random_state=42).fit(train_atr14)
    train_regime = kmeans.predict(train_atr14)
    test_regime = kmeans.predict(test_atr14)

    # per-regime search on TRAIN subgroups using position-sized backtest for scoring
    regime_params = {}
    for rl in range(km_clusters):
        mask = (train_regime == rl)
        if mask.sum() < 10:
            regime_params[rl] = {'atr_period': 10, 'multiplier': 3.0}
            continue
        df_subset = df_train.loc[mask].reset_index(drop=True)
        best_score = -1e12
        best_pair = {'atr_period': 10, 'multiplier': 3.0}
        for p in ADAPT_GRID_ATR_PERIODS:
            for m in ADAPT_GRID_MULTIPLIERS:
                st_train = supertrend_df(df_subset, period=p, multiplier=m)
                eq_train, trades_train = backtest_position_sizing(df_subset, st_train,
                                                                 initial_capital=INITIAL_CAPITAL,
                                                                 risk_pct=RISK_PCT,
                                                                 atr_stop_mult=ATR_STOP_MULT,
                                                                 max_pos_fraction=MAX_POS_FRACTION)
                rets = eq_train.pct_change().fillna(0)
                ann_ret = rets.mean() * FREQ_PER_YEAR
                ann_vol = rets.std() * math.sqrt(FREQ_PER_YEAR)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else float('nan')
                score = sharpe if not math.isnan(sharpe) else (eq_train.iloc[-1] / eq_train.iloc[0] - 1)
                if score > best_score:
                    best_score = score
                    best_pair = {'atr_period': p, 'multiplier': m}
        regime_params[rl] = best_pair
        print(f"Regime {rl}: chosen params {best_pair} (train rows {mask.sum()})")

    # build per-row SuperTrend on TEST using chosen params
    st_choices = {}
    for rl, params in regime_params.items():
        st_choices[rl] = supertrend_df(df_test, period=params['atr_period'], multiplier=params['multiplier'])
    st_test = pd.DataFrame(index=df_test.index, columns=['st', 'dir', 'atr']).astype(float)
    for idx in df_test.index:
        r = int(test_regime[idx]) if idx < len(test_regime) else 0
        st_test.loc[idx] = st_choices[r].loc[idx]

    # final backtest on TEST using realistic position sizing
    final_eq, final_trades = backtest_position_sizing(df_test, st_test,
                                                     initial_capital=INITIAL_CAPITAL,
                                                     risk_pct=RISK_PCT,
                                                     atr_stop_mult=ATR_STOP_MULT,
                                                     max_pos_fraction=MAX_POS_FRACTION)

    # align index with datetimes for metrics & save
    final_eq.index = df_test['datetime'].iloc[:len(final_eq)].values
    final_df_trades = pd.DataFrame(final_trades)

    metrics, diag = compute_metrics_and_diagnostics(final_eq, final_trades, freq_per_year=FREQ_PER_YEAR)

    print("\n--- Corrected Metrics & Diagnostics ---")
    print(f"total_trades: {metrics['total_trades']}")
    print(f"total_pnl (equity delta): {metrics['total_pnl']:.2f}")
    print(f"sum(trade.profit): {metrics['sum_trades_pnl']:.2f}")
    if not diag['pnl_consistent']:
        print("WARNING: sum(trade.profit) != equity_delta (they differ by {:.2f}).".format(diag['pnl_difference']))
        print("This indicates trades logging or equity accounting mismatch. Inspect trade list & backtest logic.")
    print(f"win_rate: {metrics['win_rate']:.2%}")
    print(f"avg_win: {metrics['avg_win']:.2f}")
    print(f"avg_loss: {metrics['avg_loss']:.2f}")
    print(f"profit_factor: {metrics['profit_factor']:.3f}")
    print(f"sharpe_ratio (arith returns): {metrics['sharpe_ratio']:.3f}")
    print(f"sortino_ratio: {metrics['sortino_ratio']:.3f}")
    print(f"max_drawdown: {metrics['max_drawdown']:.2%}")
    print(f"max_win_streak: {metrics['max_win_streak']}")
    print(f"max_loss_streak: {metrics['max_loss_streak']}")
    print(f"annualized_return (CAGR): {metrics['annualized_return_cagr']:.4%}")
    print(f"annualized_return (arithmetic): {metrics['annualized_return_arith']:.4%}")
    print(f"annualized_vol: {metrics['annualized_vol']:.4%}")
    print(f"total_return: {metrics['total_return']:.2%}")

    # Helpful diagnostics to inspect next:
    print("\n--- Quick Equity / Trades Diagnostics ---")
    print("Equity head (first 5):")
    print(final_eq.head(5))
    print("Equity tail (last 5):")
    print(final_eq.tail(5))

    # show worst and best trades
    if len(final_trades) > 0:
        tr_df = pd.DataFrame(final_trades)
        tr_df_sorted = tr_df.sort_values('profit')
        print("\nTop 10 worst trades:")
        print(tr_df_sorted.head(10).to_string(index=False))
        print("\nTop 10 best trades:")
        print(tr_df_sorted.tail(10).sort_values('profit', ascending=False).to_string(index=False))
    else:
        print("No trades recorded.")

    # Save corrected metrics and diagnostics
    pd.Series(metrics).to_csv(os.path.join(OUTDIR, "final_summary_corrected.csv"))
    pd.Series(diag).to_csv(os.path.join(OUTDIR, "final_diagnostics.csv"))
    if len(final_trades) > 0:
        tr_df.to_csv(os.path.join(OUTDIR, "final_trades_full.csv"), index=False)

    # plot equity
    try:
        plt.figure(figsize=(10,4))
        plt.plot(final_eq.index, final_eq.values)
        plt.title("Adaptive SuperTrend (position-sized) - Final TEST equity")
        plt.xlabel("Datetime"); plt.ylabel("Equity"); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "final_equity.png"))
        plt.close()
    except Exception as e:
        print("Plotting failed:", e)

    print("\nSaved outputs to:", OUTDIR)
    print("Elapsed (s):", round(time.time() - t0, 1))

if __name__ == "__main__":
    main()
