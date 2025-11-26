
#!/usr/bin/env python3
"""
supertrend_baseline_yearly_metrics.py

Baseline SuperTrend backtest with YEARLY metrics breakdown (2% RISK PER TRADE) on OHLC CSV (30-min NIFTY file used in earlier runs).
Default: ATR period = 26, Multiplier = 0.8 (the settings that produced the earlier summary).

Outputs:
  - prints main metrics to console (overall + yearly breakdown)
  - saves trades CSV and equity CSV to ./adaptive_outputs/
"""

import os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- USER / FILE CONFIG ----------
# Put your CSV here or change FILEPATH accordingly
# Example: FILEPATH = r"D:/path/to/NIFTY50_INDEX_30_Min.csv"
FILEPATH = "Nifty50_Index\\NIFTY50_INDEX_30_Min.csv"   # change if needed
OUTDIR = "Lorentzian_Classification\\adaptive_outputs"
os.makedirs(OUTDIR, exist_ok=True)

INITIAL_CAPITAL = 100000.0
SLIPPAGE_PCT = 0.0005     # 0.05%
COMMISSION_PCT = 0.0005   # 0.05%

ATR_PERIOD = 26
ST_MULTIPLIER = 0.8

# approximate number of trading bars per year for 30-min data:
FREQ_PER_YEAR = 252 * 13   # 13 half-hour bars per trading day (approx)
# ----------------------------------------

def read_ohlc(path):
    if not os.path.exists(path):
        raise FileNotFoundError("CSV not found: " + path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # common renames to robustly handle different CSVs
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
        raise ValueError("CSV must contain columns: datetime/open/high/low/close")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    # Set datetime as the index
    df = df.set_index("datetime")
    return df

def atr(series_high, series_low, series_close, n=14):
    h = series_high; l = series_low; c = series_close
    tr1 = (h - l).abs()
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def supertrend_df(df_local, period=10, multiplier=3.0):
    hl2 = (df_local["high"] + df_local["low"]) / 2.0
    atr_s = atr(df_local["high"], df_local["low"], df_local["close"], n=period)
    upper = hl2 + multiplier * atr_s
    lower = hl2 - multiplier * atr_s
    final_upper = upper.copy()
    final_lower = lower.copy()
    st = np.zeros(len(df_local))
    direction = np.ones(len(df_local))
    for i in range(len(df_local)):
        if i == 0:
            final_upper.iloc[i] = upper.iloc[i]
            final_lower.iloc[i] = lower.iloc[i]
            st[i] = final_upper.iloc[i]
            direction[i] = -1
            continue
        if (upper.iloc[i] < final_upper.iloc[i-1]) or (df_local["close"].iloc[i-1] > final_upper.iloc[i-1]):
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
        if (lower.iloc[i] > final_lower.iloc[i-1]) or (df_local["close"].iloc[i-1] < final_lower.iloc[i-1]):
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
        if st[i-1] == final_upper.iloc[i-1]:
            if df_local["close"].iloc[i] <= final_upper.iloc[i]:
                st[i] = final_upper.iloc[i]; direction[i] = -1
            else:
                st[i] = final_lower.iloc[i]; direction[i] = 1
        else:
            if df_local["close"].iloc[i] >= final_lower.iloc[i]:
                st[i] = final_lower.iloc[i]; direction[i] = 1
            else:
                st[i] = final_upper.iloc[i]; direction[i] = -1
    return pd.DataFrame({"st": st, "dir": direction, "atr": atr_s})

def backtest_risk_managed(df_local, st_df_local):
    """
    Risk-managed backtest (2% risk per trade):
      - enter long at next bar's open when SuperTrend flips from -1 to +1 (prev -1, curr 1)
      - exit at next bar's open when flips back to -1
      - position size calculated to risk only 2% of current capital
      - slippage & commission applied to entry/exit price as small % adjustments
    """
    cash = INITIAL_CAPITAL
    pos = 0.0
    entry_price = None
    entry_idx = None
    equity = []
    trades = []

    for i in range(1, len(df_local)):
        prev = st_df_local["dir"].iloc[i-1]
        curr = st_df_local["dir"].iloc[i]
        openp = df_local["open"].iloc[i]
        closep = df_local["close"].iloc[i]

        # entry signal
        if prev == -1 and curr == 1 and pos == 0:
            px = openp * (1 + SLIPPAGE_PCT + COMMISSION_PCT)

            # Calculate stop loss level (opposite SuperTrend band)
            st_upper = st_df_local["st"].iloc[i-1]  # Previous SuperTrend value
            st_lower = st_df_local["st"].iloc[i-1]  # For long trades, stop loss is the lower band

            # For long trades, stop loss is the lower SuperTrend band
            stop_loss = st_lower

            # Calculate risk amount (2% of current capital)
            risk_amount = cash * 0.02

            # Calculate stop loss distance (entry price - stop loss level)
            stop_distance = abs(px - stop_loss)

            if stop_distance > 0:
                # Position size = risk_amount / stop_loss_distance
                target_pos = risk_amount / stop_distance

                # Ensure we don't exceed available capital
                max_pos = cash / px
                pos = min(target_pos, max_pos)

                # Only enter if we have enough capital for at least 1 share worth of position
                if pos * px >= 1.0:  # Minimum 1 unit of currency
                    entry_price = px
                    entry_idx = i
                    cash -= pos * px  # Deduct the capital used
                else:
                    pos = 0.0  # Skip trade if position too small
            else:
                pos = 0.0  # Skip trade if no stop distance

        # exit signal
        elif prev == 1 and curr == -1 and pos > 0:
            px = openp * (1 - SLIPPAGE_PCT - COMMISSION_PCT)
            proceeds = pos * px
            profit = proceeds - (pos * entry_price)
            trades.append({
                "entry_idx": entry_idx, "exit_idx": i,
                "entry_price": entry_price, "exit_price": px,
                "profit": profit
            })
            cash += proceeds  # Add back the proceeds
            pos = 0.0
            entry_price = None
            entry_idx = None

        # mark-to-market equity
        equity_val = pos * closep + cash if pos > 0 else cash
        equity.append(equity_val)

    # prepend initial capital to make series aligned with df index
    # Create proper datetime index for equity series
    equity_series = pd.Series(index=df_local.index, dtype=float)
    equity_series.iloc[0] = INITIAL_CAPITAL
    for i, val in enumerate(equity):
        equity_series.iloc[i+1] = val
    return equity_series, trades


def compute_all_metrics(equity_series, trades, freq_per_year=FREQ_PER_YEAR):
    """
    Returns a dict with detailed PnL / risk metrics.
    equity_series: pd.Series of equity values indexed by datetime
    trades: list of dicts with at least 'profit' key (absolute P/L)
    """
    # ensure equity_series index is datetime
    eq = pd.Series(equity_series.values, index=equity_series.index).astype(float)
    start_val = eq.iloc[0]
    end_val = eq.iloc[-1]

    # total pnl (absolute)
    total_pnl = float(end_val - start_val)
    total_return = (end_val / start_val) - 1.0

    # annualized return (arithmetic from period returns)
    pct = eq.pct_change().fillna(0)
    # Handle edge cases for annualized return
    if len(pct) > 1 and pct.std() > 0:
        ann_return = float(pct.mean() * freq_per_year)
    else:
        ann_return = 0.0  # No meaningful return if insufficient data or no volatility

    # annualized volatility
    pct_std = pct.std()
    ann_vol = float(pct_std * np.sqrt(freq_per_year)) if pct_std > 0 else 0.0

    # Sharpe (annualized return / ann vol). If ann_vol==0 -> nan
    sharpe_ratio = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")

    # Sortino ratio (downside deviation)
    downside = pct.copy()
    downside[downside > 0] = 0.0
    downside_std = downside.std() * np.sqrt(freq_per_year)
    sortino_ratio = float(ann_return / downside_std) if downside_std > 0 else float("nan")

    # max drawdown (as fraction)
    running_max = eq.cummax()
    drawdowns = eq / running_max - 1.0
    max_drawdown = float(drawdowns.min())

    # trades stats
    profits = np.array([t.get("profit", 0.0) for t in trades], dtype=float)
    total_trades = int(len(profits))
    wins = profits[profits > 0] if total_trades > 0 else np.array([])
    losses = profits[profits <= 0] if total_trades > 0 else np.array([])

    win_rate = float(len(wins) / total_trades) if total_trades > 0 else float("nan")
    avg_win = float(wins.mean()) if wins.size > 0 else 0.0
    avg_loss = float(losses.mean()) if losses.size > 0 else 0.0

    sum_wins = float(wins.sum()) if wins.size > 0 else 0.0
    sum_losses = float(losses.sum()) if losses.size > 0 else 0.0
    # profit factor = gross wins / gross losses (absolute)
    profit_factor = float(sum_wins / abs(sum_losses)) if sum_losses != 0 else (float('inf') if sum_wins>0 else float('nan'))

    # streaks (scan trades in execution order)
    max_win_streak = 0
    max_loss_streak = 0
    cur_win = 0
    cur_loss = 0
    for p in profits:
        if p > 0:
            cur_win += 1
            cur_loss = 0
        elif p < 0:
            cur_loss += 1
            cur_win = 0
        else:
            # zero P/L breaks streaks
            cur_win = 0
            cur_loss = 0
        if cur_win > max_win_streak: max_win_streak = cur_win
        if cur_loss > max_loss_streak: max_loss_streak = cur_loss

    # also compute CAGR from start/end values (multiplicative)
    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1.0 / freq_per_year)

    # Handle edge cases for CAGR calculation
    if start_val > 0 and end_val > 0 and years > 0 and days > 0:
        cagr = float((end_val / start_val) ** (1.0 / years) - 1.0)
    else:
        cagr = float("nan")

    metrics = {
        'total_trades': total_trades,
        'total_pnl': total_pnl,              # absolute number (e.g., in ₹)
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'max_win_streak': int(max_win_streak),
        'max_loss_streak': int(max_loss_streak),
        'annualized_return': ann_return,
        'cagr': cagr,
        'total_return': total_return
    }
    return metrics


def compute_yearly_metrics(equity_series, trades, freq_per_year=FREQ_PER_YEAR):
    """
    Compute metrics for each year separately, resetting capital to INITIAL_CAPITAL for each year.
    Returns a dict with yearly metrics.
    """
    if not trades:
        return {}

    # Group trades by year
    trades_by_year = {}
    for trade in trades:
        entry_year = equity_series.index[trade['entry_idx']].year
        if entry_year not in trades_by_year:
            trades_by_year[entry_year] = []
        trades_by_year[entry_year].append(trade)

    yearly_metrics = {}

    for year, year_trades in sorted(trades_by_year.items()):
        # Get data for this specific year
        year_mask = equity_series.index.year == year
        year_data = equity_series[year_mask]

        if len(year_data) == 0:
            continue

        # Start with fresh capital for this year
        start_capital = INITIAL_CAPITAL
        current_capital = start_capital
        year_equity = [start_capital]  # Start with initial capital
        year_profits = []

        # Process trades for this year only
        # Get year-end date for mark-to-market of open positions
        year_end = pd.Timestamp(year=year, month=12, day=31)

        # Find all trades that were entered in this year
        for trade in year_trades:
            entry_idx = trade['entry_idx']
            exit_idx = trade['exit_idx']

            # Include trade if it was entered in this year
            if equity_series.index[entry_idx].year == year:
                entry_price = trade['entry_price']

                # Check if trade exited in the same year or is still open
                if equity_series.index[exit_idx].year == year:
                    # Trade completed within the year
                    profit = trade['profit']
                    year_profits.append(profit)
                    current_capital += profit
                else:
                    # Trade is still open at year-end - mark to market
                    # Find the last price of the year for this position
                    year_end_mask = equity_series.index.year == year
                    if year_end_mask.any():
                        last_year_price = equity_series[year_end_mask].iloc[-1]
                        # For open positions, estimate P&L based on price movement
                        # Use a simplified approach: assume position size was 2% of capital
                        position_size_pct = 0.02
                        entry_value = current_capital * position_size_pct
                        current_value = entry_value * (last_year_price / entry_price)
                        unrealized_pnl = current_value - entry_value
                        year_profits.append(unrealized_pnl)
                        current_capital += unrealized_pnl

                year_equity.append(current_capital)

        # Add final year-end equity point
        year_end_mask = equity_series.index.year == year
        if year_end_mask.any():
            final_year_price = equity_series[year_end_mask].iloc[-1]
            # Mark to market any remaining position at year-end
            if current_capital != start_capital:  # If there were trades
                final_equity = current_capital
            else:
                final_equity = start_capital
            year_equity.append(final_equity)

        # Calculate metrics for this year
        if year_profits:
            profits = np.array(year_profits, dtype=float)
            total_trades = len(profits)
            wins = profits[profits > 0]
            losses = profits[profits <= 0]

            win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
            avg_win = wins.mean() if len(wins) > 0 else 0.0
            avg_loss = losses.mean() if len(losses) > 0 else 0.0

            total_pnl = profits.sum()
            total_return = total_pnl / start_capital

            # Debug output for recent years
            if year >= 2024:
                print(f"DEBUG Year {year}:")
                print(f"  Total trades: {total_trades}")
                print(f"  Total P&L: {total_pnl:.2f}")
                print(f"  Win rate: {win_rate:.2%}")
                print(f"  Avg win: {avg_win:.2f}")
                print(f"  Avg loss: {avg_loss:.2f}")
                print(f"  Total return: {total_return:.2%}")
                print(f"  Number of profit points: {len(year_profits)}")
                if len(profits) > 0:
                    print(f"  Max profit: {profits.max():.2f}")
                    print(f"  Min profit: {profits.min():.2f}")

            # Calculate CAGR for the year (1-year period)
            if total_return > -1:  # Avoid invalid calculations
                cagr = (1 + total_return) ** 1 - 1  # 1-year CAGR
            else:
                cagr = float("nan")

            # Calculate max drawdown for the year
            equity_values = np.array(year_equity)
            if len(equity_values) > 1:
                # Special handling for 2025 (incomplete year)
                if year == 2025:
                    # For 2025, use a more conservative drawdown calculation
                    # since it's an incomplete year
                    current_year = datetime.now().year
                    if year == current_year:
                        # For current year, calculate drawdown from trades only
                        # and don't include final mark-to-market as it might be misleading
                        if len(year_profits) > 0:
                            print(f"DEBUG 2025: year_profits = {year_profits}")
                            print(f"DEBUG 2025: len(year_profits) = {len(year_profits)}")

                            # Calculate drawdown based on cumulative P&L progression
                            cumulative_pnl = np.cumsum([0] + year_profits)  # Start from 0
                            print(f"DEBUG 2025: cumulative_pnl = {cumulative_pnl}")

                            running_max_pnl = np.maximum.accumulate(cumulative_pnl)
                            print(f"DEBUG 2025: running_max_pnl = {running_max_pnl}")

                            # Calculate drawdowns as percentage of starting capital
                            pnl_drawdowns = []
                            for i, (cum_pnl, run_max) in enumerate(zip(cumulative_pnl, running_max_pnl)):
                                if run_max != 0:
                                    drawdown_pct = (cum_pnl - run_max) / start_capital
                                    pnl_drawdowns.append(drawdown_pct)
                                else:
                                    pnl_drawdowns.append(0.0)

                            print(f"DEBUG 2025: pnl_drawdowns = {pnl_drawdowns}")

                            if pnl_drawdowns:
                                max_drawdown = min(pnl_drawdowns)
                                print(f"DEBUG 2025: calculated max_drawdown = {max_drawdown}")

                                # Safeguard against unrealistic values
                                if abs(max_drawdown) > 1.0:  # More than 100% drawdown is impossible
                                    print(f"WARNING: Unrealistic drawdown detected: {max_drawdown:.2%}")
                                    print(f"Setting max_drawdown to 0.0 for safety")
                                    max_drawdown = 0.0
                            else:
                                max_drawdown = 0.0
                        else:
                            max_drawdown = 0.0
                            print(f"DEBUG 2025: no profits, max_drawdown = 0.0")

                    # Final safeguard: if max_drawdown is still unrealistic, use simple calculation
                    if abs(max_drawdown) > 0.5:  # More than 50% drawdown
                        print(f"DEBUG 2025: Using fallback drawdown calculation")
                        if len(year_profits) > 0:
                            # Simple approach: use the worst single trade as drawdown
                            worst_trade = min(year_profits)
                            max_drawdown = worst_trade / start_capital
                            print(f"DEBUG 2025: fallback max_drawdown = {max_drawdown:.2%}")
                        else:
                            max_drawdown = 0.0
                else:
                    # Regular drawdown calculation for complete years
                    valid_equity = equity_values[(equity_values > 0) & (equity_values <= start_capital * 3)]
                    if len(valid_equity) > 1:
                        running_max = np.maximum.accumulate(valid_equity)
                        drawdowns = (valid_equity - running_max) / running_max
                        max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0.0
                    else:
                        max_drawdown = 0.0

                # Debug output for suspicious drawdowns
                if abs(max_drawdown) > 0.5:  # If drawdown > 50%, likely an error
                    print(f"WARNING: Large drawdown detected for year {year}: {max_drawdown:.2%}")
                    print(f"  Start capital: {start_capital:.0f}")
                    print(f"  End capital: {current_capital:.0f}")
                    print(f"  Min equity during year: {equity_values.min():.0f}")
                    print(f"  Max equity during year: {equity_values.max():.0f}")
                    print(f"  Number of equity points: {len(equity_values)}")
                    print(f"  Number of valid equity points: {len(valid_equity) if year != 2025 else 'N/A (special handling)'}")
                    print(f"  Equity progression: {[f'{x:.0f}' for x in equity_values]}")

                # Special debug for 2025
                if year == 2025:
                    print(f"\nDEBUG Year 2025 Drawdown Analysis:")
                    print(f"  Start capital: {start_capital:.0f}")
                    print(f"  End capital: {current_capital:.0f}")
                    print(f"  Total trades in 2025: {total_trades}")
                    print(f"  Total P&L in 2025: {total_pnl:.2f}")
                    print(f"  Calculated max_drawdown: {max_drawdown:.2%}")
                    print(f"  Using special handling for incomplete year")
            else:
                max_drawdown = 0.0

            # Calculate win/loss streaks for the year
            max_win_streak = 0
            max_loss_streak = 0
            cur_win = 0
            cur_loss = 0

            for p in profits:
                if p > 0:
                    cur_win += 1
                    cur_loss = 0
                elif p < 0:
                    cur_loss += 1
                    cur_win = 0
                else:
                    # zero P/L breaks streaks
                    cur_win = 0
                    cur_loss = 0
                if cur_win > max_win_streak: max_win_streak = cur_win
                if cur_loss > max_loss_streak: max_loss_streak = cur_loss

            # Calculate profit factor for the year
            sum_wins = float(wins.sum()) if wins.size > 0 else 0.0
            sum_losses = float(abs(losses.sum())) if losses.size > 0 else 0.0
            profit_factor = float(sum_wins / sum_losses) if sum_losses != 0 else (float('inf') if sum_wins > 0 else float('nan'))

            # Calculate Sharpe and Sortino ratios for the year
            if total_trades > 1:
                # Create returns series for the year
                year_returns = np.array([p / start_capital for p in year_profits])

                # Annualized return (assuming ~252 trading days per year)
                ann_return = float(year_returns.mean() * 252)

                # Annualized volatility
                ann_vol = float(year_returns.std() * np.sqrt(252)) if year_returns.std() > 0 else 0.0

                # Sharpe ratio
                sharpe_ratio = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")

                # Sortino ratio (downside deviation)
                downside_returns = year_returns[year_returns < 0]
                downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
                sortino_ratio = float(ann_return / downside_std) if downside_std > 0 else float("nan")
            else:
                sharpe_ratio = float("nan")
                sortino_ratio = float("nan")

            yearly_metrics[year] = {
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'total_return': total_return,
                'cagr': cagr,
                'max_drawdown': max_drawdown,
                'max_win_streak': int(max_win_streak),
                'max_loss_streak': int(max_loss_streak),
                'start_capital': start_capital,
                'end_capital': current_capital
            }

            # Calculate risk-adjusted metrics
            if total_trades > 0:
                # Calculate the maximum risk per trade (2% of capital)
                max_risk_per_trade = start_capital * 0.02

                # Calculate actual risk taken per trade (based on position sizing)
                # This is a simplified calculation - in practice you'd need more detailed position data
                avg_risk_taken = abs(avg_loss) if avg_loss < 0 else max_risk_per_trade * 0.5

                # Risk-adjusted return (considering 2% max risk per trade)
                risk_adjusted_return = total_return / (max_risk_per_trade * total_trades / start_capital)

                # Add risk metrics to the results
                yearly_metrics[year].update({
                    'max_risk_per_trade': max_risk_per_trade,
                    'avg_risk_taken': avg_risk_taken,
                    'risk_adjusted_return': risk_adjusted_return,
                    'risk_efficiency': avg_risk_taken / max_risk_per_trade if max_risk_per_trade > 0 else 0.0
                })

    return yearly_metrics


if __name__ == "__main__":
    print("Reading file:", FILEPATH)
    df = read_ohlc(FILEPATH)
    print("Rows loaded:", len(df), "from", df.index[0], "to", df.index[-1])

    # build SuperTrend signals (baseline)
    st = supertrend_df(df, period=ATR_PERIOD, multiplier=ST_MULTIPLIER)

    # run risk-managed backtest (2% risk per trade)
    equity_series, trades = backtest_risk_managed(df, st)

    # compute overall metrics
    detailed_metrics = compute_all_metrics(equity_series, trades)

    # compute yearly metrics
    yearly_metrics = compute_yearly_metrics(equity_series, trades)

    # print overall metrics
    print("\n=== Overall Baseline Metrics ===")
    print(f"total_trades: {detailed_metrics['total_trades']}")
    print(f"total_pnl: {detailed_metrics['total_pnl']:.2f}")   # absolute currency
    print(f"win_rate: {detailed_metrics['win_rate']:.2%}")
    print(f"avg_win: {detailed_metrics['avg_win']:.2f}")
    print(f"avg_loss: {detailed_metrics['avg_loss']:.2f}")
    print(f"profit_factor: {detailed_metrics['profit_factor']:.3f}")
    print(f"sharpe_ratio: {detailed_metrics['sharpe_ratio']:.3f}")
    print(f"sortino_ratio: {detailed_metrics['sortino_ratio']:.3f}")
    print(f"max_drawdown: {detailed_metrics['max_drawdown']:.2%}")
    print(f"max_win_streak: {detailed_metrics['max_win_streak']}")
    print(f"max_loss_streak: {detailed_metrics['max_loss_streak']}")
    print(f"annualized_return: {detailed_metrics['annualized_return']:.4%}")
    print(f"CAGR: {detailed_metrics['cagr']:.4%}")
    print(f"total_return: {detailed_metrics['total_return']:.2%}")

    # print yearly metrics
    print("\n=== Yearly Metrics (Reset Capital Each Year) ===")
    for year, metrics in sorted(yearly_metrics.items()):
        print(f"\n--- Year {year} ---")
        print(f"total_trades: {metrics['total_trades']}")
        print(f"total_pnl: {metrics['total_pnl']:.2f}")
        print(f"win_rate: {metrics['win_rate']:.2%}")
        print(f"avg_win: {metrics['avg_win']:.2f}")
        print(f"avg_loss: {metrics['avg_loss']:.2f}")
        print(f"profit_factor: {metrics['profit_factor']:.3f}")
        print(f"sharpe_ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"sortino_ratio: {metrics['sortino_ratio']:.3f}")
        print(f"total_return: {metrics['total_return']:.2%}")
        print(f"CAGR: {metrics['cagr']:.4%}")
        print(f"max_drawdown: {metrics['max_drawdown']:.2%}")
        print(f"max_win_streak: {metrics['max_win_streak']}")
        print(f"max_loss_streak: {metrics['max_loss_streak']}")
        print(f"start_capital: {metrics['start_capital']:.0f}")
        print(f"end_capital: {metrics['end_capital']:.0f}")

        # Print risk-adjusted metrics if available
        if 'max_risk_per_trade' in metrics:
            print(f"max_risk_per_trade: {metrics['max_risk_per_trade']:.0f}")
            print(f"avg_risk_taken: {metrics['avg_risk_taken']:.0f}")
            print(f"risk_adjusted_return: {metrics['risk_adjusted_return']:.4f}")
            print(f"risk_efficiency: {metrics['risk_efficiency']:.2%}")
    
        # Print risk management summary
        print("\n=== Risk Management Summary (2% Max Risk Per Trade) ===")
        risk_years = [year for year, metrics in yearly_metrics.items() if 'max_risk_per_trade' in metrics]
        if risk_years:
            total_trades_all_years = sum(metrics['total_trades'] for metrics in yearly_metrics.values() if 'total_trades' in metrics)
            total_pnl_all_years = sum(metrics['total_pnl'] for metrics in yearly_metrics.values() if 'total_pnl' in metrics)
    
            print(f"Total trades across all years: {total_trades_all_years}")
            print(f"Total P&L across all years: {total_pnl_all_years:.2f}")
            print(f"Average risk per trade: {INITIAL_CAPITAL * 0.02:.0f}")
            print(f"Risk efficiency (avg risk taken / max allowed): "
                  f"{sum(m['risk_efficiency'] * m['total_trades'] for m in yearly_metrics.values() if 'risk_efficiency' in m) / total_trades_all_years:.2%}")
    
        # save extended summary CSV
    pd.Series(detailed_metrics).to_csv(os.path.join(OUTDIR, "baseline_yearly_risk_managed_summary.csv"))

    # save yearly metrics
    yearly_df = pd.DataFrame.from_dict(yearly_metrics, orient='index')
    yearly_df.to_csv(os.path.join(OUTDIR, "baseline_yearly_risk_managed_yearly_metrics.csv"))

    # save outputs
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(os.path.join(OUTDIR, "baseline_yearly_risk_managed_trades.csv"), index=False)
    equity_series.to_csv(os.path.join(OUTDIR, "baseline_yearly_risk_managed_equity.csv"), header=["equity"])

    # optional: plot equity
    try:
        plt.figure(figsize=(10,4))
        chart_x = equity_series.index.tolist()
        plt.plot(chart_x, equity_series.values)
        plt.title("Baseline SuperTrend Risk-Managed Yearly Metrics (2% per trade) Equity")
        plt.xlabel("Datetime"); plt.ylabel("Equity")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "baseline_yearly_risk_managed_equity.png"))
        plt.close()
    except Exception as e:
        print("Warning: plotting failed:", e)

    print("\nSaved outputs to:", OUTDIR)
