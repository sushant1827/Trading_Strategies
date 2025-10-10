# supertrend_optimization_with_risk.py
import pandas as pd
import numpy as np
from itertools import product

# Copy necessary functions from supertrend_heikin_ashi_strategy.py
def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    df = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
    df.columns = ['date', 'open', 'high', 'low', 'close']
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()
    df = df.drop_duplicates()
    return df

def calculate_heikin_ashi(df):
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    ha_open = pd.Series(index=df.index, dtype='float64')
    ha_open.iat[0] = (df['open'].iat[0] + df['close'].iat[0]) / 2.0
    for i in range(1, len(df)):
        ha_open.iat[i] = (ha_open.iat[i-1] + ha_close.iat[i-1]) / 2.0
    ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df['low'], ha_open, ha_close], axis=1).min(axis=1)
    ha_df = pd.DataFrame({
        'HA_Open': ha_open,
        'HA_High': ha_high,
        'HA_Low': ha_low,
        'HA_Close': ha_close
    }, index=df.index)
    return ha_df

def calculate_atr(ha_df, period=14, use_talib=True):
    high = ha_df['HA_High']
    low = ha_df['HA_Low']
    close = ha_df['HA_Close']
    if use_talib:
        try:
            import talib as ta
            atr_values = ta.ATR(high.values, low.values, close.values, timeperiod=period)
            atr = pd.Series(atr_values, index=ha_df.index)
        except:
            # Fallback to manual ATR
            prev_close = close.shift(1)
            tr1 = (high - low).abs()
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=1).mean()
    else:
        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def calculate_supertrend(ha_df, period=10, multiplier=3.0, use_talib_atr=False):
    ha_close = ha_df['HA_Close']
    ha_high = ha_df['HA_High']
    ha_low = ha_df['HA_Low']
    hl2 = (ha_high + ha_low) / 2.0
    atr = calculate_atr(ha_df, period=period, use_talib=use_talib_atr)
    if atr.isna().any():
        atr = atr.bfill()
        atr = atr.fillna(0.0)
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    final_upper = pd.Series(index=ha_df.index, dtype='float64')
    final_lower = pd.Series(index=ha_df.index, dtype='float64')
    trend = pd.Series(index=ha_df.index, dtype='int8')
    valid_idx = basic_upper.first_valid_index()
    if valid_idx is None:
        final_upper[:] = np.nan
        final_lower[:] = np.nan
        trend[:] = 1
        return trend, final_upper, final_lower, atr
    start_pos = ha_df.index.get_loc(valid_idx)
    final_upper.iat[start_pos] = basic_upper.iat[start_pos]
    final_lower.iat[start_pos] = basic_lower.iat[start_pos]
    trend.iat[start_pos] = 1
    for i in range(start_pos + 1, len(ha_df)):
        fu_prev = final_upper.iat[i-1]
        fl_prev = final_lower.iat[i-1]
        close_prev = ha_close.iat[i-1]
        if np.isnan(fu_prev):
            fu_prev = basic_upper.iat[i-1]
        if np.isnan(fl_prev):
            fl_prev = basic_lower.iat[i-1]
        if (basic_upper.iat[i] < fu_prev) or (close_prev > fu_prev):
            final_upper.iat[i] = basic_upper.iat[i]
        else:
            final_upper.iat[i] = fu_prev
        if (basic_lower.iat[i] > fl_prev) or (close_prev < fl_prev):
            final_lower.iat[i] = basic_lower.iat[i]
        else:
            final_lower.iat[i] = fl_prev
        if ha_close.iat[i] > final_upper.iat[i]:
            trend.iat[i] = 1
        elif ha_close.iat[i] < final_lower.iat[i]:
            trend.iat[i] = -1
        else:
            trend.iat[i] = trend.iat[i-1]
    return trend, final_upper, final_lower, atr

def generate_signals_from_trend(trend):
    buy = (trend == 1) & (trend.shift(1) == -1)
    sell = (trend == -1) & (trend.shift(1) == 1)
    buy = buy.fillna(False)
    sell = sell.fillna(False)
    return buy, sell

# Risk Management Parameters (from strategy.py)
TRAILING_STOP_MULTIPLIER = 1.0  # 1-2x ATR for trailing stop
RISK_PERCENTAGE = 0.01  # 1% of capital per trade
INITIAL_CAPITAL = 100000  # starting capital
MAX_LOSS_STREAK = 15  # allow more losses to continue trading

def backtest_simple(df, buy_signal, sell_signal):
    """Simple backtest without risk management"""
    position = 0
    entry_price = 0.0
    trades = []
    for i in range(len(df)):
        price = df['close'].iat[i]
        ts = df.index[i]
        if buy_signal.iat[i] and position <= 0:
            if position == -1:
                pnl = entry_price - price
                trades.append({'type': 'close_short', 'price': price, 'pnl': pnl, 'datetime': ts})
            position = 1
            entry_price = price
            trades.append({'type': 'buy', 'price': price, 'datetime': ts})
        elif sell_signal.iat[i] and position >= 0:
            if position == 1:
                pnl = price - entry_price
                trades.append({'type': 'close_long', 'price': price, 'pnl': pnl, 'datetime': ts})
            position = -1
            entry_price = price
            trades.append({'type': 'sell', 'price': price, 'datetime': ts})
    if position == 1:
        pnl = df['close'].iat[-1] - entry_price
        trades.append({'type': 'close_long', 'price': df['close'].iat[-1], 'pnl': pnl, 'datetime': df.index[-1]})
    elif position == -1:
        pnl = entry_price - df['close'].iat[-1]
        trades.append({'type': 'close_short', 'price': df['close'].iat[-1], 'pnl': pnl, 'datetime': df.index[-1]})
    trades_df = pd.DataFrame(trades)
    return trades_df

def backtest_with_risk(df, buy_signal, sell_signal, atr):
    """Backtest with risk management"""
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
    # Filter to completed trades (only closes have pnl)
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

# Main optimization function
def optimize_parameters(filepath, backtest_type='simple', atr_periods=range(20, 30), multipliers=np.arange(0.3, 1.1, 0.1)):
    df = load_and_prepare(filepath)
    ha_df = calculate_heikin_ashi(df)

    results = []
    for period, mult in product(atr_periods, multipliers):
        trend, upper, lower, atr = calculate_supertrend(ha_df, period=period, multiplier=mult)
        buy_signal, sell_signal = generate_signals_from_trend(trend)
        if backtest_type == 'simple':
            trades_df = backtest_simple(df, buy_signal, sell_signal)
            metrics = compute_basic_metrics(trades_df)
        elif backtest_type == 'with_risk':
            trades_df, final_capital = backtest_with_risk(df, buy_signal, sell_signal, atr)
            metrics = compute_basic_metrics(trades_df)
            metrics['final_capital'] = final_capital
        else:
            raise ValueError("backtest_type must be 'simple' or 'with_risk'")
        metrics['atr_period'] = period
        metrics['multiplier'] = mult
        metrics['backtest_type'] = backtest_type
        results.append(metrics)

    results_df = pd.DataFrame(results)
    # Sort by total_pnl descending
    results_df = results_df.sort_values('total_pnl', ascending=False)
    return results_df

# Walk-forward optimization
def walk_forward_optimization(filepath, backtest_type='simple', train_years=2, test_years=1):
    df = load_and_prepare(filepath)
    df['year'] = df.index.year
    years = sorted(df['year'].unique())

    wf_results = []
    for i in range(len(years) - train_years - test_years + 1):
        train_start = years[i]
        train_end = years[i + train_years - 1]
        test_start = years[i + train_years]
        test_end = years[i + train_years + test_years - 1]

        train_df = df[(df['year'] >= train_start) & (df['year'] <= train_end)]
        test_df = df[(df['year'] >= test_start) & (df['year'] <= test_end)]

        if train_df.empty or test_df.empty:
            continue

        ha_train = calculate_heikin_ashi(train_df)
        ha_test = calculate_heikin_ashi(test_df)

        # Optimize on train
        best_params = None
        best_pnl = -np.inf
        for period in range(20, 30):
            for mult in np.arange(0.3, 1.1, 0.1):
                trend, _, _, _ = calculate_supertrend(ha_train, period=period, multiplier=mult)
                buy, sell = generate_signals_from_trend(trend)
                if backtest_type == 'simple':
                    trades = backtest_simple(train_df, buy, sell)
                    pnl = trades['pnl'].sum() if not trades.empty else 0
                elif backtest_type == 'with_risk':
                    atr = calculate_atr(ha_train, period=period)
                    trades, _ = backtest_with_risk(train_df, buy, sell, atr)
                    pnl = trades['pnl'].sum() if not trades.empty else 0
                else:
                    raise ValueError("backtest_type must be 'simple' or 'with_risk'")
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_params = (period, mult)

        # Test on test
        if best_params:
            period, mult = best_params
            trend, _, _, atr = calculate_supertrend(ha_test, period=period, multiplier=mult)
            buy, sell = generate_signals_from_trend(trend)
            if backtest_type == 'simple':
                trades = backtest_simple(test_df, buy, sell)
                metrics = compute_basic_metrics(trades)
            elif backtest_type == 'with_risk':
                trades, final_capital = backtest_with_risk(test_df, buy, sell, atr)
                metrics = compute_basic_metrics(trades)
                metrics['final_capital'] = final_capital
            else:
                raise ValueError("backtest_type must be 'simple' or 'with_risk'")
            metrics['train_period'] = f"{train_start}-{train_end}"
            metrics['test_period'] = f"{test_start}-{test_end}"
            metrics['atr_period'] = period
            metrics['multiplier'] = mult
            metrics['backtest_type'] = backtest_type
            wf_results.append(metrics)

    wf_df = pd.DataFrame(wf_results)
    return wf_df

# Run optimizations
if __name__ == "__main__":
    FILE_PATH = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'

    print("Running Grid Search Optimization - Simple Backtest...")
    grid_results_simple = optimize_parameters(FILE_PATH, backtest_type='simple')
    print("Top 10 Parameter Combinations (Simple):")
    print(grid_results_simple.head(10))

    print("\nRunning Grid Search Optimization - With Risk Management...")
    grid_results_risk = optimize_parameters(FILE_PATH, backtest_type='with_risk')
    print("Top 10 Parameter Combinations (With Risk):")
    print(grid_results_risk.head(10))

    print("\nRunning Walk-Forward Optimization - Simple Backtest...")
    wf_results_simple = walk_forward_optimization(FILE_PATH, backtest_type='simple')
    print("Walk-Forward Results (Simple):")
    print(wf_results_simple)

    print("\nRunning Walk-Forward Optimization - With Risk Management...")
    wf_results_risk = walk_forward_optimization(FILE_PATH, backtest_type='with_risk')
    print("Walk-Forward Results (With Risk):")
    print(wf_results_risk)

    # Save results
    grid_results_simple.to_csv('grid_search_results_simple.csv', index=False)
    grid_results_risk.to_csv('grid_search_results_risk.csv', index=False)
    wf_results_simple.to_csv('walk_forward_results_simple.csv', index=False)
    wf_results_risk.to_csv('walk_forward_results_risk.csv', index=False)
    print("Results saved to CSV files.")