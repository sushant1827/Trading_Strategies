# auto_supertrend_prototype.py
"""
Auto-Supertrend Prototype (full script)

Place your CSV at: Nifty50_Index/NIFTY50_INDEX_30_Min.csv
Required columns: ['datetime','open','high','low','close','volume']
Run: python auto_supertrend_prototype.py
"""

import os
import pickle
import itertools
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

# -----------------------------
# Indicator helpers
# -----------------------------
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """
    Returns a Series with values 1 (up) or -1 (down).
    Implementation follows typical Supertrend logic using ATR.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    atr_val = atr(df, period)
    hl2 = (high + low) / 2
    basic_upperband = hl2 + multiplier * atr_val
    basic_lowerband = hl2 - multiplier * atr_val

    final_upperband = basic_upperband.copy()
    final_lowerband = basic_lowerband.copy()

    # iterative final bands
    for i in range(1, len(df)):
        if (basic_upperband.iloc[i] < final_upperband.iloc[i-1]) or (close.iloc[i-1] > final_upperband.iloc[i-1]):
            final_upperband.iloc[i] = basic_upperband.iloc[i]
        else:
            final_upperband.iloc[i] = final_upperband.iloc[i-1]

        if (basic_lowerband.iloc[i] > final_lowerband.iloc[i-1]) or (close.iloc[i-1] < final_lowerband.iloc[i-1]):
            final_lowerband.iloc[i] = basic_lowerband.iloc[i]
        else:
            final_lowerband.iloc[i] = final_lowerband.iloc[i-1]

    trend = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if close.iloc[i] > final_upperband.iloc[i-1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < final_lowerband.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    return trend

def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=1).mean()
    ma_down = down.rolling(period, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-8)
    return 100 - (100 / (1 + rs))

# -----------------------------
# Signals, PnL, features
# -----------------------------
def generate_signals_and_pnl(df: pd.DataFrame, periods: list, multipliers: list, exit_on_opposite: bool = True):
    """
    For each combo, compute supertrend and simulate trades.
    Returns:
      - combos (list of tuples)
      - combo_signals: dict[(p,m)] -> pd.Series of signals (1, -1, 0)
      - per_signal_pnls: DataFrame indexed like df, columns = combos MultiIndex with pnl for trade starting at that row (NaN otherwise)
      - per_signal_best: Series (index df) where each entry is the combo tuple that produced best pnl at that signal (or None)
    """
    combos = list(itertools.product(periods, multipliers))
    combo_signals = {}
    combo_trade_results = {c: [] for c in combos}

    # compute signals for each combo
    for (p, m) in combos:
        trend = supertrend(df, period=p, multiplier=m)
        sig = trend.diff().fillna(0)
        signal = sig.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        combo_signals[(p, m)] = signal

    # DataFrame to store pnl per entry index for each combo
    per_signal_pnls = pd.DataFrame(index=df.index, columns=pd.MultiIndex.from_tuples(combos, names=['period', 'multiplier']))

    # simulate trades for each combo
    index = df.index
    for combo in combos:
        signal_series = combo_signals[combo]
        position = 0
        entry_price = None
        entry_idx = None
        for i in range(len(df)):
            s = signal_series.iloc[i]
            # open position when signal appears and no position currently
            if s != 0 and position == 0:
                if i + 1 < len(df):
                    entry_price = df['open'].iloc[i+1]
                    entry_idx = i+1
                    position = s
                else:
                    position = 0
            # close on opposite signal
            elif s != 0 and position != 0 and s == -position and exit_on_opposite:
                if i + 1 < len(df) and entry_price is not None and entry_idx is not None:
                    exit_price = df['open'].iloc[i+1]
                    pnl = (exit_price - entry_price) * position
                    combo_trade_results[combo].append({'entry_idx': entry_idx, 'exit_idx': i+1, 'pnl': pnl})
                    per_signal_pnls.loc[df.index[entry_idx], combo] = pnl
                position = 0
                entry_price = None
                entry_idx = None
        # close any open at last close
        if position != 0 and entry_price is not None and entry_idx is not None:
            exit_price = df['close'].iloc[-1]
            pnl = (exit_price - entry_price) * position
            combo_trade_results[combo].append({'entry_idx': entry_idx, 'exit_idx': len(df)-1, 'pnl': pnl})
            per_signal_pnls.loc[df.index[entry_idx], combo] = pnl

    # choose best combo per entry row (max pnl)
    per_signal_best = pd.Series(index=df.index, dtype=object)
    for i in range(len(df)):
        row = per_signal_pnls.iloc[i]
        if row.isnull().all():
            per_signal_best.iloc[i] = None
            continue
        # pick maximum pnl combo
        # convert to float (NaN ignored) and idxmax gives the combo tuple
        try:
            best_combo = row.astype(float).idxmax()
            per_signal_best.iloc[i] = best_combo
        except Exception:
            per_signal_best.iloc[i] = None

    return combos, combo_signals, per_signal_pnls, per_signal_best

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f['close'] = df['close']
    f['return_1'] = df['close'].pct_change().fillna(0)
    f['return_5'] = df['close'].pct_change(5).fillna(0)
    f['vol_5'] = f['return_1'].rolling(5).std().fillna(0)
    f['vol_20'] = f['return_1'].rolling(20).std().fillna(0)
    f['atr_14'] = atr(df, 14).fillna(0)
    f['rsi_14'] = rsi(df, 14).fillna(50)
    f['ma_20'] = df['close'].rolling(20).mean().bfill()
    f['ma_50'] = df['close'].rolling(50).mean().bfill()
    f['ma_diff'] = f['ma_20'] - f['ma_50']
    # volume column could be missing/zero — keep as-is
    f['volume'] = df['volume'] if 'volume' in df.columns else 0

    # simple regime classification based on volatility and returns
    # High volatility + positive returns = regime 2 (bullish volatile)
    # High volatility + negative returns = regime 1 (bearish volatile)
    # Low volatility = regime 0 (calm)
    vol_threshold = f['vol_20'].quantile(0.7)
    return_threshold = f['return_1'].quantile(0.5)

    f['regime'] = 0  # default calm regime
    f.loc[(f['vol_20'] > vol_threshold) & (f['return_1'] > return_threshold), 'regime'] = 2  # bullish volatile
    f.loc[(f['vol_20'] > vol_threshold) & (f['return_1'] <= return_threshold), 'regime'] = 1  # bearish volatile

    f = f.ffill().bfill().fillna(0)
    return f


def create_feature_bins():
    """
    Create bins for discretizing features into mapping keys.
    Returns a dictionary of bin edges for each feature.
    """
    return {
        'return_1': [-np.inf, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, np.inf],
        'vol_20': [0, 0.005, 0.01, 0.02, 0.03, 0.05, np.inf],
        'rsi_14': [0, 20, 40, 60, 80, 100],
        'ma_diff': [-np.inf, -50, -20, -10, 10, 20, 50, np.inf],
        'regime': [0, 1, 2, 3]
    }

def discretize_features(features: pd.DataFrame, bins: dict) -> pd.DataFrame:
    """
    Discretize continuous features into bins for mapping.
    """
    discretized = features.copy()

    for col, bin_edges in bins.items():
        if col in features.columns:
            discretized[col] = pd.cut(features[col], bins=bin_edges, labels=False, duplicates='drop')

    return discretized

def build_parameter_mapping(df: pd.DataFrame, periods: list, multipliers: list, bins: dict):
    """
    Build a mapping from discretized features to best performing supertrend parameters.
    Returns a dictionary mapping feature combinations to best parameter tuples.
    """
    combos, combo_signals, per_signal_pnls, per_signal_best = generate_signals_and_pnl(df, periods, multipliers)
    features = compute_features(df)

    # Discretize features
    discretized_features = discretize_features(features, bins)

    # Build mapping
    mapping = {}
    feature_counts = {}

    for i in range(len(df)):
        best = per_signal_best.iloc[i]
        if best is None:
            continue

        # Get discretized feature values as tuple (hashable key)
        feature_key = tuple(discretized_features.iloc[i].fillna(-1).astype(int))

        if feature_key not in mapping:
            mapping[feature_key] = {}
            feature_counts[feature_key] = 0

        # Count occurrences of each parameter for this feature combination
        param_str = str(best)
        if param_str not in mapping[feature_key]:
            mapping[feature_key][param_str] = 0
        mapping[feature_key][param_str] += 1
        feature_counts[feature_key] += 1

    # Convert counts to most frequent parameter for each feature combination
    final_mapping = {}
    for feature_key, param_counts in mapping.items():
        if feature_counts[feature_key] > 0:
            # Get the most frequent parameter
            best_param = max(param_counts.items(), key=lambda x: x[1])[0]
            final_mapping[feature_key] = eval(best_param)  # Convert string back to tuple

    return final_mapping, combos, features, per_signal_best

def sliding_window_mapping(df: pd.DataFrame, periods: list, multipliers: list, train_months: int = 3, test_months: int = 1):
    """
    Implement sliding window approach for building and testing parameter mapping.
    Returns training results and test performance for each window.
    """
    bins = create_feature_bins()
    results = []

    # Get unique months from the data
    df_months = df.index.to_period('M').unique()
    total_months = len(df_months)

    print(f"Total months available: {total_months}")
    print(f"Training months: {train_months}, Test months: {test_months}")
    print(f"Minimum months needed: {train_months + test_months}")
    print(f"Number of possible windows: {total_months - train_months - test_months + 1}")

    if total_months < train_months + test_months:
        print(f"⚠️  WARNING: Not enough data for {train_months} train + {test_months} test months")
        print(f"   Available: {total_months} months, Required: {train_months + test_months} months")
        return results  # Return empty results

    for start_month in range(0, total_months - train_months - test_months + 1, test_months):
        train_start = df_months[start_month]
        train_end = df_months[start_month + train_months - 1]
        test_start = df_months[start_month + train_months]
        test_end = df_months[start_month + train_months + test_months - 1]

        print(f"\nWindow: Train {train_start}-{train_end}, Test {test_start}-{test_end}")

        # Filter data for training and testing periods
        train_mask = (df.index.to_period('M') >= train_start) & (df.index.to_period('M') <= train_end)
        test_mask = (df.index.to_period('M') >= test_start) & (df.index.to_period('M') <= test_end)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) == 0 or len(test_df) == 0:
            print("Skipping window due to insufficient data")
            continue

        # Build mapping on training data
        mapping, combos, features, _ = build_parameter_mapping(train_df, periods, multipliers, bins)

        # Create fallback - most frequent parameter from training
        all_params = list(mapping.values())
        fallback_param = max(set(all_params), key=all_params.count) if all_params else (10, 3.0)

        # Store fallback parameter for later use (only if results list is not empty)
        if results:
            results[-1]['fallback_param'] = fallback_param

        # Calculate best parameters for test data (ground truth)
        _, _, _, test_per_signal_best = generate_signals_and_pnl(test_df, periods, multipliers)

        # Test mapping on test data
        test_features = compute_features(test_df)
        test_discretized = discretize_features(test_features, bins)

        # Apply mapping to test data
        test_predictions = []
        test_actuals = []
        no_mapping_count = 0
        total_signals = 0

        for i in range(len(test_df)):
            feature_key = tuple(test_discretized.iloc[i].fillna(-1).astype(int))
            predicted_param = mapping.get(feature_key, fallback_param)

            actual_best = test_per_signal_best.iloc[i]

            test_predictions.append(predicted_param)
            test_actuals.append(actual_best)

            if actual_best is not None:
                total_signals += 1
                if predicted_param == fallback_param:
                    no_mapping_count += 1

        # Calculate accuracy for this window
        valid_predictions = [(p, a) for p, a in zip(test_predictions, test_actuals) if p is not None and a is not None]
        if valid_predictions:
            correct = sum(1 for p, a in valid_predictions if str(p) == str(a))
            accuracy = correct / len(valid_predictions)
        else:
            accuracy = 0.0

        # Debug information with explanations
        print(f"  📊 Total signals in test period: {total_signals}")
        print(f"  🔍 No mapping found (using fallback): {no_mapping_count}")
        print(f"  ✅ Valid predictions made: {len(valid_predictions)}")
        print(f"  🎯 Correct predictions (exact matches): {correct}")
        print(f"  📈 Accuracy: {correct}/{len(valid_predictions)} = {accuracy:.3f}")

        # Calculate fallback rate for analysis
        fallback_rate = (no_mapping_count / total_signals) * 100 if total_signals > 0 else 0

        if no_mapping_count > 0:
            print(f"  ⚠️  Fallback rate: {fallback_rate:.1f}% of signals used default parameter")
            print(f"  💡 This suggests market conditions in test period differ from training period")

        results.append({
            'train_start': str(train_start),
            'train_end': str(train_end),
            'test_start': str(test_start),
            'test_end': str(test_end),
            'mapping': mapping,
            'test_accuracy': accuracy,
            'n_mapping_rules': len(mapping),
            'test_predictions': test_predictions,
            'test_actuals': test_actuals
        })

        print(f"Window accuracy: {accuracy:.3f}, Mapping rules: {len(mapping)}")

    return results

# -----------------------------
# Backtest using parameter mapping
# -----------------------------
def apply_mapping_and_backtest(df: pd.DataFrame, mapping: dict, combos: list, periods: list, multipliers: list):
    """
    Backtest using parameter mapping:
      - For each bar, lookup the best parameter combo based on current features.
      - If the predicted combo has a signal at that bar, we open/close using next-bar open.
      - Naive position sizing: fixed fraction (1% capital) per trade.
      - No slippage/commission (can be added easily).
    """
    # Generate signals for all combos
    _, combo_signals, _, _ = generate_signals_and_pnl(df, periods, multipliers)
    features = compute_features(df)
    bins = create_feature_bins()
    discretized_features = discretize_features(features, bins)

    # Create fallback - most frequent parameter from mapping
    all_params = list(mapping.values())
    fallback_param = max(set(all_params), key=all_params.count) if all_params else (10, 3.0)

    capital = 100000.0
    cash = capital
    holdings = 0.0
    last_entry = None
    trades = []

    # iterate through bars (stop one before last because we may use i+1 open)
    for i in range(len(df) - 1):
        # Get feature key for current bar
        feature_key = tuple(discretized_features.iloc[i].fillna(-1).astype(int))
        pred_combo = mapping.get(feature_key, fallback_param)

        # Skip if no signal for this parameter combo
        if pred_combo not in combo_signals:
            continue

        sig = combo_signals[pred_combo].iloc[i]
        # open
        if sig != 0 and holdings == 0:
            entry_price = df['open'].iloc[i+1]
            # naive size: 1% of capital
            size = (capital * 0.01) / entry_price
            holdings = size * entry_price
            cash -= size * entry_price
            last_entry = {'idx': i+1, 'price': entry_price, 'size': size, 'direction': sig}
        # close on opposite
        elif sig != 0 and holdings != 0 and last_entry is not None and sig == -last_entry['direction']:
            exit_price = df['open'].iloc[i+1]
            pnl = (exit_price - last_entry['price']) * last_entry['size'] * last_entry['direction']
            cash += last_entry['size'] * exit_price
            trades.append(pnl)
            holdings = 0
            last_entry = None

    # Close any remaining position at the end
    if holdings != 0 and last_entry is not None:
        exit_price = df['close'].iloc[-1]
        pnl = (exit_price - last_entry['price']) * last_entry['size'] * last_entry['direction']
        cash += last_entry['size'] * exit_price
        trades.append(pnl)

    total_pnl = sum(trades)
    return {'total_pnl': total_pnl, 'n_trades': len(trades), 'trades': trades}

def calculate_trading_metrics(trades: list, capital: float = 100000.0) -> dict:
    """
    Calculate comprehensive trading performance metrics.
    """
    if not trades:
        return {
            'total_pnl': 0.0,
            'n_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'profit_factor': 0.0,
            'total_return_pct': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }

    # Basic metrics
    total_pnl = sum(trades)
    n_trades = len(trades)
    winning_trades = [t for t in trades if t > 0]
    losing_trades = [t for t in trades if t < 0]

    win_rate = len(winning_trades) / n_trades * 100 if n_trades > 0 else 0
    avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = abs(sum(losing_trades) / len(losing_trades)) if losing_trades else 0
    max_win = max(winning_trades) if winning_trades else 0
    max_loss = abs(min(trades)) if trades else 0

    # Profit factor
    total_wins = sum(winning_trades)
    total_losses = abs(sum(losing_trades))
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    # Total return percentage
    total_return_pct = (total_pnl / capital) * 100

    # Maximum drawdown
    cumulative = 0
    peak = 0
    max_drawdown = 0

    for trade in trades:
        cumulative += trade
        peak = max(peak, cumulative)
        drawdown = peak - cumulative
        max_drawdown = max(max_drawdown, drawdown)

    max_drawdown_pct = (max_drawdown / capital) * 100 if capital > 0 else 0

    # Sharpe ratio (simplified - assumes risk-free rate = 0)
    if len(trades) > 1:
        returns = np.array(trades)
        sharpe_ratio = np.sqrt(len(trades)) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    else:
        sharpe_ratio = 0

    return {
        'total_pnl': total_pnl,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_win': max_win,
        'max_loss': max_loss,
        'profit_factor': profit_factor,
        'total_return_pct': total_return_pct,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio
    }

# -----------------------------
# Main execution
# -----------------------------
if __name__ == '__main__':
    # Path to your file (change if needed)
    data_path = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f'CSV not found at path: {data_path}. Update data_path variable in script.')

    # Read CSV (expects column named 'datetime')
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df = df.sort_values('Date').set_index('Date')
    df.index.name = 'datetime'   # rename index for consistency

    # Ensure required columns exist and are lowercase
    expected = ['open', 'high', 'low', 'close']
    lower_cols = {c: c.lower() for c in df.columns}
    # If columns are capitalized differently, try to rename
    df.columns = [c.lower() for c in df.columns]

    for c in expected:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in CSV. Found columns: {list(df.columns)}")

    # If volume missing, create a zero column to keep feature code simple
    if 'volume' not in df.columns:
        df['volume'] = 0

    periods = [7, 10, 14, 21, 25]
    multipliers = [1.0, 2.0, 3.0, 4.0, 5.0]

    print('Running sliding window parameter mapping...')
    print(f'Data period: {df.index.min()} to {df.index.max()}')
    window_results = sliding_window_mapping(df, periods, multipliers, train_months=6, test_months=1)

    # Save the mapping from the last window for future use
    if window_results:
        final_mapping = window_results[-1]['mapping']
        print(f'Saving final mapping with {len(final_mapping)} rules...')
        with open('auto_supertrend_mapping.pkl', 'wb') as f:
            pickle.dump({
                'mapping': final_mapping,
                'periods': periods,
                'multipliers': multipliers,
                'window_results': window_results
            }, f)

    # Run backtest using the final mapping on the entire dataset
    if window_results:
        final_mapping = window_results[-1]['mapping']
        print('Running backtest using final mapping...')
        backtest_result = apply_mapping_and_backtest(df, final_mapping, list(final_mapping.values()), periods, multipliers)

        # Calculate comprehensive trading metrics
        metrics = calculate_trading_metrics(backtest_result['trades'])
        print('\n=== COMPREHENSIVE TRADING METRICS ===')
        print(f"Total P&L: ₹{metrics['total_pnl']:.2f}")
        print(f"Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"Number of Trades: {metrics['n_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Average Win: ₹{metrics['avg_win']:.2f}")
        print(f"Average Loss: ₹{metrics['avg_loss']:.2f}")
        print(f"Best Trade: ₹{metrics['max_win']:.2f}")
        print(f"Worst Trade: ₹{metrics['max_loss']:.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Maximum Drawdown: ₹{metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")

        # Print summary of all windows
        print('\n=== SLIDING WINDOW SUMMARY ===')
        total_accuracy = 0
        total_rules = 0
        total_signals = 0
        total_no_mapping = 0
        total_correct = 0

        for i, result in enumerate(window_results):
            print(f"Window {i+1}: {result['train_start']}-{result['train_end']} -> {result['test_start']}-{result['test_end']}")
            print(f"  Accuracy: {result['test_accuracy']:.3f}, Rules: {result['n_mapping_rules']}")

            # Extract debug info from test_predictions and test_actuals if available
            if 'test_predictions' in result and 'test_actuals' in result:
                predictions = result['test_predictions']
                actuals = result['test_actuals']
                fallback_param = result.get('fallback_param', (10, 3.0))  # Default if not available

                signals_in_window = sum(1 for a in actuals if a is not None)
                no_mapping_in_window = sum(1 for p in predictions if p == fallback_param and any(a is not None for a in actuals))
                correct_in_window = sum(1 for p, a in zip(predictions, actuals) if p is not None and a is not None and str(p) == str(a))

                total_signals += signals_in_window
                total_no_mapping += no_mapping_in_window
                total_correct += correct_in_window

            total_accuracy += result['test_accuracy']
            total_rules += result['n_mapping_rules']

        avg_accuracy = total_accuracy / len(window_results) if window_results else 0
        avg_rules = total_rules / len(window_results) if window_results else 0

        print(f"\n=== OVERALL MAPPING PERFORMANCE ===")
        print(f"Average accuracy across all windows: {avg_accuracy:.3f}")
        print(f"Average mapping rules per window: {avg_rules:.1f}")
        print(f"Total windows processed: {len(window_results)}")
        print(f"Final mapping size: {len(final_mapping)} rules")

        # Show most common parameters in final mapping
        param_counts = {}
        for param in final_mapping.values():
            param_str = str(param)
            param_counts[param_str] = param_counts.get(param_str, 0) + 1

        most_common = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n=== MOST COMMON PARAMETERS ===")
        for i, (param, count) in enumerate(most_common, 1):
            percentage = (count / len(final_mapping)) * 100
            print(f"#{i}: {param} (used in {count} rules, {percentage:.1f}%)")

        # Analysis and recommendations
        print(f"\n=== PERFORMANCE ANALYSIS ===")
        if avg_accuracy > 0.7:
            print(f"🟢 EXCELLENT: High accuracy ({avg_accuracy:.3f}) - mapping works very well!")
        elif avg_accuracy > 0.5:
            print(f"🟡 GOOD: Moderate accuracy ({avg_accuracy:.3f}) - mapping shows promise")
        elif avg_accuracy > 0.3:
            print(f"🟠 MODERATE: Low accuracy ({avg_accuracy:.3f}) - may need parameter tuning")
        else:
            print(f"🔴 LOW: Very low accuracy ({avg_accuracy:.3f}) - significant issues detected")

        if avg_rules < 50:
            print(f"💡 Few mapping rules ({avg_rules:.0f}) - consider broader feature bins")
        elif avg_rules > 200:
            print(f"💡 Many mapping rules ({avg_rules:.0f}) - consider fewer feature bins")

        print(f"💡 Total unique market conditions identified: {len(final_mapping)}")
        print(f"💡 Strategy adapts to {len(final_mapping)} different market scenarios")

        # Calculate overall fallback rate
        overall_fallback_rate = (total_no_mapping / total_signals) * 100 if total_signals > 0 else 0

        # Explain current results
        print(f"\n=== UNDERSTANDING YOUR RESULTS ===")
        print(f"📊 Your mapping system learned {len(final_mapping)} rules from 6 months of training data")
        print(f"📈 Across all test periods, it encountered {total_signals} trading opportunities")
        print(f"🎯 For {total_correct} of these, the learned rules gave the optimal parameter")
        print(f"🔄 For {total_no_mapping} signals, it used the fallback parameter (most common from training)")

        if overall_fallback_rate > 80:
            print(f"\n⚠️  HIGH FALLBACK RATE DETECTED:")
            print(f"   • Test market conditions are very different from training period")
            print(f"   • Consider using more training data or adjusting feature bins")
            print(f"   • May indicate regime change or unusual market conditions")
        elif overall_fallback_rate > 50:
            print(f"\n🟡 MODERATE FALLBACK RATE:")
            print(f"   • Some market conditions are new/unseen")
            print(f"   • Mapping is learning but could be improved")
        else:
            print(f"\n🟢 LOW FALLBACK RATE:")
            print(f"   • Mapping generalizes well to new market conditions")
            print(f"   • Good balance between specificity and generalization")
    else:
        print('No valid windows found for mapping.')
