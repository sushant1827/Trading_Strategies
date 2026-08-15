"""
Out-of-sample swing-trading experiment for Renko + HA + SuperTrend.

Parameters are selected on the first TRAIN_FRACTION of history, frozen, then
the baseline and entry/exit variants are compared only on the untouched test
period. The original strategy module is not modified.
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
import vectorbt as vbt

import Renko_HA_Supertrend_Strategy as base


OUTPUT_DIR = os.path.join(base.BASE_DIR, 'PyScripts', 'renko_ha_supertrend_swing_exit_experiment')
TRAIN_FRACTION = 0.70
DEFAULT_SYMBOLS = ['NIFTYBEES', 'BANKBEES']
DEFAULT_TIMEFRAMES = ['Daily']

VARIANTS = {
    'Baseline': {'rearm_after_exit': False, 'cooldown_bars': 0, 'close_confirmed_stop': False},
    'Fresh_Rearm': {'rearm_after_exit': True, 'cooldown_bars': 0, 'close_confirmed_stop': False},
    'Cooldown_3': {'rearm_after_exit': False, 'cooldown_bars': 3, 'close_confirmed_stop': False},
    'Cooldown_5': {'rearm_after_exit': False, 'cooldown_bars': 5, 'close_confirmed_stop': False},
    'Close_Confirmed_Stop': {'rearm_after_exit': False, 'cooldown_bars': 0, 'close_confirmed_stop': True},
    'Rearm_Plus_Close_Stop': {'rearm_after_exit': True, 'cooldown_bars': 0, 'close_confirmed_stop': True},
}


def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    aggregation = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    return df.resample('ME').agg(aggregation).dropna(subset=['Open', 'High', 'Low', 'Close'])


def year_frequency(timeframe: str) -> pd.Timedelta:
    if timeframe == 'Daily':
        return pd.Timedelta(days=base.NSE_TRADING_DAYS_PER_YEAR)
    if timeframe == 'Weekly':
        return pd.Timedelta(days=7 * base.NSE_TRADING_DAYS_PER_YEAR / base.NSE_TRADING_DAYS_PER_WEEK)
    if timeframe == 'Monthly':
        return pd.Timedelta(days=365.25)
    raise ValueError(f'Unsupported timeframe: {timeframe}')


def build_variant_orders(
    df: pd.DataFrame,
    signals: dict,
    start_index: int,
    rearm_after_exit: bool,
    cooldown_bars: int,
    close_confirmed_stop: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Build causal orders from a flat position at ``start_index``.

    A fresh re-arm requires the three-way entry condition to become false after
    an exit before another aligned signal can open a new position. A close-
    confirmed stop exits at the next open only after the prior close is below
    the prior SuperTrend line; it preserves the baseline's signal exits.
    """
    long_entry = signals['long_entry'].to_numpy(dtype=bool)
    scheduled_entries = signals['long_entry'].shift(1, fill_value=False).to_numpy(dtype=bool)
    scheduled_exits = signals['long_exit'].shift(1, fill_value=False).to_numpy(dtype=bool)
    previous_st = signals['supertrend'].shift(1).to_numpy(dtype=float)
    previous_st_dir = signals['supertrend_dir'].shift(1).to_numpy(dtype=np.int64)

    close_stop_signal = (
        signals['supertrend_dir'].shift(1).eq(1)
        & df['Close'].lt(signals['supertrend'].shift(1))
    ).shift(1, fill_value=False).to_numpy(dtype=bool)

    open_prices = df['Open'].to_numpy(dtype=float)
    low_prices = df['Low'].to_numpy(dtype=float)
    entries = np.zeros(len(df), dtype=bool)
    exits = np.zeros(len(df), dtype=bool)
    execution_prices = open_prices.copy()
    exit_reasons = np.full(len(df), '', dtype=object)

    in_position = False
    waiting_for_rearm = False
    bars_remaining = 0

    for index in range(start_index, len(df)):
        if waiting_for_rearm and not long_entry[index]:
            waiting_for_rearm = False

        if bars_remaining > 0:
            bars_remaining -= 1

        if in_position:
            stop_hit = (
                not close_confirmed_stop
                and previous_st_dir[index] == 1
                and np.isfinite(previous_st[index])
                and low_prices[index] <= previous_st[index]
            )
            exit_at_open = scheduled_exits[index] or (close_confirmed_stop and close_stop_signal[index])

            if exit_at_open or stop_hit:
                exits[index] = True
                if stop_hit:
                    execution_prices[index] = min(open_prices[index], previous_st[index])
                    exit_reasons[index] = 'intrabar_supertrend_stop'
                elif close_confirmed_stop and close_stop_signal[index]:
                    exit_reasons[index] = 'close_confirmed_supertrend_stop'
                else:
                    exit_reasons[index] = 'strategy_signal_exit'
                in_position = False
                waiting_for_rearm = rearm_after_exit
                bars_remaining = cooldown_bars
                continue

        if not in_position and scheduled_entries[index] and not waiting_for_rearm and bars_remaining == 0:
            entries[index] = True
            in_position = True

    return (
        pd.Series(entries, index=df.index),
        pd.Series(exits, index=df.index),
        pd.Series(execution_prices, index=df.index),
        pd.Series(exit_reasons, index=df.index, name='Exit_Reason'),
    )


def run_variant_backtest(
    df: pd.DataFrame,
    signals: dict,
    test_start: int,
    freq: str,
    settings: dict,
) -> tuple[vbt.Portfolio, pd.Series]:
    entries, exits, prices, exit_reasons = build_variant_orders(df, signals, test_start, **settings)
    test_df = df.iloc[test_start:]
    portfolio = vbt.Portfolio.from_signals(
        test_df['Close'],
        entries=entries.iloc[test_start:],
        exits=exits.iloc[test_start:],
        price=prices.iloc[test_start:],
        open=test_df['Open'], high=test_df['High'], low=test_df['Low'],
        init_cash=base.INIT_CASH, fixed_fees=base.FIXED_FEE_PER_ORDER,
        freq=freq, size_granularity=1,
    )
    return portfolio, exit_reasons.iloc[test_start:]


def choose_train_parameters(symbol: str, train_df: pd.DataFrame, timeframe: str, freq: str, min_trades: int) -> pd.Series | None:
    """Choose one parameter set from train data only, then freeze it for test."""
    rows = []
    annualization = year_frequency(timeframe)
    grid = itertools.product(
        base.RENKO_ATR_PERIODS, base.RENKO_MULTIPLIERS,
        base.ST_ATR_PERIODS, base.ST_MULTIPLIERS, base.EXIT_MODES,
    )
    for renko_period, renko_mult, st_period, st_mult, exit_mode in grid:
        signals = base.build_signals(train_df, renko_period, renko_mult, st_period, st_mult, exit_mode)
        portfolio = base.run_backtest(train_df, signals, freq)
        metrics = base.portfolio_metrics(portfolio, annualization)
        rows.append({
            'Symbol': symbol, 'Timeframe': timeframe,
            'Renko_ATR_Period': renko_period, 'Renko_ATR_Mult': renko_mult,
            'ST_ATR_Period': st_period, 'ST_Multiplier': st_mult,
            'Exit_Mode': exit_mode, **metrics,
        })

    candidates = pd.DataFrame(rows)
    candidates = candidates[np.isfinite(candidates['sharpe_ratio']) & candidates['num_trades'].ge(min_trades)]
    if candidates.empty:
        return None
    candidates = candidates.assign(composite_score=candidates.apply(base.composite_score, axis=1))
    return candidates.sort_values('composite_score', ascending=False).iloc[0]


def evaluate_symbol_timeframe(
    symbol: str,
    df: pd.DataFrame,
    timeframe: str,
    freq: str,
    min_train_trades: int,
) -> tuple[list[dict], pd.Series | None, dict[str, pd.DataFrame]]:
    split_at = int(len(df) * TRAIN_FRACTION)
    train_df = df.iloc[:split_at]
    selected = choose_train_parameters(symbol, train_df, timeframe, freq, min_train_trades)
    if selected is None:
        return [], None, {}

    signals = base.build_signals(
        df, int(selected['Renko_ATR_Period']), selected['Renko_ATR_Mult'],
        int(selected['ST_ATR_Period']), selected['ST_Multiplier'], selected['Exit_Mode'],
    )
    annualization = year_frequency(timeframe)
    rows = []
    trades_by_variant = {}
    for variant, settings in VARIANTS.items():
        portfolio, reasons = run_variant_backtest(df, signals, split_at, freq, settings)
        metrics = base.portfolio_metrics(portfolio, annualization)
        rows.append({
            **selected.to_dict(), 'Variant': variant,
            'Train_End': train_df.index[-1], 'Test_Start': df.index[split_at],
            **metrics,
        })
        trades = portfolio.trades.records_readable.copy()
        if not trades.empty:
            trades['Exit_Reason'] = trades['Exit Timestamp'].map(reasons).replace('', 'open_position')
        trades_by_variant[variant] = trades
    return rows, selected, trades_by_variant


def paired_comparison(results: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        'Symbol', 'Timeframe', 'Renko_ATR_Period', 'Renko_ATR_Mult',
        'ST_ATR_Period', 'ST_Multiplier', 'Exit_Mode',
    ]
    metrics = ['total_return_pct', 'cagr_pct', 'sharpe_ratio', 'max_drawdown_pct', 'num_trades', 'win_rate_pct']
    baseline = results[results['Variant'] == 'Baseline'].set_index(identifiers)[metrics]
    rows = []
    for variant, group in results[results['Variant'] != 'Baseline'].groupby('Variant'):
        candidate = group.set_index(identifiers)[metrics]
        aligned_baseline, aligned_candidate = baseline.align(candidate, join='inner', axis=0)
        delta = aligned_candidate - aligned_baseline
        delta.columns = [f'delta_{column}' for column in delta.columns]
        comparison = aligned_candidate.reset_index()[identifiers].copy()
        comparison['Variant'] = variant
        rows.append(pd.concat([comparison, delta.reset_index(drop=True)], axis=1))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Out-of-sample swing exit experiment for Renko HA SuperTrend.')
    parser.add_argument('--symbols', nargs='+', default=DEFAULT_SYMBOLS)
    parser.add_argument('--timeframes', nargs='+', choices=['Daily', 'Weekly', 'Monthly'], default=DEFAULT_TIMEFRAMES)
    parser.add_argument('--min-train-trades', type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_rows = []
    selection_rows = []
    all_trades = {}
    discovered = dict(base.discover_bees_symbols(base.BASE_DIR))

    for symbol in args.symbols:
        csv_path = discovered.get(symbol)
        if csv_path is None:
            print(f'Skipping {symbol}: no daily BEES data found.')
            continue
        daily = base.load_daily(csv_path)
        data_by_timeframe = {
            'Daily': (daily, '1D'),
            'Weekly': (base.resample_weekly(daily), '1W'),
            'Monthly': (resample_monthly(daily), '30D'),
        }
        for timeframe in args.timeframes:
            df, freq = data_by_timeframe[timeframe]
            if len(df) < 40:
                print(f'Skipping {symbol} {timeframe}: insufficient bars ({len(df)}).')
                continue
            print(f'Testing {symbol} {timeframe} ({len(df)} bars): train-select, then frozen OOS comparison...')
            rows, selected, trades_by_variant = evaluate_symbol_timeframe(
                symbol, df, timeframe, freq, args.min_train_trades,
            )
            if selected is None:
                print(f'  No train parameter set reached {args.min_train_trades} trades.')
                continue
            all_rows.extend(rows)
            selection_rows.append(selected.to_dict())
            all_trades[(symbol, timeframe)] = trades_by_variant

    if not all_rows:
        print('No eligible tests were produced.')
        return

    results = pd.DataFrame(all_rows)
    selections = pd.DataFrame(selection_rows)
    comparison = paired_comparison(results)
    results.to_csv(os.path.join(OUTPUT_DIR, 'swing_exit_oos_results.csv'), index=False)
    selections.to_csv(os.path.join(OUTPUT_DIR, 'swing_exit_train_selected_params.csv'), index=False)
    comparison.to_csv(os.path.join(OUTPUT_DIR, 'swing_exit_oos_deltas_vs_baseline.csv'), index=False)
    for (symbol, timeframe), trades_by_variant in all_trades.items():
        for variant, trades in trades_by_variant.items():
            filename = f'{symbol}_{timeframe}_{variant.lower()}_oos_trades.csv'
            trades.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

    print('\nOut-of-sample results:')
    columns = ['Symbol', 'Timeframe', 'Variant', 'total_return_pct', 'cagr_pct', 'sharpe_ratio', 'max_drawdown_pct', 'num_trades', 'win_rate_pct']
    print(results[columns].to_string(index=False, float_format=lambda value: f'{value:.2f}'))
    print(f'\nOutputs written to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()