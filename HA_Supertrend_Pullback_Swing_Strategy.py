"""
Heikin-Ashi + Supertrend pullback swing strategy.

The strategy waits for an established Supertrend trend, a counter-trend
Heikin-Ashi pullback, and the first Heikin-Ashi candle back in the trend
direction. A stop-entry order is then placed beyond the confirming HA candle.
The initial and trailing stop is the prior confirmed Supertrend value. Positions
are also closed after a fixed, short holding period.
"""

import os
import itertools

import numpy as np
import pandas as pd
import vectorbt as vbt

import Renko_HA_Supertrend_Strategy as base


OUTPUT_DIR = os.path.join(base.BASE_DIR, 'PyScripts', 'ha_supertrend_pullback_swing_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMEFRAMES = ['Daily', 'Weekly']
TRADE_DIRECTION = 'long_only'  # 'long_only', 'short_only', or 'both'

ST_ATR_PERIOD = 21
ST_MULTIPLIER = 3.0
MIN_TREND_CANDLES = 3
MAX_HOLDING_BARS = 3

# Parameters are selected using bars before this date only. The remaining bars
# are reserved for the out-of-sample report and must not influence selection.
TRAIN_END_DATE = pd.Timestamp('2025-01-01')
OPT_ST_ATR_PERIODS = [10, 14, 21, 28]
OPT_ST_MULTIPLIERS = [1.5, 2.0, 2.5, 3.0]
OPT_MIN_TREND_CANDLES = [2, 3, 4]
OPT_MAX_HOLDING_BARS = [2, 3, 4, 5]
MIN_TRAIN_TRADES_PER_SYMBOL = 3
MIN_TRAIN_SYMBOL_COVERAGE = 0.60


def build_setup_signals(
    df: pd.DataFrame,
    st_period: int = ST_ATR_PERIOD,
    st_multiplier: float = ST_MULTIPLIER,
    min_trend_candles: int = MIN_TREND_CANDLES,
) -> dict:
    """Identify confirmed HA pullback reversals without using future bars."""
    if min_trend_candles < 1:
        raise ValueError('min_trend_candles must be at least 1')

    ha = base.heikin_ashi(df)
    supertrend = base.supertrend(df, st_period, st_multiplier)
    ha_green = ha['HA_Close'] > ha['HA_Open']
    ha_red = ha['HA_Close'] < ha['HA_Open']
    indicators_ready = supertrend['ATR'].notna()

    long_confirmation = pd.Series(False, index=df.index)
    short_confirmation = pd.Series(False, index=df.index)
    trend_state = 0
    trend_candles = 0
    pullback_candles = 0

    for index in range(len(df)):
        if not indicators_ready.iloc[index]:
            trend_state = 0
            trend_candles = 0
            pullback_candles = 0
            continue

        st_direction = supertrend['ST_Dir'].iloc[index]
        if st_direction == 1:
            if ha_green.iloc[index]:
                if trend_state == 2 and pullback_candles > 0:
                    long_confirmation.iloc[index] = True
                trend_state = 1
                trend_candles += 1
                pullback_candles = 0
            elif ha_red.iloc[index] and trend_state == 1 and trend_candles >= min_trend_candles:
                trend_state = 2
                pullback_candles += 1
            else:
                trend_state = 0
                trend_candles = 0
                pullback_candles = 0
        elif st_direction == -1:
            if ha_red.iloc[index]:
                if trend_state == -2 and pullback_candles > 0:
                    short_confirmation.iloc[index] = True
                trend_state = -1
                trend_candles += 1
                pullback_candles = 0
            elif ha_green.iloc[index] and trend_state == -1 and trend_candles >= min_trend_candles:
                trend_state = -2
                pullback_candles += 1
            else:
                trend_state = 0
                trend_candles = 0
                pullback_candles = 0
        else:
            trend_state = 0
            trend_candles = 0
            pullback_candles = 0

    return {
        'ha': ha,
        'supertrend': supertrend['ST'],
        'supertrend_dir': supertrend['ST_Dir'],
        'long_confirmation': long_confirmation,
        'short_confirmation': short_confirmation,
        'long_entry_stop': ha['HA_High'].where(long_confirmation),
        'short_entry_stop': ha['HA_Low'].where(short_confirmation),
    }


def build_execution_signals(
    df: pd.DataFrame,
    setup: dict,
    max_holding_bars: int = MAX_HOLDING_BARS,
    trade_direction: str = TRADE_DIRECTION,
) -> dict:
    """Turn confirmed setups into next-bar stop entries and causal exits."""
    if max_holding_bars < 1:
        raise ValueError('max_holding_bars must be at least 1')
    if trade_direction not in {'long_only', 'short_only', 'both'}:
        raise ValueError("trade_direction must be 'long_only', 'short_only', or 'both'")

    index = df.index
    long_entries = pd.Series(False, index=index)
    long_exits = pd.Series(False, index=index)
    short_entries = pd.Series(False, index=index)
    short_exits = pd.Series(False, index=index)
    execution_price = df['Open'].copy()
    exit_reason = pd.Series('', index=index, dtype=object)

    position = 0
    entry_bar = -1
    pending_side = 0
    pending_stop = np.nan

    for bar in range(len(df)):
        previous_st = setup['supertrend'].iloc[bar - 1] if bar else np.nan
        previous_st_dir = setup['supertrend_dir'].iloc[bar - 1] if bar else 0

        if position == 1:
            stop_hit = previous_st_dir == 1 and np.isfinite(previous_st) and df['Low'].iloc[bar] <= previous_st
            time_exit = bar - entry_bar >= max_holding_bars
            if stop_hit or time_exit:
                long_exits.iloc[bar] = True
                execution_price.iloc[bar] = min(df['Open'].iloc[bar], previous_st) if stop_hit else df['Open'].iloc[bar]
                exit_reason.iloc[bar] = 'supertrend_stop' if stop_hit else 'time_exit'
                position = 0
                entry_bar = -1
        elif position == -1:
            stop_hit = previous_st_dir == -1 and np.isfinite(previous_st) and df['High'].iloc[bar] >= previous_st
            time_exit = bar - entry_bar >= max_holding_bars
            if stop_hit or time_exit:
                short_exits.iloc[bar] = True
                execution_price.iloc[bar] = max(df['Open'].iloc[bar], previous_st) if stop_hit else df['Open'].iloc[bar]
                exit_reason.iloc[bar] = 'supertrend_stop' if stop_hit else 'time_exit'
                position = 0
                entry_bar = -1

        if position == 0 and pending_side:
            trend_is_valid = previous_st_dir == pending_side
            if not trend_is_valid:
                pending_side = 0
                pending_stop = np.nan
            elif pending_side == 1 and df['High'].iloc[bar] >= pending_stop:
                long_entries.iloc[bar] = True
                execution_price.iloc[bar] = max(df['Open'].iloc[bar], pending_stop)
                position = 1
                entry_bar = bar
                pending_side = 0
                pending_stop = np.nan
            elif pending_side == -1 and df['Low'].iloc[bar] <= pending_stop:
                short_entries.iloc[bar] = True
                execution_price.iloc[bar] = min(df['Open'].iloc[bar], pending_stop)
                position = -1
                entry_bar = bar
                pending_side = 0
                pending_stop = np.nan

        if position == 0 and pending_side == 0:
            if trade_direction in {'long_only', 'both'} and setup['long_confirmation'].iloc[bar]:
                pending_side = 1
                pending_stop = setup['long_entry_stop'].iloc[bar]
            elif trade_direction in {'short_only', 'both'} and setup['short_confirmation'].iloc[bar]:
                pending_side = -1
                pending_stop = setup['short_entry_stop'].iloc[bar]

    return {
        'long_entries': long_entries,
        'long_exits': long_exits,
        'short_entries': short_entries,
        'short_exits': short_exits,
        'execution_price': execution_price,
        'exit_reason': exit_reason,
    }


def run_backtest(df: pd.DataFrame, execution: dict, freq: str) -> vbt.Portfolio:
    """Run the stop-entry strategy using the existing ETF cost model."""
    return vbt.Portfolio.from_signals(
        df['Close'],
        entries=execution['long_entries'],
        exits=execution['long_exits'],
        short_entries=execution['short_entries'],
        short_exits=execution['short_exits'],
        price=execution['execution_price'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        init_cash=base.INIT_CASH,
        fixed_fees=base.FIXED_FEE_PER_ORDER,
        freq=freq,
        size_granularity=1,
    )


def execution_slice(execution: dict, mask: pd.Series) -> dict:
    """Keep only executable signals in a date range while starting flat."""
    return {
        name: values.loc[mask] if isinstance(values, pd.Series) else values
        for name, values in execution.items()
    }


def evaluate_parameters(
    df: pd.DataFrame,
    freq: str,
    timeframe: str,
    st_period: int,
    st_multiplier: float,
    min_trend_candles: int,
    max_holding_bars: int,
    mask: pd.Series,
) -> dict:
    """Run one parameter set over a specified date range, starting flat."""
    setup = build_setup_signals(df, st_period, st_multiplier, min_trend_candles)
    execution = build_execution_signals(df, setup, max_holding_bars, TRADE_DIRECTION)
    sliced_df = df.loc[mask]
    portfolio = run_backtest(sliced_df, execution_slice(execution, mask), freq)
    return base.portfolio_metrics(portfolio, base.annualization_year_freq(sliced_df, timeframe))


def optimization_score(metrics: dict) -> float:
    """Use the parent strategy's reward-versus-drawdown ranking consistently."""
    if metrics['num_trades'] < MIN_TRAIN_TRADES_PER_SYMBOL or not np.isfinite(metrics['sharpe_ratio']):
        return np.nan
    return base.composite_score(pd.Series(metrics))


def optimize_timeframe(
    data: list[tuple[str, pd.DataFrame, str]],
    timeframe: str,
) -> tuple[pd.DataFrame, dict | None]:
    """Select one robust configuration from pre-2025 results across all symbols."""
    rows = []
    parameter_grid = itertools.product(
        OPT_ST_ATR_PERIODS,
        OPT_ST_MULTIPLIERS,
        OPT_MIN_TREND_CANDLES,
        OPT_MAX_HOLDING_BARS,
    )
    for st_period, st_multiplier, min_trend_candles, max_holding_bars in parameter_grid:
        scores = []
        metrics_by_symbol = []
        for _, df, freq in data:
            train_mask = df.index < TRAIN_END_DATE
            if train_mask.sum() <= st_period + max_holding_bars:
                continue
            metrics = evaluate_parameters(
                df, freq, timeframe, st_period, st_multiplier,
                min_trend_candles, max_holding_bars, train_mask,
            )
            metrics_by_symbol.append(metrics)
            score = optimization_score(metrics)
            if np.isfinite(score):
                scores.append(score)

        coverage = len(scores) / len(metrics_by_symbol) if metrics_by_symbol else 0.0
        rows.append({
            'Timeframe': timeframe,
            'ST_ATR_Period': st_period,
            'ST_Multiplier': st_multiplier,
            'Min_Trend_Candles': min_trend_candles,
            'Max_Holding_Bars': max_holding_bars,
            'Train_Symbols_Evaluated': len(metrics_by_symbol),
            'Train_Symbols_Qualified': len(scores),
            'Train_Symbol_Coverage': coverage,
            'Train_Median_Score': np.median(scores) if scores else np.nan,
            'Train_Median_Sharpe': np.median([m['sharpe_ratio'] for m in metrics_by_symbol]) if metrics_by_symbol else np.nan,
            'Train_Median_CAGR_Pct': np.median([m['cagr_pct'] for m in metrics_by_symbol]) if metrics_by_symbol else np.nan,
            'Train_Median_Max_Drawdown_Pct': np.median([m['max_drawdown_pct'] for m in metrics_by_symbol]) if metrics_by_symbol else np.nan,
        })

    results = pd.DataFrame(rows)
    eligible = results[
        (results['Train_Symbol_Coverage'] >= MIN_TRAIN_SYMBOL_COVERAGE)
        & np.isfinite(results['Train_Median_Score'])
    ]
    if eligible.empty:
        return results, None
    best = eligible.sort_values(
        ['Train_Median_Score', 'Train_Symbol_Coverage'], ascending=[False, False]
    ).iloc[0].to_dict()
    return results, best


def run_optimization() -> None:
    """Optimize shared parameters on 2021-2024 data and report unseen results."""
    symbols = base.discover_bees_symbols(base.BASE_DIR)
    if not symbols:
        print('No *BEES symbols with daily data found under', base.BASE_DIR)
        return

    data_by_timeframe: dict[str, list[tuple[str, pd.DataFrame, str]]] = {timeframe: [] for timeframe in TIMEFRAMES}
    for symbol, csv_path in symbols:
        daily = base.load_daily(csv_path)
        data_by_timeframe['Daily'].append((symbol, daily, '1D'))
        data_by_timeframe['Weekly'].append((symbol, base.resample_weekly(daily), '1W'))

    print(
        f'Optimizing {len(OPT_ST_ATR_PERIODS) * len(OPT_ST_MULTIPLIERS) * len(OPT_MIN_TREND_CANDLES) * len(OPT_MAX_HOLDING_BARS)} '
        f'parameter sets per timeframe on data before {TRAIN_END_DATE.date()} only...'
    )
    optimization_tables = []
    evaluation_rows = []
    for timeframe, data in data_by_timeframe.items():
        data = [(symbol, df, freq) for symbol, df, freq in data if not df.empty]
        results, best = optimize_timeframe(data, timeframe)
        optimization_tables.append(results)
        if best is None:
            print(f'{timeframe}: no parameter set met the train trade-coverage requirement.')
            continue

        print(
            f'{timeframe} train winner: ST=({int(best["ST_ATR_Period"])}, {best["ST_Multiplier"]}), '
            f'trend={int(best["Min_Trend_Candles"])}, hold={int(best["Max_Holding_Bars"])}; '
            f'median score={best["Train_Median_Score"]:.2f}, coverage={best["Train_Symbol_Coverage"]:.0%}'
        )
        for symbol, df, freq in data:
            for split, mask in (
                ('Train', df.index < TRAIN_END_DATE),
                ('Test', df.index >= TRAIN_END_DATE),
            ):
                if mask.sum() <= int(best['ST_ATR_Period']) + int(best['Max_Holding_Bars']):
                    continue
                metrics = evaluate_parameters(
                    df, freq, timeframe,
                    int(best['ST_ATR_Period']), best['ST_Multiplier'],
                    int(best['Min_Trend_Candles']), int(best['Max_Holding_Bars']), mask,
                )
                evaluation_rows.append({
                    'Symbol': symbol,
                    'Timeframe': timeframe,
                    'Split': split,
                    'ST_ATR_Period': int(best['ST_ATR_Period']),
                    'ST_Multiplier': best['ST_Multiplier'],
                    'Min_Trend_Candles': int(best['Min_Trend_Candles']),
                    'Max_Holding_Bars': int(best['Max_Holding_Bars']),
                    **metrics,
                })

    if optimization_tables:
        pd.concat(optimization_tables, ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, 'ha_supertrend_pullback_optimization_grid.csv'), index=False
        )
    if evaluation_rows:
        evaluation = pd.DataFrame(evaluation_rows)
        evaluation.to_csv(
            os.path.join(OUTPUT_DIR, 'ha_supertrend_pullback_optimized_train_test_results.csv'), index=False
        )
        summary = evaluation.groupby(['Timeframe', 'Split'], as_index=False).agg(
            symbols=('Symbol', 'nunique'),
            median_total_return_pct=('total_return_pct', 'median'),
            median_cagr_pct=('cagr_pct', 'median'),
            median_sharpe_ratio=('sharpe_ratio', 'median'),
            median_max_drawdown_pct=('max_drawdown_pct', 'median'),
            total_trades=('num_trades', 'sum'),
        )
        summary.to_csv(
            os.path.join(OUTPUT_DIR, 'ha_supertrend_pullback_optimized_summary.csv'), index=False
        )
        print('\nFrozen-parameter train/test summary:')
        print(summary.to_string(index=False, float_format=lambda value: f'{value:.2f}'))
    print(f'Optimization outputs written to: {OUTPUT_DIR}')


def save_symbol_outputs(
    symbol: str,
    timeframe: str,
    portfolio: vbt.Portfolio,
    setup: dict,
    execution: dict,
) -> None:
    prefix = f'{symbol}_{timeframe}'
    trades = portfolio.trades.records_readable.copy()
    if not trades.empty:
        trades['Exit_Reason'] = trades['Exit Timestamp'].map(execution['exit_reason']).replace('', 'open_position')
    trades.to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_trades.csv'), index=False)

    setups = pd.DataFrame({
        'HA_High': setup['ha']['HA_High'],
        'HA_Low': setup['ha']['HA_Low'],
        'Supertrend': setup['supertrend'],
        'Supertrend_Direction': setup['supertrend_dir'],
        'Long_Confirmation': setup['long_confirmation'],
        'Short_Confirmation': setup['short_confirmation'],
        'Long_Entry_Stop': setup['long_entry_stop'],
        'Short_Entry_Stop': setup['short_entry_stop'],
        'Long_Entry_Filled': execution['long_entries'],
        'Short_Entry_Filled': execution['short_entries'],
    })
    setups.to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_signals.csv'))


def main() -> None:
    symbols = base.discover_bees_symbols(base.BASE_DIR)
    if not symbols:
        print('No *BEES symbols with daily data found under', base.BASE_DIR)
        return

    print(
        f'Running HA + Supertrend pullback swing strategy for {len(symbols)} symbols: '
        f'ST=({ST_ATR_PERIOD}, {ST_MULTIPLIER}), trend={MIN_TREND_CANDLES} bars, '
        f'holding={MAX_HOLDING_BARS} bars, direction={TRADE_DIRECTION}'
    )
    results = []
    for symbol, csv_path in symbols:
        daily = base.load_daily(csv_path)
        data_by_timeframe = {
            'Daily': (daily, '1D'),
            'Weekly': (base.resample_weekly(daily), '1W'),
        }
        for timeframe in TIMEFRAMES:
            df, freq = data_by_timeframe[timeframe]
            if len(df) < ST_ATR_PERIOD + MIN_TREND_CANDLES + MAX_HOLDING_BARS + 2:
                print(f'Skipping {symbol} {timeframe}: not enough bars ({len(df)})')
                continue

            setup = build_setup_signals(df)
            execution = build_execution_signals(df, setup)
            portfolio = run_backtest(df, execution, freq)
            metrics = base.portfolio_metrics(portfolio, base.annualization_year_freq(df, timeframe))
            results.append({'Symbol': symbol, 'Timeframe': timeframe, **metrics})
            save_symbol_outputs(symbol, timeframe, portfolio, setup, execution)
            print(
                f'{symbol} {timeframe}: return={metrics["total_return_pct"]:.2f}%, '
                f'sharpe={metrics["sharpe_ratio"]:.2f}, trades={metrics["num_trades"]}'
            )

    if results:
        summary = pd.DataFrame(results)
        summary.to_csv(os.path.join(OUTPUT_DIR, 'ha_supertrend_pullback_swing_summary.csv'), index=False)
        print(f'Outputs written to: {OUTPUT_DIR}')


if __name__ == '__main__':
    run_optimization()