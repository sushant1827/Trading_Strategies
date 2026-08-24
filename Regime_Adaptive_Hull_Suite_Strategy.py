"""
Regime_Adaptive_Hull_Suite_Strategy.py
=======================================
Extends Hull_Suite_Strategy.py: instead of one fixed Hull variation/length for
the whole series, each bar is classified into a market phase:

  - Consolidation : choppy / no clean direction  (low Kaufman Efficiency Ratio)
  - Slow_Trend    : trending but at a measured pace
  - Fast_Trend    : trending with a fast ATR-normalized push

Each phase gets its own Hull variation/length and ATR stop multiplier. The
"active" Hull line at any bar is spliced from whichever phase that bar
belongs to (each candidate Hull line is itself causal, so splicing introduces
no look-ahead). Long entries/exits still fire on the MHULL/SHULL crossover
rule used in Hull_Suite_Strategy.py: confirmed at close, filled at the next
open, with an ATR trailing stop identical in mechanics to that script.

Universe, cost model, and backtest engine (vectorbt) are identical to
Hull_Suite_Strategy.py so results are directly comparable.
"""

import glob
import itertools
import os

import numpy as np
import pandas as pd
import vectorbt as vbt


# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = r'D:\Sushant\Fyers_AlgoTrade\Fyers_Data'
OUTPUT_DIR = os.path.join(BASE_DIR, 'PyScripts', 'regime_adaptive_hull_suite_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BACKTEST_START_DATE = pd.Timestamp('2021-01-01')
TIMEFRAMES = ['Daily', 'Weekly']

INIT_CASH = 100_000.0
ROUND_TRIP_CHARGES = 250.0
FIXED_FEE_PER_ORDER = ROUND_TRIP_CHARGES / 2
NSE_TRADING_DAYS_PER_YEAR = 252
NSE_TRADING_DAYS_PER_WEEK = 5

# --- Regime classification (causal: uses only bars up to and including i) ---
REGIME_LOOKBACK = 20          # window for efficiency ratio / velocity
REGIME_ATR_PERIOD = 14        # ATR period used to normalize velocity and the stop
# Thresholds calibrated on the pooled ER/velocity distribution across all BEES
# ETFs so each of the three phases gets a meaningful share of bars (roughly
# 40% Consolidation / 30% Slow_Trend / 30% Fast_Trend) instead of collapsing
# to two phases.
ER_CONSOLIDATION_MAX = 0.20    # efficiency ratio below this => choppy, no clean trend
VELOCITY_FAST_MIN = 0.55       # ATR-normalized displacement/bar at/above this => fast trend

# --- Per-regime Hull variation (fixed) + length/stop candidates (optimized) ---
REGIME_VARIATION = {'Consolidation': 'Thma', 'Slow_Trend': 'Hma', 'Fast_Trend': 'Ehma'}
REGIME_LENGTH_CANDIDATES = {'Consolidation': [100, 150], 'Slow_Trend': [55, 75], 'Fast_Trend': [20, 35]}
REGIME_STOP_MULT_CANDIDATES = {
    'Consolidation': [2.0, 2.5, 3.0], 'Slow_Trend': [2.0, 2.5, 3.0], 'Fast_Trend': [1.5, 2.0, 2.5],
}
REGIME_LABELS = ('Consolidation', 'Slow_Trend', 'Fast_Trend')


# ==============================================================================
# DATA LOADING
# ==============================================================================

def discover_bees_symbols(base_dir: str) -> list[tuple[str, str]]:
    """Return (symbol, daily CSV path) for every *BEES folder with daily data."""
    found = []
    for folder in sorted(glob.glob(os.path.join(base_dir, '*BEES'))):
        symbol = os.path.basename(folder)
        daily_files = glob.glob(os.path.join(folder, f'{symbol}_EQ_D_Min.csv'))
        if daily_files:
            found.append((symbol, daily_files[0]))
    return found


def load_daily(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date').drop_duplicates('Date').set_index('Date')
    return df.loc[df.index >= BACKTEST_START_DATE, ['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    aggregation = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    return df.resample('W-FRI').agg(aggregation).dropna(subset=['Open', 'High', 'Low', 'Close'])


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().to_numpy()


# ==============================================================================
# REGIME CLASSIFICATION
# ==============================================================================

def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Kaufman's Efficiency Ratio: net displacement / total path length over the
    window. Near 1 = a clean, efficient trend; near 0 = choppy/consolidating."""
    net_move = close.diff(window).abs()
    path_length = close.diff().abs().rolling(window).sum()
    return net_move / path_length.replace(0.0, np.nan)


def normalized_velocity(close: pd.Series, atr: pd.Series, window: int) -> pd.Series:
    """ATR-normalized displacement per bar over the window - the 'pace' of the
    move, independent of the instrument's absolute volatility scale."""
    velocity = close.diff(window) / (atr * np.sqrt(window))
    return velocity.replace([np.inf, -np.inf], np.nan)


def classify_regime(df: pd.DataFrame, window: int = REGIME_LOOKBACK,
                     atr_period: int = REGIME_ATR_PERIOD) -> pd.Series:
    """Causal per-bar market phase label using only data up to and including
    bar i (rolling windows only look backward), so no look-ahead bias."""
    atr = pd.Series(
        wilder_atr(df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy(), atr_period),
        index=df.index,
    )
    er = efficiency_ratio(df['Close'], window)
    velocity = normalized_velocity(df['Close'], atr, window).abs()

    regime = pd.Series(
        np.where(er < ER_CONSOLIDATION_MAX, 'Consolidation',
                 np.where(velocity >= VELOCITY_FAST_MIN, 'Fast_Trend', 'Slow_Trend')),
        index=df.index, dtype=object,
    )
    warmup = er.isna() | atr.isna()
    regime[warmup] = None
    return regime


# ==============================================================================
# HULL INDICATORS (identical math to Hull_Suite_Strategy.py)
# ==============================================================================

def pine_round_positive(value: float) -> int:
    return max(1, int(np.floor(value + 0.5)))


def pine_length_divisor(length: int, divisor: int) -> int:
    return max(1, int(length / divisor))


def wma(source: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)
    return source.rolling(length, min_periods=length).apply(
        lambda values: np.dot(values, weights) / weights.sum(), raw=True
    )


def hma(source: pd.Series, length: int) -> pd.Series:
    half_length = pine_length_divisor(length, 2)
    sqrt_length = pine_round_positive(np.sqrt(length))
    return wma(2.0 * wma(source, half_length) - wma(source, length), sqrt_length)


def ehma(source: pd.Series, length: int) -> pd.Series:
    half_length = pine_length_divisor(length, 2)
    sqrt_length = pine_round_positive(np.sqrt(length))
    return (2.0 * source.ewm(span=half_length, adjust=False, min_periods=half_length).mean()
            - source.ewm(span=length, adjust=False, min_periods=length).mean()).ewm(
                span=sqrt_length, adjust=False, min_periods=sqrt_length
            ).mean()


def thma(source: pd.Series, length: int) -> pd.Series:
    return wma(
        3.0 * wma(source, pine_length_divisor(length, 3))
        - wma(source, pine_length_divisor(length, 2))
        - wma(source, length),
        length,
    )


def hull(source: pd.Series, variation: str, length: int) -> pd.Series:
    if variation == 'Hma':
        return hma(source, length)
    if variation == 'Ehma':
        return ehma(source, length)
    if variation == 'Thma':
        return thma(source, pine_length_divisor(length, 2))
    raise ValueError(f'Unsupported Hull variation: {variation}')


# ==============================================================================
# SIGNAL GENERATION AND BACKTEST
# ==============================================================================

def build_regime_signals(
    df: pd.DataFrame, regime: pd.Series, combo: dict[str, tuple[int, float]],
) -> dict[str, pd.Series]:
    """combo maps each regime label to (hull_length, atr_stop_mult). The active
    Hull line/stop multiplier at each bar is spliced from that bar's own regime."""
    selected_hull = pd.Series(np.nan, index=df.index)
    selected_stop_mult = pd.Series(np.nan, index=df.index)
    for label, (length, stop_mult) in combo.items():
        mask = (regime == label).to_numpy()
        hull_line = hull(df['Close'], REGIME_VARIATION[label], length)
        selected_hull[mask] = hull_line[mask]
        selected_stop_mult[mask] = stop_mult

    mhull = selected_hull
    shull = mhull.shift(2)
    valid = mhull.notna() & shull.notna() & regime.notna()

    # Pine crossover: strictly less/greater on the previous bar.
    long_entry = valid & mhull.gt(shull) & mhull.shift(1).lt(shull.shift(1))
    long_exit = valid & mhull.lt(shull) & mhull.shift(1).gt(shull.shift(1))

    atr = pd.Series(
        wilder_atr(df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy(), REGIME_ATR_PERIOD),
        index=df.index,
    )
    stop = mhull - selected_stop_mult * atr
    stop_level = stop.where(mhull.gt(shull))

    return {
        'long_entry': long_entry.fillna(False),
        'long_exit': long_exit.fillna(False),
        'stop_level': stop_level,
        'regime': regime,
    }


def run_backtest(df: pd.DataFrame, signals: dict[str, pd.Series], freq: str) -> vbt.Portfolio:
    # Signals are known only after each bar closes; transact at the next open.
    entries = signals['long_entry'].shift(1, fill_value=False)
    exits = signals['long_exit'].shift(1, fill_value=False)
    execution_price = df['Open'].copy()

    prev_stop = signals['stop_level'].shift(1)
    stop_hit = df['Low'].le(prev_stop) & prev_stop.notna()
    # Fill at the stop price unless price already gapped below it at the open.
    execution_price = execution_price.where(~stop_hit, np.minimum(df['Open'], prev_stop))
    exits = exits | stop_hit.fillna(False)

    return vbt.Portfolio.from_signals(
        df['Close'], entries=entries, exits=exits, price=execution_price,
        open=df['Open'], high=df['High'], low=df['Low'],
        init_cash=INIT_CASH, fixed_fees=FIXED_FEE_PER_ORDER, freq=freq,
        size_granularity=1,
    )


def annualization_year_freq(timeframe: str) -> pd.Timedelta:
    if timeframe == 'Daily':
        return pd.Timedelta(days=NSE_TRADING_DAYS_PER_YEAR)
    if timeframe == 'Weekly':
        return pd.Timedelta(days=7 * NSE_TRADING_DAYS_PER_YEAR / NSE_TRADING_DAYS_PER_WEEK)
    raise ValueError(f'Unsupported timeframe: {timeframe}')


def portfolio_metrics(portfolio: vbt.Portfolio, year_freq: pd.Timedelta) -> dict[str, float]:
    trades = portfolio.trades.records_readable
    return {
        'total_return_pct': portfolio.total_return() * 100,
        'cagr_pct': portfolio.annualized_return(year_freq=year_freq) * 100,
        'sharpe_ratio': portfolio.sharpe_ratio(year_freq=year_freq),
        'max_drawdown_pct': portfolio.max_drawdown() * 100,
        'num_trades': len(trades),
        'win_rate_pct': portfolio.trades.win_rate() * 100 if len(trades) else 0.0,
    }


def composite_score(row: pd.Series) -> float:
    """Reward risk-adjusted growth and penalize deep drawdowns."""
    return (0.5 * row['sharpe_ratio']
            + 0.3 * (row['cagr_pct'] / 20.0)
            - 0.2 * (abs(row['max_drawdown_pct']) / 20.0))


def regime_distribution(regime: pd.Series) -> dict[str, float]:
    counts = regime.value_counts(normalize=True, dropna=True) * 100
    return {f'{label}_pct': counts.get(label, 0.0) for label in REGIME_LABELS}


# ==============================================================================
# OPTIMIZATION
# ==============================================================================

def optimize_symbol(symbol: str, df: pd.DataFrame, timeframe: str, freq: str) -> pd.DataFrame:
    regime = classify_regime(df)
    dist = regime_distribution(regime)
    rows = []
    year_freq = annualization_year_freq(timeframe)

    per_regime_options = {
        label: list(itertools.product(REGIME_LENGTH_CANDIDATES[label], REGIME_STOP_MULT_CANDIDATES[label]))
        for label in REGIME_LABELS
    }
    for cons, slow, fast in itertools.product(
        per_regime_options['Consolidation'], per_regime_options['Slow_Trend'], per_regime_options['Fast_Trend']
    ):
        combo = {'Consolidation': cons, 'Slow_Trend': slow, 'Fast_Trend': fast}
        signals = build_regime_signals(df, regime, combo)
        portfolio = run_backtest(df, signals, freq)
        rows.append({
            'Symbol': symbol, 'Timeframe': timeframe,
            'Consolidation_Length': cons[0], 'Consolidation_Stop_Mult': cons[1],
            'Slow_Trend_Length': slow[0], 'Slow_Trend_Stop_Mult': slow[1],
            'Fast_Trend_Length': fast[0], 'Fast_Trend_Stop_Mult': fast[1],
            **dist,
            **portfolio_metrics(portfolio, year_freq),
        })
    return pd.DataFrame(rows)


def best_combo_from_row(row: pd.Series) -> dict[str, tuple[int, float]]:
    return {
        'Consolidation': (int(row['Consolidation_Length']), row['Consolidation_Stop_Mult']),
        'Slow_Trend': (int(row['Slow_Trend_Length']), row['Slow_Trend_Stop_Mult']),
        'Fast_Trend': (int(row['Fast_Trend_Length']), row['Fast_Trend_Stop_Mult']),
    }


def main() -> None:
    symbols = discover_bees_symbols(BASE_DIR)
    if not symbols:
        print('No *BEES symbols with daily data found under', BASE_DIR)
        return

    print(f'Found {len(symbols)} BEES symbol(s) with daily data: {[symbol for symbol, _ in symbols]}')
    min_bars_needed = max(max(v) for v in REGIME_LENGTH_CANDIDATES.values()) + REGIME_LOOKBACK
    all_results = []
    best_rows = []

    for symbol, csv_path in symbols:
        daily = load_daily(csv_path)
        data_by_timeframe = {'Daily': (daily, '1D'), 'Weekly': (resample_weekly(daily), '1W')}

        for timeframe in TIMEFRAMES:
            df, freq = data_by_timeframe[timeframe]
            if len(df) < min_bars_needed:
                print(f'Skipping {symbol} {timeframe}: not enough bars ({len(df)})')
                continue

            print(f'Optimizing {symbol} ({timeframe}, {len(df)} bars)...')
            results = optimize_symbol(symbol, df, timeframe, freq)
            all_results.append(results)

            valid = results[np.isfinite(results['sharpe_ratio']) & (results['num_trades'] > 0)]
            if valid.empty:
                print(f'  No parameter combo produced trades for {symbol} {timeframe}')
                continue

            best = valid.assign(composite_score=valid.apply(composite_score, axis=1)).sort_values(
                'composite_score', ascending=False
            ).iloc[0]
            best_rows.append(best)

            regime = classify_regime(df)
            signals = build_regime_signals(df, regime, best_combo_from_row(best))
            portfolio = run_backtest(df, signals, freq)
            portfolio.trades.records_readable.to_csv(
                os.path.join(OUTPUT_DIR, f'{symbol}_{timeframe}_best_trades.csv'), index=False
            )
            print(
                f'  Best: Consolidation=Thma({int(best["Consolidation_Length"])},'
                f'{best["Consolidation_Stop_Mult"]:.1f}x) '
                f'Slow=Hma({int(best["Slow_Trend_Length"])},{best["Slow_Trend_Stop_Mult"]:.1f}x) '
                f'Fast=Ehma({int(best["Fast_Trend_Length"])},{best["Fast_Trend_Stop_Mult"]:.1f}x) -> '
                f'return={best["total_return_pct"]:.2f}%, cagr={best["cagr_pct"]:.2f}%, '
                f'sharpe={best["sharpe_ratio"]:.2f}, maxdd={best["max_drawdown_pct"]:.2f}%, '
                f'score={best["composite_score"]:.2f}, trades={int(best["num_trades"])} | '
                f'regime mix: consolidation={best["Consolidation_pct"]:.0f}%, '
                f'slow={best["Slow_Trend_pct"]:.0f}%, fast={best["Fast_Trend_pct"]:.0f}%'
            )

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, 'regime_adaptive_hull_suite_optimization_results.csv'), index=False
        )

    if best_rows:
        best_df = pd.DataFrame(best_rows).reset_index(drop=True)
        best_df.to_csv(os.path.join(OUTPUT_DIR, 'regime_adaptive_hull_suite_best_params.csv'), index=False)
        print('\nBest parameters per symbol/timeframe saved to regime_adaptive_hull_suite_best_params.csv')
        print(best_df.to_string(index=False))


if __name__ == '__main__':
    main()
