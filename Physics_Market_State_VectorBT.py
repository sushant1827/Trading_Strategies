"""Physics-inspired market-state strategy using local BEES OHLC data and VectorBT.

This is a fixed-parameter research baseline, not a production trading recommendation.
Signals use information known at each bar close and are filled at the next bar's
actual open. Trading charges are fixed at Rs 125 per order (Rs 250 round trip)
and slippage is explicitly zero.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

BASE_DIR = Path(r"D:\Sushant\Fyers_AlgoTrade\Fyers_Data")
OUTPUT_DIR = BASE_DIR / "PyScripts" / "physics_market_state_outputs"
TIMEFRAMES = {
    "daily": {"source": "daily", "freq": "1D"},
    "weekly": {"source": "weekly", "freq": "1W"},
}
INITIAL_CASH = 100_000.0
ROUND_TRIP_COST = 250.0
FIXED_FEE_PER_ORDER = ROUND_TRIP_COST / 2.0
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
TEST_START = pd.Timestamp("2024-01-01")

VELOCITY_WINDOW = 5
ACCELERATION_LAG = 3
EQUILIBRIUM_WINDOW = 20
SHORT_VOL_WINDOW = 5
LONG_VOL_WINDOW = 20
ENTROPY_WINDOW = 20
MAX_ENTRY_ENTROPY = 0.95

# Compact, pre-declared grid. Parameter selection uses only data through TRAIN_END.
VELOCITY_WINDOWS = (5, 10, 20)
ACCELERATION_LAGS = (2, 3, 5)
EQUILIBRIUM_WINDOWS = (20, 40)
SHORT_VOL_WINDOWS = (5, 10)
LONG_VOL_WINDOWS = (20, 40)
MAX_ENTRY_ENTROPIES = (0.85, 0.95, 1.05)
SCORE_RETURN_WEIGHT = 0.30
SCORE_DRAWDOWN_WEIGHT = 0.30
SCORE_SHARPE_WEIGHT = 0.40
RETURN_SCALE_PCT = 20.0
DRAWDOWN_SCALE_PCT = 20.0
MIN_TRAIN_TRADES = 8


def discover_bees_files() -> list[Path]:
    """Return one daily OHLC file from every first-level *BEES directory."""
    files: list[Path] = []
    pattern = "*_EQ_D_Min.csv"
    for directory in sorted(BASE_DIR.iterdir()):
        if directory.is_dir() and directory.name.upper().endswith("BEES"):
            matches = sorted(directory.glob(pattern))
            if matches:
                files.append(matches[0])
    return files


def load_ohlc(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["Date"])
    data = data.sort_values("Date").drop_duplicates("Date").set_index("Date")
    columns = ["Open", "High", "Low", "Close", "Volume"]
    data = data.loc[:, columns].apply(pd.to_numeric, errors="coerce").dropna(subset=columns[:4])
    return data.astype(float)


def resample_weekly(data: pd.DataFrame) -> pd.DataFrame:
    """Create Friday-ending weekly bars from complete daily OHLC bars."""
    weekly = data.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return weekly.dropna(subset=["Open", "High", "Low", "Close"])


def trailing_valid(series: pd.Series, window: int, operation: str) -> pd.Series:
    """Roll only over valid intraday observations, then restore original timestamps."""
    valid = series.dropna()
    if operation == "mean":
        result = valid.rolling(window, min_periods=window).mean()
    elif operation == "std":
        result = valid.rolling(window, min_periods=window).std(ddof=0)
    else:
        raise ValueError(f"Unsupported rolling operation: {operation}")
    return result.reindex(series.index)


def shannon_entropy(returns: pd.Series, window: int) -> pd.Series:
    """Trailing entropy of negative, neutral, and positive intraday return states."""
    values = returns.dropna()
    states = pd.Series(np.select([values < 0, values > 0], [0, 2], default=1), index=values.index)

    def entropy(window: np.ndarray) -> float:
        probabilities = np.bincount(window.astype(int), minlength=3) / len(window)
        probabilities = probabilities[probabilities > 0]
        return float(-(probabilities * np.log(probabilities)).sum())

    result = states.rolling(window, min_periods=window).apply(entropy, raw=True)
    return result.reindex(returns.index)


def build_state(data: pd.DataFrame, parameters: dict[str, float | int]) -> pd.DataFrame:
    close = data["Close"]
    log_return = np.log(close).diff()

    velocity_window = int(parameters["velocity_window"])
    acceleration_lag = int(parameters["acceleration_lag"])
    equilibrium_window = int(parameters["equilibrium_window"])
    short_vol_window = int(parameters["short_vol_window"])
    long_vol_window = int(parameters["long_vol_window"])
    velocity = trailing_valid(log_return, velocity_window, "mean")
    acceleration = velocity.dropna().diff(acceleration_lag).reindex(data.index)
    short_volatility = trailing_valid(log_return, short_vol_window, "std")
    long_volatility = trailing_valid(log_return, long_vol_window, "std")
    equilibrium = close.rolling(equilibrium_window, min_periods=equilibrium_window).mean()
    displacement_z = (close - equilibrium) / close.rolling(
        equilibrium_window, min_periods=equilibrium_window
    ).std(ddof=0).replace(0.0, np.nan)
    entropy = shannon_entropy(log_return, long_vol_window)
    energy = velocity.pow(2) * long_volatility

    state = pd.DataFrame(
        {
            "Close": close,
            "Velocity": velocity,
            "Acceleration": acceleration,
            "ShortVolatility": short_volatility,
            "LongVolatility": long_volatility,
            "Energy": energy,
            "Entropy": entropy,
            "DisplacementZ": displacement_z,
        },
        index=data.index,
    )
    state["EntrySignal"] = (
        state["Velocity"].gt(0)
        & state["Acceleration"].gt(0)
        & state["DisplacementZ"].gt(0)
        & state["ShortVolatility"].gt(state["LongVolatility"])
        & state["Entropy"].le(float(parameters["max_entry_entropy"]))
    )
    state["ExitSignal"] = (
        state["Velocity"].le(0)
        | state["Acceleration"].lt(0)
        | state["DisplacementZ"].le(0)
    )
    return state


def backtest(data: pd.DataFrame, state: pd.DataFrame, frequency: str) -> vbt.Portfolio:
    entries = state["EntrySignal"].fillna(False).shift(1, fill_value=False)
    exits = state["ExitSignal"].fillna(False).shift(1, fill_value=False)
    return vbt.Portfolio.from_signals(
        close=data["Close"],
        entries=entries,
        exits=exits,
        price=data["Open"],
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        init_cash=INITIAL_CASH,
        fixed_fees=FIXED_FEE_PER_ORDER,
        fees=0.0,
        slippage=0.0,
        freq=frequency,
        size_granularity=1.0,
    )


def metrics(portfolio: vbt.Portfolio, data: pd.DataFrame) -> dict[str, float | int | str]:
    trades = portfolio.trades
    trade_count = int(trades.count())
    start_date = data.index[0]
    end_date = data.index[-1]
    duration_days = (end_date - start_date).days
    duration_years = duration_days / 365.25
    cagr = (
        (portfolio.final_value() / INITIAL_CASH) ** (1 / duration_years) - 1
        if duration_years > 0
        else np.nan
    )
    return {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "duration_days": duration_days,
        "duration_years": duration_years,
        "total_return_pct": float(portfolio.total_return() * 100),
        "cagr_pct": float(cagr * 100),
        "sharpe_ratio": float(portfolio.sharpe_ratio()) if trade_count else np.nan,
        "final_value": float(portfolio.final_value()),
        "max_drawdown_pct": float(portfolio.max_drawdown() * 100),
        "trade_count": trade_count,
        "win_rate_pct": float(trades.win_rate() * 100) if trade_count else 0.0,
        "profit_factor": float(trades.profit_factor()) if trade_count else np.nan,
    }


def parameter_grid() -> list[dict[str, float | int]]:
    return [
        {
            "velocity_window": velocity_window,
            "acceleration_lag": acceleration_lag,
            "equilibrium_window": equilibrium_window,
            "short_vol_window": short_vol_window,
            "long_vol_window": long_vol_window,
            "max_entry_entropy": max_entry_entropy,
        }
        for velocity_window, acceleration_lag, equilibrium_window, short_vol_window,
        long_vol_window, max_entry_entropy in product(
            VELOCITY_WINDOWS,
            ACCELERATION_LAGS,
            EQUILIBRIUM_WINDOWS,
            SHORT_VOL_WINDOWS,
            LONG_VOL_WINDOWS,
            MAX_ENTRY_ENTROPIES,
        )
        if short_vol_window < long_vol_window
    ]


def composite_score(metric: dict[str, float | int | str]) -> float:
    """Reward return and Sharpe while penalizing drawdown; reject too-few-trade results."""
    sharpe = float(metric["sharpe_ratio"])
    if int(metric["trade_count"]) < MIN_TRAIN_TRADES or not np.isfinite(sharpe):
        return -np.inf
    return (
        SCORE_RETURN_WEIGHT * (float(metric["total_return_pct"]) / RETURN_SCALE_PCT)
        - SCORE_DRAWDOWN_WEIGHT * (abs(float(metric["max_drawdown_pct"])) / DRAWDOWN_SCALE_PCT)
        + SCORE_SHARPE_WEIGHT * sharpe
    )


def optimize_parameters(
    data: pd.DataFrame, frequency: str
) -> tuple[dict[str, float | int] | None, pd.DataFrame]:
    train_data = data.loc[data.index <= TRAIN_END]
    results: list[dict[str, float | int | str]] = []
    for parameters in parameter_grid():
        state = build_state(train_data, parameters)
        metric = metrics(backtest(train_data, state, frequency), train_data)
        result: dict[str, float | int | str] = {**parameters, **metric}
        result["composite_score"] = composite_score(metric)
        results.append(result)
    ranking = pd.DataFrame(results).sort_values("composite_score", ascending=False).reset_index(drop=True)
    viable = ranking[np.isfinite(ranking["composite_score"])]
    if viable.empty:
        return None, ranking
    best = viable.iloc[0]
    return {key: best[key] for key in parameter_grid()[0]}, ranking


def split_metrics(
    data: pd.DataFrame, state: pd.DataFrame, split: str, frequency: str
) -> dict[str, float | int | str]:
    if split == "train":
        mask = data.index <= TRAIN_END
    elif split == "test":
        mask = data.index >= TEST_START
    else:
        mask = pd.Series(True, index=data.index)
    subset = data.loc[mask]
    if subset.empty:
        return {
            "start_date": None,
            "end_date": None,
            "duration_days": 0,
            "duration_years": 0.0,
            "total_return_pct": np.nan,
            "cagr_pct": np.nan,
            "sharpe_ratio": np.nan,
            "final_value": np.nan,
            "max_drawdown_pct": np.nan,
            "trade_count": 0,
            "win_rate_pct": np.nan,
            "profit_factor": np.nan,
        }
    return metrics(backtest(subset, state.loc[mask], frequency), subset)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = discover_bees_files()
    if not files:
        raise FileNotFoundError(f"No *BEES/*_EQ_D_Min.csv files found under {BASE_DIR}")

    summary_rows: list[dict[str, float | int | str]] = []
    for path in files:
        symbol = path.parent.name.upper()
        daily_data = load_ohlc(path)
        for timeframe, config in TIMEFRAMES.items():
            data = daily_data if config["source"] == "daily" else resample_weekly(daily_data)
            best_parameters, ranking = optimize_parameters(data, config["freq"])
            ranking.insert(0, "timeframe", timeframe)
            ranking.insert(0, "symbol", symbol)
            ranking.to_csv(OUTPUT_DIR / f"{symbol}_{timeframe}_optimization.csv", index=False)
            if best_parameters is None:
                print(
                    f"Skipping {symbol} {timeframe}: no parameter set reached "
                    f"{MIN_TRAIN_TRADES} training trades"
                )
                continue
            state = build_state(data, best_parameters)
            portfolio = backtest(data, state, config["freq"])
            state.assign(Open=data["Open"], High=data["High"], Low=data["Low"]).to_csv(
                OUTPUT_DIR / f"{symbol}_{timeframe}_state.csv"
            )
            portfolio.trades.records_readable.to_csv(
                OUTPUT_DIR / f"{symbol}_{timeframe}_trades.csv", index=False
            )
            for split in ("full", "train", "test"):
                row: dict[str, float | int | str] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "split": split,
                    "bars": len(data),
                    "selection": "train_best_composite",
                    "composite_score": float(ranking.iloc[0]["composite_score"]),
                    **best_parameters,
                }
                row.update(split_metrics(data, state, split, config["freq"]))
                summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "physics_market_state_summary.csv", index=False)
    summary[summary["split"] == "test"].to_csv(
        OUTPUT_DIR / "physics_market_state_oos_summary.csv", index=False
    )
    print("Physics Market State VectorBT backtest complete")
    print(f"Symbols: {len(files)} | Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Costs: Rs {ROUND_TRIP_COST:.0f} round trip (Rs {FIXED_FEE_PER_ORDER:.0f} per order) | Slippage: 0")
    print(f"Parameter grid: {len(parameter_grid())} combinations; selection: train through {TRAIN_END.date()}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()