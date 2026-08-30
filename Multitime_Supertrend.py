"""Bias-aware multi-timeframe Supertrend backtests using VectorBT.

Input bars are assumed to have opening-time timestamps. Higher timeframe bars
are available only at their closing boundary. A filled order costs Rs. 125;
therefore, a normal entry and exit costs Rs. 250. No slippage is modelled.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import vectorbt as vbt


TIMEFRAME_PRESETS: Mapping[str, tuple[str, str, str]] = {
    "hourly_daily_weekly": ("1h", "1D", "1W"),
    "daily_weekly_monthly": ("1D", "1W", "1ME"),
}


@dataclass(frozen=True)
class BacktestResult:
    preset: str
    parameters: tuple[dict[str, float | int | str], ...]
    signals: pd.DataFrame
    portfolio: vbt.Portfolio


class MultiTimeframeSupertrend:
    """Three-timeframe Supertrend confirmation with no same-bar fills."""

    def __init__(self, initial_capital: float = 100_000.0, order_cost: float = 125.0) -> None:
        if initial_capital <= 0 or order_cost < 0:
            raise ValueError("Capital must be positive and order cost cannot be negative")
        self.initial_capital = float(initial_capital)
        self.order_cost = float(order_cost)

    @staticmethod
    def normalise_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
        required = {"Open", "High", "Low", "Close"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Data must use a DatetimeIndex")
        clean = data.copy()
        clean.index = pd.to_datetime(clean.index)
        clean = clean[~clean.index.duplicated(keep="last")].sort_index()
        clean = clean.dropna(subset=["Open", "High", "Low", "Close"])
        if clean.empty:
            raise ValueError("No valid OHLC rows remain after cleaning")
        if "Volume" not in clean:
            clean["Volume"] = 0.0
        return clean[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    @staticmethod
    def calculate_supertrend(
        data: pd.DataFrame, period: int = 10, multiplier: float = 3.0, source: str = "hl2"
    ) -> pd.DataFrame:
        """Calculate Supertrend solely from current and prior completed bars."""
        if period < 2 or multiplier <= 0:
            raise ValueError("period must be at least 2 and multiplier must be positive")
        sources = {
            "hl2": (data["High"] + data["Low"]) / 2.0,
            "hlc3": (data["High"] + data["Low"] + data["Close"]) / 3.0,
            "ohlc4": (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4.0,
            "close": data["Close"],
        }
        if source not in sources:
            raise ValueError("source must be one of: hl2, hlc3, ohlc4, close")
        previous_close = data["Close"].shift(1)
        true_range = pd.concat(
            [data["High"] - data["Low"], (data["High"] - previous_close).abs(),
             (data["Low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        atr = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        upper = sources[source] + multiplier * atr
        lower = sources[source] - multiplier * atr
        final_upper = pd.Series(np.nan, index=data.index)
        final_lower = pd.Series(np.nan, index=data.index)
        trend = pd.Series(0, index=data.index, dtype=np.int8)
        first_valid = atr.first_valid_index()
        if first_valid is None:
            return pd.DataFrame({"supertrend": np.nan, "trend": trend, "atr": atr})
        start = data.index.get_loc(first_valid)
        final_upper.iloc[start], final_lower.iloc[start], trend.iloc[start] = upper.iloc[start], lower.iloc[start], 1
        for row in range(start + 1, len(data)):
            prior_upper, prior_lower = final_upper.iloc[row - 1], final_lower.iloc[row - 1]
            prior_close = data["Close"].iloc[row - 1]
            final_upper.iloc[row] = upper.iloc[row] if upper.iloc[row] < prior_upper or prior_close > prior_upper else prior_upper
            final_lower.iloc[row] = lower.iloc[row] if lower.iloc[row] > prior_lower or prior_close < prior_lower else prior_lower
            close = data["Close"].iloc[row]
            trend.iloc[row] = 1 if close > prior_upper else -1 if close < prior_lower else trend.iloc[row - 1]
        supertrend = pd.Series(np.where(trend == 1, final_lower, final_upper), index=data.index)
        supertrend[trend == 0] = np.nan
        return pd.DataFrame({"supertrend": supertrend, "trend": trend, "atr": atr})

    @staticmethod
    def resample_ohlcv(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Create completed bars labelled at the time their information is known."""
        resample_options = {"label": "right", "closed": "left"}
        if timeframe.endswith(("min", "h")):
            resample_options.update({"origin": "start_day", "offset": "15min"})
        return data.resample(timeframe, **resample_options).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna()

    def build_signals(
        self, data: pd.DataFrame, timeframes: Sequence[str],
        parameters: Sequence[Mapping[str, float | int | str]],
    ) -> pd.DataFrame:
        if len(timeframes) != 3 or len(parameters) != 3:
            raise ValueError("Exactly three timeframes and three parameter dictionaries are required")
        base = self.normalise_ohlcv(data)
        signals = base[["Open", "High", "Low", "Close"]].copy()
        columns: list[str] = []
        for number, (timeframe, config) in enumerate(zip(timeframes, parameters), start=1):
            higher = self.resample_ohlcv(base, timeframe)
            trend = self.calculate_supertrend(higher, **dict(config))["trend"]
            column = f"trend_{number}_{timeframe}"
            signals[column] = trend.reindex(base.index, method="ffill").fillna(0).astype(np.int8)
            columns.append(column)
        signals["long_regime"] = signals[columns].eq(1).all(axis=1)
        signals["short_regime"] = signals[columns].eq(-1).all(axis=1)
        new_long = signals["long_regime"] & ~signals["long_regime"].shift(fill_value=False)
        new_short = signals["short_regime"] & ~signals["short_regime"].shift(fill_value=False)
        signals["long_entries"] = new_long.shift(1, fill_value=False) & signals["long_regime"]
        signals["short_entries"] = new_short.shift(1, fill_value=False) & signals["short_regime"]
        signals["long_exits"] = ~signals["long_regime"]
        signals["short_exits"] = ~signals["short_regime"]
        signals.loc[signals.index[-1], ["long_exits", "short_exits"]] = True
        return signals

    def backtest(
        self, data: pd.DataFrame, preset: str,
        parameters: Sequence[Mapping[str, float | int | str]],
    ) -> BacktestResult:
        if preset not in TIMEFRAME_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from {sorted(TIMEFRAME_PRESETS)}")
        signals = self.build_signals(data, TIMEFRAME_PRESETS[preset], parameters)
        portfolio = vbt.Portfolio.from_signals(
            close=signals["Close"], entries=signals["long_entries"], exits=signals["long_exits"],
            short_entries=signals["short_entries"], short_exits=signals["short_exits"],
            price=signals["Open"], size=np.inf, fixed_fees=self.order_cost,
            init_cash=self.initial_capital, upon_opposite_entry="close",
        )
        return BacktestResult(preset, tuple(dict(config) for config in parameters), signals, portfolio)

    def optimize(
        self, training_data: pd.DataFrame, preset: str,
        parameter_grid: Mapping[str, Iterable[float | int | str]],
    ) -> pd.DataFrame:
        """Rank only train-period configurations. Keep the selected winner fixed for test."""
        configs = [dict(zip(parameter_grid, values)) for values in product(*(parameter_grid[key] for key in parameter_grid))]
        results: list[dict[str, object]] = []
        for triple in product(configs, repeat=3):
            statistics = self.backtest(training_data, preset, triple).portfolio.stats()
            total_return = float(statistics["Total Return [%]"])
            drawdown = abs(float(statistics["Max Drawdown [%]"]))
            results.append({"params_1": triple[0], "params_2": triple[1], "params_3": triple[2],
                            "total_return_pct": total_return, "max_drawdown_pct": drawdown,
                            "total_trades": int(statistics["Total Trades"]),
                            "profit_factor": float(statistics["Profit Factor"]),
                            "score": total_return / drawdown if drawdown else total_return})
        return pd.DataFrame(results).sort_values("score", ascending=False, ignore_index=True)


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    timestamp = "DateTime" if "DateTime" in data else "Date" if "Date" in data else None
    if timestamp is None:
        raise ValueError("CSV must contain a Date or DateTime column")
    data[timestamp] = pd.to_datetime(data[timestamp])
    return data.set_index(timestamp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bias-aware VectorBT multi-timeframe Supertrend")
    default_csv = Path("Nifty50_Index/NIFTY50_INDEX_1_Min.csv")
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=default_csv,
        help=f"OHLCV CSV with Date/DateTime and Open/High/Low/Close (default: {default_csv})",
    )
    parser.add_argument("--preset", choices=TIMEFRAME_PRESETS, default="hourly_daily_weekly")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--period", type=int, default=10)
    parser.add_argument("--multiplier", type=float, default=3.0)
    parser.add_argument("--source", choices=("hl2", "hlc3", "ohlc4", "close"), default="hl2")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"CSV file not found: {args.csv.resolve()}")
    config = {"period": args.period, "multiplier": args.multiplier, "source": args.source}
    result = MultiTimeframeSupertrend(args.capital).backtest(load_ohlcv_csv(args.csv), args.preset, (config, config, config))
    print(result.portfolio.stats().to_string())
    print("\nOrders:")
    print(result.portfolio.orders.records_readable.to_string(index=False))


if __name__ == "__main__":
    main()