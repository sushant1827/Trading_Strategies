# FabTrader Algorithmic Trading Platform
# ----------------------------------------
# Copyright (c) 2022 FabTrader (Unit of Rough Sketch Company)
#
# LICENSE: PROPRIETARY SOFTWARE
# - This software is the exclusive property of FabTrader.
# - Unauthorized copying, modification, distribution, or use is strictly prohibited.
# - Written permission from the author is required for any use beyond personal,
#   non-commercial purposes.
#
# CONTACT:
# - Website: https://fabtrader.in
# - Email: hello@fabtrader.in
#
# Usage: Internal use only. Not for commercial redistribution.
# Permissions and licensing inquiries should be directed to the contact email.
"""
ETF Shop Backtester
Python 3.11+ style, object-oriented, production-quality.

Replace `get_historical_data` placeholder with real data loader.
"""
from __future__ import annotations
import os
import math
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

# ----------------------------
# Placeholder: data fetching
# ----------------------------
def get_historical_data(symbol: str, start_date: date, end_date: date, interval: str = "1d") -> pd.DataFrame:
    """
    Placeholder function - replace this with your data loader.
    Should return a DataFrame with columns ['Open','High','Low','Close','Volume'] and a DatetimeIndex named 'Date'.
    Index must be tz-naive dates at daily frequency (market days only).
    """
    # Example raising to indicate to the integrator: implement this
    raise NotImplementedError("Please implement get_historical_data(symbol, start_date, end_date, interval).")

# ----------------------------
# Data structures / dataclasses
# ----------------------------
@dataclass
class TradeLot:
    symbol: str
    entry_date: pd.Timestamp
    qty: int
    entry_price: float
    # will be filled when closed
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    brokerage: float = 0.0
    pnl: Optional[float] = None
    net_pnl: Optional[float] = None

    def to_series_on_close(self) -> Dict[str, Any]:
        """Return a dict for adding to trade log DataFrame (filled when closed)."""
        assert self.exit_price is not None and self.exit_date is not None, "Lot not closed yet"
        gross = (self.exit_price - self.entry_price) * self.qty
        net = gross - self.brokerage
        pnl_pct = (self.exit_price / self.entry_price - 1.0) * 100.0
        net_pct = (1 + net / (self.entry_price * self.qty)) - 1.0
        return {
            "tradingSymbol": self.symbol,
            "tradeState": "completed",
            "entry_date": self.entry_date.date(),
            "direction": "Long",
            "qty": self.qty,
            "entry": self.entry_price,
            "Target": None,
            "StopLoss": None,
            "exit_date": self.exit_date.date(),
            "exit": self.exit_price,
            "pnl": round(gross, 2),
            "pnl%": round(pnl_pct, 2),
            "brokerage": round(self.brokerage, 2),
            "netPnl": round(net, 2),
            "netPnl%": round(net_pct * 100.0, 2),
        }

# ----------------------------
# Backtester class
# ----------------------------
class Backtester:
    def __init__(
        self,
        start_date: date,
        end_date: date,
        instruments: List[str],
        initial_capital: float = 400000.0,
        timeframe: str = "1d",
        static_allocation: float = 10000.0,
        dynamic_divisor: float = 40.0,
        sizing_mode: str = "static",  # 'static' or 'dynamic'
        averaging_threshold: float = 0.0314,  # 3.14% drop threshold for averaging (as fraction)
        target_pct: float = 0.05,  # 5% target (as fraction)
        brokerage_per_trade: float = 40.0,
        slippage: float = 0.0,
        results_path: str = os.path.join("Results", "tradebook.csv"),
        verbose: bool = True,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.instruments = instruments
        self.initial_capital = initial_capital
        self.timeframe = timeframe
        self.static_allocation = static_allocation
        self.dynamic_divisor = dynamic_divisor
        self.sizing_mode = sizing_mode.lower()
        assert self.sizing_mode in ("static", "dynamic"), "sizing_mode must be 'static' or 'dynamic'"
        self.averaging_threshold = averaging_threshold
        self.target_pct = target_pct
        self.brokerage_per_trade = brokerage_per_trade
        self.slippage = slippage
        self.results_path = results_path
        self.verbose = verbose

        # internal state
        self.cash = float(initial_capital)
        # holdings: symbol -> list[TradeLot]
        self.holdings: Dict[str, List[TradeLot]] = {}
        # trade log - includes open and closed trades
        self.trade_log_df = pd.DataFrame(
            columns=[
                "tradingSymbol", "tradeState", "entry_date", "direction", "qty", "entry",
                "Target", "StopLoss", "exit_date", "exit", "pnl", "pnl%", "brokerage", "netPnl", "netPnl%"
            ]
        )
        # price data cache: symbol -> DataFrame
        self.price_data: Dict[str, pd.DataFrame] = {}
        # merged calendar (union of available dates)
        self.calendar: pd.DatetimeIndex = pd.DatetimeIndex([])
        # Completed trade lots (to compute metrics)
        self.completed_lots: List[TradeLot] = []

        # Create results directory
        results_dir = os.path.dirname(self.results_path)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)

    def _load_all_data(self):
        """
        Load historical data for all instruments.
        We fetch data back 1 year (approx 365 days) before the backtest start to compute rolling 52-week low.
        """
        # backdate start by 370 days to ensure 52-week calculation safety (account weekends/holidays)
        back_start = (self.start_date - timedelta(days=370))
        if self.verbose:
            print(f"Loading data from {back_start} to {self.end_date} for {len(self.instruments)} instruments...")
        for sym in self.instruments:
            try:
                df = get_historical_data(sym, back_start, self.end_date, interval='day')
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise ValueError(f"Data for {sym} must have DatetimeIndex.")
                # enforce column names
                expected = {"Open", "High", "Low", "Close", "Volume"}
                if not expected.issubset(set(df.columns)):
                    raise ValueError(f"Data for {sym} missing required columns: {expected - set(df.columns)}")
                df = df.sort_index()
                # Make sure index is date only (no time) for daily work
                df.index = pd.to_datetime(df.index).normalize()
                self.price_data[sym] = df
                self.calendar = self.calendar.union(df.loc[self.start_date:self.end_date].index)
            except Exception as e:
                warnings.warn(f"Failed to load {sym}: {e}")
        self.calendar = self.calendar.sort_values()
        if self.calendar.empty:
            raise RuntimeError("No trading days in calendar. Check data availability and date range.")
        if self.verbose:
            print(f"Loaded price data. Trading days in backtest: {len(self.calendar)}")

    def _get_price(self, symbol: str, asof: pd.Timestamp) -> Optional[float]:
        """Return close price for symbol on asof date or None if missing."""
        df = self.price_data.get(symbol)
        if df is None:
            return None
        try:
            return float(df.at[asof, "Close"])
        except Exception:
            return None

    def _compute_52w_low(self, symbol: str, asof: pd.Timestamp) -> Optional[float]:
        """
        Compute trailing 52-week low as of 'asof' (inclusive).
        Use approximately last 252 trading days (common proxy) or last 365 calendar days window.
        """
        df = self.price_data.get(symbol)
        if df is None:
            return None
        # take up to 365 calendar days back from asof
        start_window = asof - pd.Timedelta(days=365)
        window = df.loc[start_window:asof]
        if window.empty:
            return None
        return float(window["Low"].min())

    def _portfolio_value(self, asof: pd.Timestamp) -> float:
        """Return total portfolio value (cash + market value of positions) as of 'asof' using close prices."""
        mv = 0.0
        for sym, lots in self.holdings.items():
            price = self._get_price(sym, asof)
            if price is None:
                # if price missing, compute using last available price before asof
                df = self.price_data.get(sym)
                if df is not None:
                    series = df.loc[:asof]["Close"]
                    if not series.empty:
                        price = float(series.iloc[-1])
            if price is None:
                continue
            total_qty = sum(l.qty for l in lots)
            mv += total_qty * price
        return self.cash + mv

    def _num_shares_from_amount(self, amount: float, price: float) -> int:
        if price <= 0:
            return 0
        return int(math.floor(amount / price))

    def _enter_long(self, symbol: str, asof: pd.Timestamp, allocation: float) -> Optional[TradeLot]:
        """
        Execute buy at close price of asof date.
        Returns TradeLot or None if executed not possible (e.g., no data or no cash).
        We do not deduct brokerage at buy time (we charge at exit per-round-trip).
        """
        price = self._get_price(symbol, asof)
        if price is None or np.isnan(price) or price <= 0:
            return None
        qty = self._num_shares_from_amount(allocation, price)
        if qty <= 0:
            return None
        total_cost = qty * price  # ignoring slippage (assumed zero)
        # check cash
        if total_cost > self.cash + 1e-9:
            # Not enough free cash
            return None
        # perform buy
        self.cash -= total_cost
        lot = TradeLot(symbol=symbol, entry_date=asof, qty=qty, entry_price=price)
        self.holdings.setdefault(symbol, []).append(lot)
        if self.verbose:
            print(f"{asof.date()} BUY {symbol} qty={qty} @ {price:.2f}. Cash left: {self.cash:.2f}")
        return lot

    def _close_position(self, symbol: str, asof: pd.Timestamp, exit_price: Optional[float] = None):
        """
        Close entire position for a symbol at given price (or at close on 'asof' if exit_price None).
        When closing, we compute the profit per lot and charge brokerage_per_trade per lot.
        Each lot becomes a completed TradeLot with exit fields filled and appended to completed_lots and trade_log_df.
        """
        if symbol not in self.holdings:
            return
        lots = self.holdings[symbol]
        if not lots:
            return
        price = exit_price if exit_price is not None else self._get_price(symbol, asof)
        if price is None:
            # cannot close due to missing price
            return
        total_qty = sum(l.qty for l in lots)
        # realize cash: we credit total_qty * price
        gross_proceeds = total_qty * price
        # We'll assign brokerage per lot (per-round-trip) equally: each lot carries brokerage_per_trade
        for lot in lots:
            lot.exit_date = asof
            lot.exit_price = price
            lot.brokerage = self.brokerage_per_trade
            lot.pnl = (lot.exit_price - lot.entry_price) * lot.qty
            lot.net_pnl = lot.pnl - lot.brokerage
            # add to completed lists
            self.completed_lots.append(lot)
            # append to trade log
            row = lot.to_series_on_close()
            # Place the row in chronological order of entry - we will sort at the end
            self.trade_log_df = pd.concat([self.trade_log_df, pd.DataFrame([row])], ignore_index=True)
        # credit cash
        self.cash += gross_proceeds
        if self.verbose:
            total_net = sum(l.net_pnl for l in lots if l.net_pnl is not None)
            print(f"{asof.date()} EXIT {symbol} qty={total_qty} @ {price:.2f}. Realized net pnl (sum): {total_net:.2f}. New cash: {self.cash:.2f}")
        # remove holding
        del self.holdings[symbol]

    def _should_enter_from_top10(self, asof: pd.Timestamp) -> List[str]:
        """
        For the asof day, compute distance to 52w low for all instruments and return top 10 symbols
        sorted by smallest distance ((close - 52wlow) / 52wlow).
        Returns list of symbols (length <= 10).
        """
        distances = []
        for sym in self.instruments:
            try:
                close = self._get_price(sym, asof)
                low52 = self._compute_52w_low(sym, asof)
                if close is None or low52 is None or low52 <= 0:
                    continue
                dist = (close - low52) / low52
                distances.append((sym, dist))
            except Exception:
                continue
        if not distances:
            return []
        df = pd.DataFrame(distances, columns=["symbol", "dist"]).sort_values("dist")
        return df.head(10)["symbol"].tolist()

    def _get_last_buy_price(self, symbol: str) -> Optional[float]:
        """Return last buy price (price of most recent lot) for the symbol if any."""
        lots = self.holdings.get(symbol)
        if not lots:
            return None
        # newest entry date lot
        lots_sorted = sorted(lots, key=lambda l: l.entry_date)
        return lots_sorted[-1].entry_price

    def run_backtest(self):
        """
        Main backtest loop:
        - Load data
        - Iterate dates in chronological order:
            1. Perform exits based on target % (use average buy price)
            2. Determine top 10 nearest to 52w low
            3. Attempt buys from top 10 (skip if already holding)
            4. If all top 10 are already in holdings -> averaging mode: attempt average buy based on threshold
        """
        self._load_all_data()

        # MAIN LOOP
        for asof in self.calendar:
            # 1) Exits first (end-of-day exit using that day's close)
            # For each holding, check average buy price and exit if target reached
            symbols_currently = list(self.holdings.keys())
            for sym in symbols_currently:
                lots = self.holdings.get(sym, [])
                if not lots:
                    continue
                # compute weighted average buy price
                total_qty = sum(l.qty for l in lots)
                if total_qty == 0:
                    continue
                avg_buy = sum(l.qty * l.entry_price for l in lots) / total_qty
                price = self._get_price(sym, asof)
                if price is None:
                    continue
                # check target hit
                if price >= avg_buy * (1 + self.target_pct - 1e-12):
                    # exit entire position
                    self._close_position(sym, asof, exit_price=price)

            # 2) Identify top 10 nearest to 52w low as of asof
            top10 = self._should_enter_from_top10(asof)

            # 3) Start buying from top10 (skip if already in holding)
            bought_today = False
            for sym in top10:
                if sym in self.holdings:
                    continue
                # determine allocation amount based on sizing mode
                if self.sizing_mode == "static":
                    allocation = self.static_allocation
                else:
                    portfolio_val = self._portfolio_value(asof)
                    allocation = portfolio_val / self.dynamic_divisor
                # attempt buy
                lot = self._enter_long(sym, asof, allocation)
                if lot is not None:
                    bought_today = True
                    break  # buy only one per day from top list starting at top
            # 4) If all top10 were already in holding -> averaging mode
            if (top10 and all(sym in self.holdings for sym in top10)) and not bought_today:
                # find holdings that fell more than averaging_threshold from their last buy price
                candidates = []
                for sym, lots in self.holdings.items():
                    last_buy = self._get_last_buy_price(sym)
                    price = self._get_price(sym, asof)
                    if price is None or last_buy is None:
                        continue
                    drop = (price - last_buy) / last_buy
                    if drop <= -self.averaging_threshold:
                        candidates.append((sym, drop))
                if candidates:
                    # pick the one that has fallen the most (most negative drop)
                    candidates.sort(key=lambda x: x[1])  # ascending -> most negative first
                    sym_to_avg = candidates[0][0]
                    # allocate amount for averaging (same as normal per-trade allocation)
                    if self.sizing_mode == "static":
                        allocation = self.static_allocation
                    else:
                        allocation = self._portfolio_value(asof) / self.dynamic_divisor
                    lot = self._enter_long(sym_to_avg, asof, allocation)
                    # If buy unsuccessful (no cash or qty 0) -> no averaging buy
            # continue to next day

        # End of loop
        # After loop, mark remaining open trades in trade_log_df as 'open'
        # Note: we only persist completed trades to CSV as per requirement
        # prepare open trades list
        open_rows = []
        for sym, lots in self.holdings.items():
            for lot in lots:
                open_row = {
                    "tradingSymbol": lot.symbol,
                    "tradeState": "open",
                    "entry_date": lot.entry_date.date(),
                    "direction": "Long",
                    "qty": lot.qty,
                    "entry": lot.entry_price,
                    "Target": None,
                    "StopLoss": None,
                    "exit_date": None,
                    "exit": None,
                    "pnl": None,
                    "pnl%": None,
                    "brokerage": None,
                    "netPnl": None,
                    "netPnl%": None,
                }
                open_rows.append(open_row)
        if open_rows:
            self.trade_log_df = pd.concat([self.trade_log_df, pd.DataFrame(open_rows)], ignore_index=True)
        # reorder by chronological entry date
        self.trade_log_df["entry_date"] = pd.to_datetime(self.trade_log_df["entry_date"])
        self.trade_log_df = self.trade_log_df.sort_values("entry_date").reset_index(drop=True)

        # Save only closed trades to CSV
        closed_df = self.trade_log_df[self.trade_log_df["tradeState"] == "completed"].copy()
        if not closed_df.empty:
            closed_df.to_csv(self.results_path, index=False)
            if self.verbose:
                print(f"Saved closed trades to {self.results_path} ({len(closed_df)} trades).")
        else:
            if self.verbose:
                print("No closed trades to save.")

    # ----------------------------
    # Performance metrics
    # ----------------------------
    def compute_performance(self) -> Dict[str, Any]:
        """
        Compute performance metrics using only completed trades and the equity curve formed by realized net PnL over time.
        Returns a dict of metrics.
        """
        # use completed lots list
        if not self.completed_lots:
            return {
                "Starting Balance": self.initial_capital,
                "Ending Balance": self.initial_capital,
                "Total Net PnL": 0.0,
                "Total Net PnL %": 0.0,
                "CAGR": 0.0,
                "Max Drawdown (%)": 0.0,
                "Max Drawdown (Amount)": 0.0,
                "Number of Trades": 0,
                "Win Rate (%)": 0.0,
                "Average Win (%)": 0.0,
                "Average Loss (%)": 0.0,
                "Average Win Amount": 0.0,
                "Average Loss Amount": 0.0,
                "Risk/Reward Ratio": 0.0,
                "Sharpe Ratio": 0.0,
                "Sortino Ratio": 0.0,
            }
        # Build a DataFrame of closures with date, net pnl
        rows = []
        for lot in self.completed_lots:
            rows.append({"date": lot.exit_date.normalize(), "net_pnl": lot.net_pnl})
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        if df.empty:
            return {}
        # cumulative PnL over time (only at close dates)
        df["cum_net_pnl"] = df["net_pnl"].cumsum()
        # create daily index from first close date to last close date
        first_date = df["date"].min()
        last_date = df["date"].max()
        daily_index = pd.date_range(start=first_date, end=last_date, freq="D")
        cum_series = pd.Series(index=daily_index, dtype=float).fillna(0.0)
        # fill realized pnls on their exit dates
        grouped = df.groupby("date")["net_pnl"].sum()
        running = 0.0
        for d in daily_index:
            if d in grouped.index:
                running += grouped.loc[d]
            cum_series.loc[d] = running
        equity = self.initial_capital + cum_series  # realized equity curve
        # Daily returns (use pct change)
        daily_returns = equity.pct_change().fillna(0.0)

        # Metrics:
        starting_balance = self.initial_capital
        ending_balance = equity.iloc[-1]
        total_net_pnl = ending_balance - starting_balance
        total_net_pnl_pct = (ending_balance / starting_balance - 1.0) * 100.0
        num_days = (last_date - first_date).days if (last_date - first_date).days > 0 else 1
        years = num_days / 365.25
        cagr = (ending_balance / starting_balance) ** (1.0 / years) - 1.0 if years > 0 else 0.0

        # Max drawdown
        roll_max = equity.cummax()
        drawdown = equity - roll_max
        drawdown_pct = drawdown / roll_max.replace(0, np.nan)
        max_drawdown_amt = drawdown.min()
        max_drawdown_pct = (drawdown_pct.min() * -1.0) * 100.0 if not np.isnan(drawdown_pct.min()) else 0.0

        # trade-level stats
        pnl_list = [l.net_pnl for l in self.completed_lots if l.net_pnl is not None]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]
        num_trades = len(pnl_list)
        win_rate = (len(wins) / num_trades * 100.0) if num_trades > 0 else 0.0
        avg_win_pct = (np.mean([ ( ( (l.exit_price / l.entry_price) - 1.0 ) * 100.0 ) for l in self.completed_lots if l.net_pnl and l.net_pnl > 0 ]) ) if wins else 0.0
        avg_loss_pct = (np.mean([ ( ( (l.exit_price / l.entry_price) - 1.0 ) * 100.0 ) for l in self.completed_lots if l.net_pnl and l.net_pnl <= 0 ]) ) if losses else 0.0
        avg_win_amount = np.mean(wins) if wins else 0.0
        avg_loss_amount = np.mean(losses) if losses else 0.0
        # Risk/Reward: mean(win) / abs(mean(loss)) if loss exists
        rr_ratio = (abs(avg_win_amount) / abs(avg_loss_amount)) if avg_loss_amount != 0 else float("inf")

        # Sharpe and Sortino (use daily_returns)
        # assume risk-free rate = 0 for simplicity; you can parametrize.
        mean_daily = daily_returns.mean()
        std_daily = daily_returns.std(ddof=0)
        sharpe = (mean_daily / std_daily * math.sqrt(252)) if std_daily != 0 else 0.0

        # Sortino: downside deviation only
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std(ddof=0)
        sortino = (mean_daily / downside_std * math.sqrt(252)) if downside_std != 0 else 0.0

        metrics = {
            "Starting Balance": round(starting_balance, 2),
            "Ending Balance": round(ending_balance, 2),
            "Total Net PnL": round(total_net_pnl, 2),
            "Total Net PnL %": round(total_net_pnl_pct, 2),
            "CAGR": round(cagr * 100.0, 2),
            "Max Drawdown (%)": round(max_drawdown_pct, 2),
            "Max Drawdown (Amount)": round(max_drawdown_amt, 2),
            "Number of Trades": num_trades,
            "Win Rate (%)": round(win_rate, 2),
            "Average Win (%)": round(avg_win_pct, 2),
            "Average Loss (%)": round(avg_loss_pct, 2),
            "Average Win Amount": round(avg_win_amount, 2),
            "Average Loss Amount": round(avg_loss_amount, 2),
            "Risk/Reward Ratio": round(rr_ratio, 2) if rr_ratio != float("inf") else float("inf"),
            "Sharpe Ratio": round(sharpe, 4),
            "Sortino Ratio": round(sortino, 4),
        }
        return metrics

    def print_performance(self):
        m = self.compute_performance()
        print("\n=== Performance Metrics (completed trades only) ===")
        for k, v in m.items():
            print(f"{k:30s}: {v}")
        print("=================================================\n")

    @property
    def trade_log(self) -> pd.DataFrame:
        """Return trade log DataFrame (sorted chronologically)."""
        return self.trade_log_df.copy()


if __name__ == "__main__":


    pd.set_option("display.max_rows", None, "display.max_columns", None)

    instruments = [
        'NIFTYBEES', 'SILVERBEES', 'GOLDBEES', 'BANKBEES', 'PSUBNKBEES', 'ITBEES',
        'JUNIORBEES', 'MON100', 'HDFCSML250', 'HNGSNGBEES'
    ]

    bt = Backtester(
        start_date=date(2021, 7, 1),
        end_date=date(2025, 6, 30),
        instruments=instruments,
        initial_capital=400000,
        static_allocation=10000,
        dynamic_divisor=20,
        sizing_mode="dynamic",  # or 'dynamic'
        averaging_threshold=0.0314,
        target_pct=0.08,
        brokerage_per_trade=40.0,
        slippage=0.0,
        results_path=os.path.join("Results", "tradebook.csv"),
        verbose=True,
    )

    # Run the backtest (will raise NotImplementedError until you implement get_historical_data)
    try:
        bt.run_backtest()
        bt.print_performance()
        # Save trade log with both open and closed trades (if you want)
        # bt.trade_log.to_csv("Results/tradebook_all.csv", index=False)
    except NotImplementedError as e:
        print(str(e))
