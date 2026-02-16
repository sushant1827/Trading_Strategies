"""
MidcapShop_backtester.py

MidcapShop backtester (production-quality, modular, parameterized).

Requirements implemented:
 - Mandatory position_sizing_mode: 'static' | 'dynamic' | 'divisor'
 - Separate sizing parameters for fresh and averaging trades
 - Divisor mode uses portfolio_value / divisor -> position size (cash amount)
 - Position size converted to whole shares using floor(amount / close_price)
 - Ensure available cash supports the buy (qty * price + brokerage)
 - Each buy preserved as a separate Lot for audit trail
 - Exits triggered when day's HIGH >= target_price; filled at target_price
 - Exits executed first, then entries
 - Brokerage per order (flat)
 - XIRR computed from dated cashflows (Newton-Raphson)
 - Performance metrics and tradebook saved to Results/tradebook.csv (only closed trades)
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import warnings
import yfinance as yf
import time
import copy

# ---------------------------
# Placeholder: implement this
# ---------------------------
def get_historical_data(symbol: str, start_date: date, end_date: date, interval: str = "1d") -> pd.DataFrame:
    """
    Placeholder function — replace with your data loader.
    Must return a pd.DataFrame indexed by pd.DatetimeIndex (named 'Date' optionally)
    with columns: ['Open','High','Low','Close','Volume'], sorted ascending.
    """
    df = yf.download(symbol+".NS", start=start_date - timedelta(days=150), end=end_date, auto_adjust=True,
                     interval="1d", progress=False, multi_level_index=None,
                     rounding=True, threads=True)
    time.sleep(0.25)
    if df.empty:
        raise ValueError(f"No data found for {symbol} between {start_date} and {end_date}")

    # Reset index to keep Date as a column
    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.set_index("Date", inplace=True)
    return df


# ---------------------------
# Data classes
# ---------------------------
@dataclass
class Lot:
    qty: int
    price: float
    date: pd.Timestamp
    buy_brokerage: float

@dataclass
class Position:
    symbol: str
    lots: List[Lot] = field(default_factory=list)

    def total_qty(self) -> int:
        return sum(l.qty for l in self.lots)

    def avg_price(self) -> Optional[float]:
        q = self.total_qty()
        return (sum(l.qty * l.price for l in self.lots) / q) if q > 0 else None

    def last_buy_price(self) -> Optional[float]:
        return self.lots[-1].price if self.lots else None

    def total_buy_brokerage(self) -> float:
        return sum(l.buy_brokerage for l in self.lots)


# ---------------------------
# Utility: XIRR function (Newton-Raphson)
# ---------------------------
def xirr(cashflows: List[Tuple[date, float]], guess: float = 0.10) -> Optional[float]:
    """
    Compute XIRR given cashflows: list of (date: datetime.date, amount: float).
    Amounts: negative for outflows (buys), positive for inflows (sells).
    Returns annualized rate as decimal (0.12 = 12%) or None if cannot compute.
    """
    if not cashflows:
        return None
    # Need at least one positive and one negative flow
    amounts = [amt for _, amt in cashflows]
    if not any(a < 0 for a in amounts) or not any(a > 0 for a in amounts):
        return None

    cfs = sorted(cashflows, key=lambda x: x[0])
    start_date = cfs[0][0]
    times = np.array([(cf[0] - start_date).days / 365.0 for cf in cfs], dtype=float)
    amounts_arr = np.array([cf[1] for cf in cfs], dtype=float)

    rate = guess
    for _ in range(200):
        denom = (1.0 + rate) ** times
        if np.any(denom == 0) or np.any(np.isinf(denom)) or np.any(np.isnan(denom)):
            return None
        npv = np.sum(amounts_arr / denom)
        d_npv = np.sum(-times * amounts_arr / denom / (1.0 + rate))
        if abs(d_npv) < 1e-12:
            break
        new_rate = rate - npv / d_npv
        if not np.isfinite(new_rate):
            return None
        if abs(new_rate - rate) < 1e-10:
            return new_rate
        rate = new_rate
    return float(rate)


# ---------------------------
# Backtester
# ---------------------------
class MidcapShopBacktester:
    def __init__(
        self,
        instruments: List[str],
        start_date: date,
        end_date: date,
        position_sizing_mode: str,  # required: 'static'|'dynamic'|'divisor'
        # Static sizing (INR amounts)
        fresh_static_amt: float = 10000.0,
        avg_static_amt: float = 5000.0,
        # Dynamic sizing (fractions of available cash, e.g., 0.05 = 5% of available cash)
        fresh_cash_pct: float = 0.025,
        avg_cash_pct: float = 0.0125,
        # Divisor sizing (portfolio_value / divisor)
        fresh_trade_divisor: Optional[float] = None,
        avg_trade_divisor: Optional[float] = None,
        # Strategy params
        initial_capital: float = 400000.0,
        target_pct: float = 0.05,        # 5% target
        avg_trigger_pct: float = 0.03,   # 3% drop from last buy triggers averaging
        brokerage_per_order: float = 40.0,
        dma_window: int = 20,
        max_avg: int = 3,
    ):
        if position_sizing_mode not in ("static", "dynamic", "divisor"):
            raise ValueError("position_sizing_mode must be 'static', 'dynamic', or 'divisor'.")
        if position_sizing_mode == "divisor":
            if fresh_trade_divisor is None or fresh_trade_divisor <= 0:
                raise ValueError("fresh_trade_divisor must be provided and > 0 for divisor mode.")
            if avg_trade_divisor is None or avg_trade_divisor <= 0:
                raise ValueError("avg_trade_divisor must be provided and > 0 for divisor mode.")

        # config
        self.instruments = instruments
        self.start_date = start_date
        self.end_date = end_date
        self.position_sizing_mode = position_sizing_mode

        # sizing params
        self.fresh_static_amt = float(fresh_static_amt)
        self.avg_static_amt = float(avg_static_amt)
        self.fresh_cash_pct = float(fresh_cash_pct)
        self.avg_cash_pct = float(avg_cash_pct)
        self.fresh_trade_divisor = float(fresh_trade_divisor) if fresh_trade_divisor is not None else None
        self.avg_trade_divisor = float(avg_trade_divisor) if avg_trade_divisor is not None else None

        # strategy params
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.target_pct = float(target_pct)
        self.avg_trigger_pct = float(avg_trigger_pct)
        self.brokerage_per_order = float(brokerage_per_order)
        self.dma_window = int(dma_window)
        self.max_avg = int(max_avg)

        # internal state
        self.positions: Dict[str, Position] = {}  # symbol -> Position (open lots)
        self._completed_trades: List[Dict] = []   # rows for closed lots
        self.realized_pnl_by_date: Dict[pd.Timestamp, float] = {}  # for equity curve
        self.data: Dict[str, pd.DataFrame] = {}   # historical data cache
        self._fresh_buy = False  # Indicator to check if a fresh buy was made that day

        # cashflow ledger for XIRR: list of (date, amount)
        # we'll record entry outflows and exit inflows per lot
        self.cashflow_ledger: List[Tuple[date, float]] = []

    # ---------------------------
    # Data loading & preprocessing
    # ---------------------------
    def load_all_data(self):
        for sym in self.instruments:
            try:
                df = get_historical_data(sym, self.start_date - timedelta(days=30), self.end_date, interval="day")
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise ValueError(f"Data for {sym} must have a DatetimeIndex.")
                required = {"Open", "High", "Low", "Close", "Volume"}
                if not required.issubset(set(df.columns)):
                    raise ValueError(f"{sym} data missing required columns {required}.")
                df = df.sort_index()
                self.data[sym] = df.copy()
            except Exception as e:
                warnings.warn(f"[Warning] Could not load {sym}: {e}. Symbol will be skipped.")
        if not self.data:
            raise RuntimeError("No instrument data loaded; aborting backtest.")

    @staticmethod
    def compute_moving_average(df: pd.DataFrame, window: int, column: str = "Close") -> pd.Series:
        return df[column].rolling(window=window, min_periods=window).mean()

    def _get_latest_close(self, symbol: str, dt: pd.Timestamp) -> Optional[float]:
        """Return latest Close on or before dt for valuation; None if none available."""
        df = self.data.get(symbol)
        if df is None:
            return None
        subset = df.loc[df.index <= dt]
        if subset.empty:
            return None
        return float(subset.iloc[-1]["Close"])

    def portfolio_value(self, dt: pd.Timestamp) -> float:
        """Compute portfolio value = cash + market value of open lots using latest close <= dt."""
        total = float(self.cash)
        for sym, pos in self.positions.items():
            price = self._get_latest_close(sym, dt)
            if price is None:
                # conservative: skip valuation if no price available
                continue
            total += pos.total_qty() * price
        return float(total)

    # ---------------------------
    # Sizing helpers
    # ---------------------------
    def _alloc_amount_for_trade(self, trade_kind: str, dt: pd.Timestamp) -> float:
        """
        Compute the cash allocation amount (INR) for a trade_kind ('fresh'|'avg') on date dt
        according to current position_sizing_mode.
        Note: this returns a **cash amount**, later converted to qty by floor(amount / price).
        """
        if self.position_sizing_mode == "static":
            return float(self.fresh_static_amt) if trade_kind == "fresh" else float(self.avg_static_amt)
        elif self.position_sizing_mode == "dynamic":
            pct = float(self.fresh_cash_pct) if trade_kind == "fresh" else float(self.avg_cash_pct)
            # pct expected as fraction (0.05 = 5% of available cash)
            return float(self.cash) * float(pct)
        else:  # divisor
            divisor = self.fresh_trade_divisor if trade_kind == "fresh" else self.avg_trade_divisor
            port_val = self.portfolio_value(dt)
            return float(port_val) / float(divisor)

    def _qty_from_amount_and_price(self, amount: float, price: float) -> int:
        """Return whole-share qty purchasable from amount at price (floor)."""
        if amount <= 0 or price <= 0:
            return 0
        return math.floor(amount / price)

    def _determine_qty_for_buy(self, trade_kind: str, price: float, dt: pd.Timestamp) -> int:
        """
        Given trade_kind ('fresh'|'avg'), price and date, determine final qty:
         - compute allocation amount according to sizing mode
         - convert to qty = floor(amount / price)
         - ensure cash supports qty (qty*price + brokerage <= cash)
         - final qty = min(qty_by_alloc, qty_affordable_by_cash)
        """
        alloc_amount = self._alloc_amount_for_trade(trade_kind, dt)

        # In divisor mode we must also ensure available cash supports that amount; we'll handle via final check
        qty_by_alloc = self._qty_from_amount_and_price(alloc_amount, price)
        if qty_by_alloc <= 0:
            return 0

        if self.cash <= self.brokerage_per_order:
            return 0
        max_qty_by_cash = math.floor((self.cash - self.brokerage_per_order) / price)
        if max_qty_by_cash <= 0:
            return 0

        final_qty = int(min(qty_by_alloc, max_qty_by_cash))
        return final_qty

    # ---------------------------
    # Core backtest loop
    # ---------------------------
    def run_backtest(self) -> pd.DataFrame:
        self.load_all_data()

        # unified date list from data
        all_dates = sorted({d for df in self.data.values() for d in df.index})
        all_dates = [d for d in all_dates if (d.date() >= self.start_date and d.date() <= self.end_date)]
        if not all_dates:
            raise RuntimeError("No trading dates in the data for the requested period.")

        # precompute DMA and pct_below_20dma
        for sym, df in self.data.items():
            df["20DMA"] = self.compute_moving_average(df, self.dma_window, "Close")
            df["pct_below_20dma"] = np.where(
                (df["20DMA"].notna()) & (df["Close"] < df["20DMA"]),
                (df["20DMA"] - df["Close"]) / df["20DMA"],
                0.0,
            )
            self.data[sym] = df

        # DAILY loop
        for current_dt in all_dates:

            self._fresh_buy = False
            # 1) Exits first (intraday high >= target). For each position check avg_price target.
            self._process_exits_for_date(current_dt)

            # 2) Identify top5 most below 20DMA (descending pct_below_20dma)
            top5 = self._get_top5_below_20dma(current_dt)

            # 3) Entries from top5: buy first not already held
            self._process_entries_for_top5(current_dt, top5)

            # 4) Averaging mode if all top5 are already held
            if not self._fresh_buy:
                if top5 and all(sym in self.positions and self.positions[sym].total_qty() > 0 for sym in top5):
                    self._process_averaging_mode(current_dt)

        # After loop: build tradebook of closed trades and save CSV
        tradebook_df = pd.DataFrame(self._completed_trades)
        if not tradebook_df.empty:
            tradebook_df.sort_values(by="entry_date", inplace=True)
            save_path = os.path.join("Results", FILENAME)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            tradebook_df.to_csv(save_path, index=False)
            print(f"Saved closed tradebook to {save_path}")
        else:
            print("No closed trades to save.")

        # compute metrics and print
        self._compute_and_print_performance_metrics()

        return tradebook_df

    # ---------------------------
    # Exit processing
    # ---------------------------
    def _process_exits_for_date(self, dt: pd.Timestamp):
        symbols_to_exit: List[Tuple[str, float, pd.Timestamp]] = []
        for sym, pos in list(self.positions.items()):
            if pos.total_qty() == 0:
                continue
            df = self.data.get(sym)
            if df is None or dt not in df.index:
                continue
            row = df.loc[dt]

            # Use Logic below if exit is to be buy price of each individual holding
            for lot in pos.lots:
                target_price = lot.price * (1.0 + self.target_pct)
                # trigger if intraday high reached target
                if row["High"] >= target_price:
                    symbols_to_exit.append((sym, target_price, lot.date))

            # Use Logic below if exit is to be based on avg buy price of all holdings
            # avg_price = pos.avg_price()
            # if avg_price is None or math.isnan(avg_price):
            #     continue
            # target_price = avg_price * (1.0 + self.target_pct)
            # # trigger if intraday high reached target
            # if row["High"] >= target_price:
            #     symbols_to_exit.append((sym, target_price, dt))

        for sym, exit_price, entry_dt in symbols_to_exit:
            self._execute_exit(sym, exit_price, entry_dt, dt)

    def _execute_exit(self, symbol: str, exit_price: float, entry_dt: pd.Timestamp, dt: pd.Timestamp):
        """
        Exit the entire position for 'symbol' at exit_price.
        Produce one completed trade row per Lot (audit trail).
        Sell brokerage (single sell order) is allocated proportionally across lot values at exit.
        Update cash, realized pnl ledger, and cashflow_ledger for XIRR.
        """
        pos = self.positions.get(symbol)
        exit_position = copy.deepcopy(pos)
        exit_position.lots = [lot for lot in exit_position.lots if lot.date == entry_dt]

        if not exit_position or exit_position.total_qty() == 0:
            return

        total_qty = exit_position.total_qty()
        total_buy_cost = sum(l.qty * l.price for l in exit_position.lots)
        total_buy_brokerage = pos.total_buy_brokerage()

        total_proceeds = total_qty * exit_price
        sell_brokerage = self.brokerage_per_order
        # allocate sell brokerage proportionally to lot value at exit
        lot_values = [l.qty * exit_price for l in exit_position.lots]
        total_value = sum(lot_values) if lot_values else 0.0
        proportions = [lv / total_value if total_value > 0 else (1.0 / len(lot_values)) for lv in lot_values]

        # cash update: add proceeds minus sell brokerage
        self.cash += total_proceeds - sell_brokerage

        # realized pnl date key
        date_key = dt.normalize()
        self.realized_pnl_by_date.setdefault(date_key, 0.0)

        # For each lot create a completed trade row and cashflow entries
        for lot, prop in zip(exit_position.lots, proportions):
            lot_qty = lot.qty
            entry_price = lot.price
            entry_date = lot.date.date()
            lot_buy_brokerage = lot.buy_brokerage
            lot_sell_brokerage = sell_brokerage * prop
            gross_pnl = lot_qty * (exit_price - entry_price)
            lot_total_brokerage = lot_buy_brokerage + lot_sell_brokerage
            net_pnl = gross_pnl - lot_total_brokerage
            pnl_pct = ((exit_price - entry_price) / entry_price * 100.0) if entry_price > 0 else float("nan")
            net_pnl_pct = (net_pnl / (entry_price * lot_qty) * 100.0) if (entry_price > 0 and lot_qty > 0) else float("nan")

            trade_row = {
                "tradingSymbol": symbol,
                "tradeState": "completed",
                "entry_date": entry_date,
                "direction": "Long",
                "qty": lot_qty,
                "entry": round(entry_price, 6),
                "Target": round(entry_price * (1.0 + self.target_pct), 6),
                "StopLoss": float("nan"),
                "exit_date": dt.date(),
                "exit": round(exit_price, 6),
                "pnl": round(gross_pnl, 6),
                "pnl%": round(pnl_pct, 4),
                "brokerage": round(lot_total_brokerage, 6),
                "netPnl": round(net_pnl, 6),
                "netPnl%": round(net_pnl_pct, 4),
            }
            self._completed_trades.append(trade_row)
            # update realized pnl timeline
            self.realized_pnl_by_date[date_key] += net_pnl

            # Record cashflows for XIRR:
            # Entry outflow: -(qty * entry + buy_brokerage)
            # Exit inflow: +(qty * exit - allocated_sell_brokerage)
            self.cashflow_ledger.append((entry_date, -(lot_qty * entry_price + lot_buy_brokerage)))
            self.cashflow_ledger.append((dt.date(), (lot_qty * exit_price - lot_sell_brokerage)))

        # remove position
        # del self.positions[symbol]
        pos.lots = [lot for lot in pos.lots if lot.date != entry_dt]
        if not pos.lots:
            del self.positions[symbol]

    # ---------------------------
    # Entry processing
    # ---------------------------
    def _get_top5_below_20dma(self, dt: pd.Timestamp) -> List[str]:
        candidates: List[Tuple[str, float]] = []
        for sym, df in self.data.items():
            if dt not in df.index:
                continue
            pct = df.loc[dt, "pct_below_20dma"]
            if not pd.isna(pct) and pct > 0:
                candidates.append((sym, float(pct)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in candidates[:5]]

    def _process_entries_for_top5(self, dt: pd.Timestamp, top5: List[str]):
        if not top5:
            return
        for sym in top5:
            pos = self.positions.get(sym)
            if pos and pos.total_qty() > 0:
                continue

            df = self.data.get(sym)
            if df is None or dt not in df.index:
                continue
            close_price = float(df.loc[dt, "Close"])
            # Determine qty depending on sizing mode
            qty = self._determine_qty_for_buy("fresh", close_price, dt)

            # In divisor mode: ensure available cash supports the computed position size (per your rule)
            if self.position_sizing_mode == "divisor":
                # alloc_amount computed inside _alloc_amount_for_trade; we verify that cash >= alloc_amount
                alloc_amount = self._alloc_amount_for_trade("fresh", dt)
                # If alloc_amount > cash, reduce qty via _determine_qty_for_buy already, so only need to check qty > 0
                # but keep an explicit check to match your specification: skip trade if cash < alloc_amount (strict)
                if self.cash < alloc_amount and qty > 0:
                    # Instead of forcibly allowing smaller buy, we will allow buying up to cash (we already capped qty by cash).
                    # The user's requirement: "ensure available cash supports that buy; buy until cash on hand".
                    # So it's already handled by max_qty_by_cash cap.
                    pass

            if qty <= 0:
                continue
            total_cost = qty * close_price + self.brokerage_per_order
            if total_cost > self.cash + 1e-9:
                # safety check (qty should have been capped)
                continue

            # Execute buy: create lot and reduce cash; record buy cashflow
            self.cash -= total_cost
            lot = Lot(qty=qty, price=close_price, date=dt, buy_brokerage=self.brokerage_per_order)
            self.positions[sym] = Position(symbol=sym, lots=[lot])

            # Record buy cashflow for XIRR (entry_date, negative amount including buy brokerage)
            self.cashflow_ledger.append((dt.date(), -(qty * close_price + self.brokerage_per_order)))

            # After first successful buy from top5, stop scanning
            self._fresh_buy = True
            break

    def _process_averaging_mode(self, dt: pd.Timestamp):
        """
        If all top5 are held, scan existing holdings and buy additional lots for those that have
        fallen more than avg_trigger_pct from their last buy price.
        """
        for sym, pos in list(self.positions.items()):
            if pos.total_qty() == 0:
                continue
            # Limit no. of positions/averaging
            if len(pos.lots) >= self.max_avg:
                continue
            df = self.data.get(sym)
            if df is None or dt not in df.index:
                continue

            close_price = float(df.loc[dt, "Close"])
            last_buy_price = pos.last_buy_price()
            if last_buy_price is None:
                continue
            pct_drop = (last_buy_price - close_price) / last_buy_price
            if pct_drop > self.avg_trigger_pct:
                qty = self._determine_qty_for_buy("avg", close_price, dt)
                if qty <= 0:
                    continue
                total_cost = qty * close_price + self.brokerage_per_order
                if total_cost > self.cash + 1e-9:
                    continue
                # execute averaging buy (append lot)
                self.cash -= total_cost
                lot = Lot(qty=qty, price=close_price, date=dt, buy_brokerage=self.brokerage_per_order)
                pos.lots.append(lot)
                # record buy cashflow
                self.cashflow_ledger.append((dt.date(), -(qty * close_price + self.brokerage_per_order)))
                # Only one averaging allowed per day, so exit
                break

    # ---------------------------
    # Performance metrics
    # ---------------------------
    def _compute_and_print_performance_metrics(self):
        trades = pd.DataFrame(self._completed_trades)
        starting_balance = float(self.initial_capital)
        print("\n--- MidcapShop Backtest Performance Summary ---")
        if trades.empty:
            print("No completed trades. Only starting balance available.")
            print(f"Starting Balance: {starting_balance:,.2f}")
            print(f"Ending Balance (no closed trades): {starting_balance:,.2f}")
            return

        # Ending balance based solely on closed trades (per requirement)
        total_net_pnl = trades["netPnl"].sum()
        ending_balance = starting_balance + float(total_net_pnl)
        total_net_pnl_pct = (ending_balance - starting_balance) / starting_balance * 100.0

        # CAGR
        days = (pd.Timestamp(self.end_date) - pd.Timestamp(self.start_date)).days
        years = days / 365.25 if days > 0 else 1.0
        cagr = (ending_balance / starting_balance) ** (1.0 / years) - 1.0 if years > 0 else float("nan")

        # Equity curve (realized only) -> daily returns for Sharpe/Sortino and drawdown
        if self.realized_pnl_by_date:
            idx = sorted(self.realized_pnl_by_date.keys())
            cum = starting_balance
            dates = []
            eq = []
            for dt in idx:
                cum += self.realized_pnl_by_date[dt]
                dates.append(dt)
                eq.append(cum)
            equity = pd.Series(data=eq, index=pd.DatetimeIndex(dates)).sort_index()
            full_index = pd.date_range(start=pd.Timestamp(self.start_date), end=pd.Timestamp(self.end_date), freq="D")
            equity_ff = equity.reindex(full_index).ffill().fillna(starting_balance)
            daily_returns = equity_ff.pct_change().fillna(0.0)
            running_max = equity_ff.cummax()
            drawdown = (equity_ff - running_max) / running_max
            max_dd_pct = drawdown.min() * 100.0
            max_dd_amount = (running_max - equity_ff).max()
        else:
            equity_ff = pd.Series(starting_balance, index=pd.date_range(start=self.start_date, end=self.end_date))
            daily_returns = equity_ff.pct_change().fillna(0.0)
            max_dd_pct = 0.0
            max_dd_amount = 0.0

        # trade stats
        num_trades = len(trades)
        wins = trades[trades["netPnl"] > 0]
        losses = trades[trades["netPnl"] <= 0]
        win_rate = (len(wins) / num_trades) * 100.0 if num_trades > 0 else float("nan")
        avg_win_amt = wins["netPnl"].mean() if not wins.empty else 0.0
        avg_loss_amt = losses["netPnl"].mean() if not losses.empty else 0.0
        avg_win_pct = wins["pnl%"].mean() if not wins.empty else 0.0
        avg_loss_pct = losses["pnl%"].mean() if not losses.empty else 0.0
        rr_ratio = (avg_win_pct / abs(avg_loss_pct)) if (avg_loss_pct != 0 and not math.isnan(avg_loss_pct)) else float("inf")

        # Sharpe (annualized), risk-free assumed 0
        mean_daily = daily_returns.mean()
        std_daily = daily_returns.std(ddof=1)
        sharpe = (mean_daily / std_daily) * math.sqrt(252) if std_daily > 0 else float("nan")

        # Sortino (annualized)
        neg_returns = daily_returns[daily_returns < 0]
        if neg_returns.shape[0] > 0:
            downside_std = neg_returns.std(ddof=1)
            sortino = (mean_daily / downside_std) * math.sqrt(252) if downside_std > 0 else float("nan")
        else:
            sortino = float("nan")

        # Build XIRR cashflows aggregated by date (summing flows on same date)
        if self.cashflow_ledger:
            cf_df = pd.DataFrame(self.cashflow_ledger, columns=["date", "amt"])
            cf_df = cf_df.groupby("date", as_index=False).sum().sort_values("date")
            cashflows = [(pd.to_datetime(r["date"]).date(), float(r["amt"])) for _, r in cf_df.iterrows()]
            # Optionally, include final cash at end_date as an inflow to represent remaining portfolio cash
            # But user requested XIRR only from completed trades; still it's common to include final cash.
            # We'll **include leftover cash on end_date** to make XIRR reflect final capital.
            end_cash = float(self.cash)
            if end_cash != 0.0:
                cashflows.append((self.end_date, end_cash))
            cashflows.sort(key=lambda x: x[0])
            xirr_val = xirr(cashflows, guess=0.10)
        else:
            xirr_val = None

        # Print metrics
        print(f"Backtest period: {self.start_date.isoformat()} to {self.end_date.isoformat()}")
        print(f"Starting Balance          : {starting_balance:,.2f}")
        print(f"Ending Balance (realized) : {ending_balance:,.2f}")
        print(f"Total Net PnL            : {total_net_pnl:,.2f}")
        print(f"Total Net PnL (%)        : {total_net_pnl_pct:.2f}%")
        print(f"CAGR                     : {cagr * 100.0:.2f}%")
        print(f"XIRR                     : {xirr_val * 100.0:.2f}%" if xirr_val is not None else "XIRR: n/a")
        print(f"Max Drawdown (%)         : {max_dd_pct:.2f}%")
        print(f"Max Drawdown (Amount)    : {max_dd_amount:,.2f}")
        print(f"Number of Trades (lots)  : {num_trades}")
        print(f"Win Rate (%)             : {win_rate:.2f}%")
        print(f"Average Win (%)          : {avg_win_pct:.4f}%")
        print(f"Average Loss (%)         : {avg_loss_pct:.4f}%")
        print(f"Average Win Amount       : {avg_win_amt:,.2f}")
        print(f"Average Loss Amount      : {avg_loss_amt:,.2f}")
        print(f"Risk/Reward Ratio        : {rr_ratio if rr_ratio != float('inf') else 'Inf (no losing trades)'}")
        print(f"Sharpe Ratio (annualized): {sharpe:.4f}")
        print(f"Sortino Ratio (annualized): {sortino:.4f}")
        print("--- End Performance Summary ---\n")


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    instruments = ['PERSISTENT', 'PRESTIGE', 'AUBANK', 'GODREJPROP', 'COFORGE', 'PHOENIXLTD', 'HDFCAMC', 'OBEROIRLTY',
                   'PAYTM', 'NHPC', 'TIINDIA', 'INDUSTOWER', 'OIL', 'IRCTC', 'SBICARD', 'BHEL', 'MPHASIS',
                   'MUTHOOTFIN', 'DABUR', 'GMRAIRPORT', 'COLPAL', 'SRF', 'HINDPETRO', 'UPL', 'FORTIS', 'SUPREMEIND',
                   'DIXON', 'POLYCAB', 'NMDC', 'BHARATFORG', 'PAGEIND', 'JUBLFOOD', 'FEDERALBNK', 'APLAPOLLO',
                   'CUMMINSIND', 'BSE', 'ASHOKLEY', 'IDFCFIRSTB', 'YESBANK', 'LUPIN', 'MANKIND', 'PIIND', 'SUZLON',
                   'MARICO', 'MFSL', 'HEROMOTOCO', 'AUROPHARMA', 'INDUSINDBK', 'POLICYBZR', 'OFSS']

    start = date(2020, 1, 1)
    end = date(2025, 10, 17)

    FILENAME = 'Scenario mcap50New.csv'

    # Example config: choose the sizing mode explicitly (mandatory)
    backtester = MidcapShopBacktester(
        instruments=instruments,
        start_date=start,
        end_date=end,
        position_sizing_mode="divisor",  # 'static' | 'dynamic' | 'divisor'
        fresh_static_amt=10000.0,
        avg_static_amt=20000.0,
        fresh_cash_pct=0.04,      # 2.5% of available cash if dynamic
        avg_cash_pct=0.06,       # 1.25% of available cash if dynamic
        fresh_trade_divisor=10.0,  # used in divisor mode
        avg_trade_divisor=40.0,    # used in divisor mode
        initial_capital=400000.0,
        target_pct=0.08,
        avg_trigger_pct=0.05,
        brokerage_per_order=40.0,
        dma_window=20,
        max_avg=3
    )

    trades_df = backtester.run_backtest()
