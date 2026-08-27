# McGinley Dynamic Supertrend Strategy - Stock/ETF Edition
# Adapts the strategy for equity trading (long-only) on BANKBEES or any stock
# Fixes: same-bar flip, stop-loss, position sizing, transaction costs, whipsaw filters
#
# LIVE REALISM:
# - Prefer McGinley_Live_Realistic_Backtest.py for vectorbt + walk-forward + mistake memory.
# - This runner now defaults to NEXT-OPEN fills after CLOSE signals (USE_NEXT_OPEN_FILL).
# - Stops checked on bar High/Low with gap-aware exit price when possible.

import pandas as pd
import numpy as np
from numba import jit
import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# =======================================================================================
# CONFIGURATION
# =======================================================================================

csv_file_path = 'BANKBEES/BANKBEES_EQ_60_Min_cleaned.csv'
# Signal confirmed on bar close; order filled at next bar open (live-safe).
USE_NEXT_OPEN_FILL = True

@dataclass
class StrategyParams:
    period: int = 24
    base_multiplier: float = 1.2
    rolling_period: int = 40
    mcginley_length: int = 19
    atr_threshold: float = 0.0  # Minimum ATR% to allow trading (0 = disabled)

@dataclass
class RiskParams:
    atr_sl_multiplier: float = 2.0          # Stop loss at N x ATR from entry
    max_loss_per_trade: float = 2000.0      # Max rupee loss per trade
    time_stop_bars: int = 12                # Force exit after N bars
    max_daily_loss: float = 5000.0          # Stop trading after this daily loss
    max_consecutive_losses: int = 3         # Circuit breaker
    same_bar_cooldown: bool = True          # Prevent re-entry on same bar as exit
    trail_sl_after_bars: int = 0            # Start trailing after N bars (0 = disabled)

@dataclass
class EquityParams:
    quantity: int = 100                     # Shares per trade
    position_sizing: str = "fixed"          # "fixed" or "percent_of_capital"
    capital: float = 100000.0               # Total capital for percent sizing
    use_htf_filter: bool = True             # Use higher-timeframe Supertrend as filter
    htf_file_path: str = 'BANKBEES/BANKBEES_EQ_15_Min_cleaned.csv'
    htf_period: int = 14
    htf_base_multiplier: float = 2.0
    htf_rolling_period: int = 20
    htf_mcginley_length: int = 14
    htf_tf_name: str = "15min"
    use_volatility_sizing: bool = True
    risk_per_trade_pct: float = 0.01        # Risk 1% of capital per trade
    atr_sizing_multiplier: float = 2.0      # SL distance for sizing = N x ATR

@dataclass
class BrokerageParams:
    brokerage_per_order: float = 20.0       # Flat or per-order (use 0 for percentage)
    brokerage_pct: float = 0.0005           # 0.05% if percentage-based
    stt_pct: float = 0.001                  # 0.1% on sell side for delivery
    transaction_pct: float = 0.0000345      # ~0.00345% on both sides
    sebi_pct: float = 1e-7                  # ₹10 per crore
    stamp_duty_pct: float = 0.00015         # 0.015% on buy side
    gst_pct: float = 0.18                   # 18% on (brokerage + sebi + txn)

# =======================================================================================
# NUMBA INDICATOR FUNCTIONS
# =======================================================================================

@jit(nopython=True)
def _calculate_supertrend_numba(ha_high, ha_low, mg, tr, atr, period, multipliers, basic_upper_band, basic_lower_band, first_valid_atr_idx):
    trending_up = np.empty_like(mg)
    trending_down = np.empty_like(mg)
    direction = np.empty_like(mg, dtype=np.int32)
    supertrend = np.empty_like(mg)

    if first_valid_atr_idx is None or first_valid_atr_idx >= len(mg):
        return trending_up, trending_down, direction, supertrend

    trending_up[first_valid_atr_idx] = basic_lower_band[first_valid_atr_idx]
    trending_down[first_valid_atr_idx] = basic_upper_band[first_valid_atr_idx]

    if mg[first_valid_atr_idx] > trending_down[first_valid_atr_idx]:
        direction[first_valid_atr_idx] = 1
    elif mg[first_valid_atr_idx] < trending_up[first_valid_atr_idx]:
        direction[first_valid_atr_idx] = -1
    else:
        direction[first_valid_atr_idx] = 1

    supertrend[first_valid_atr_idx] = trending_up[first_valid_atr_idx] if direction[first_valid_atr_idx] == 1 else trending_down[first_valid_atr_idx]

    for i in range(first_valid_atr_idx + 1, len(mg)):
        prev_mg = mg[i - 1]
        prev_trending_up = trending_up[i - 1]
        prev_trending_down = trending_down[i - 1]
        prev_direction = direction[i - 1]

        current_basic_upper_band = basic_upper_band[i]
        current_basic_lower_band = basic_lower_band[i]
        current_mg = mg[i]

        if prev_mg > prev_trending_up:
            trending_up[i] = max(current_basic_lower_band, prev_trending_up)
        else:
            trending_up[i] = current_basic_lower_band

        if prev_mg < prev_trending_down:
            trending_down[i] = min(current_basic_upper_band, prev_trending_down)
        else:
            trending_down[i] = current_basic_upper_band

        if current_mg > trending_down[i - 1]:
            direction[i] = 1
        elif current_mg < trending_up[i - 1]:
            direction[i] = -1
        else:
            direction[i] = prev_direction

        supertrend[i] = trending_up[i] if direction[i] == 1 else trending_down[i]

    return trending_up, trending_down, direction, supertrend

@jit(nopython=True)
def _calculate_mcginley_numba(source, length):
    mg = np.zeros_like(source)
    if len(source) > 0:
        mg[0] = source[0]
    for i in range(1, len(source)):
        ratio = source[i] / mg[i-1] if mg[i-1] != 0 else 1.0
        mg[i] = mg[i-1] + (source[i] - mg[i-1]) / (length * ratio ** 4)
    return mg

@jit(nopython=True)
def _calculate_heikin_ashi_numba(open_prices, high_prices, low_prices, close_prices):
    ha_close = (open_prices + high_prices + low_prices + close_prices) / 4.0
    ha_open = np.zeros_like(open_prices, dtype=np.float64)
    ha_high = np.zeros_like(high_prices, dtype=np.float64)
    ha_low = np.zeros_like(low_prices, dtype=np.float64)

    if len(open_prices) > 0:
        ha_open[0] = open_prices[0]

    for i in range(1, len(open_prices)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    for i in range(len(open_prices)):
        ha_high[i] = max(high_prices[i], max(ha_open[i], ha_close[i]))
        ha_low[i] = min(low_prices[i], min(ha_open[i], ha_close[i]))

    return ha_open, ha_high, ha_low, ha_close

# =======================================================================================
# INDICATOR CALCULATIONS
# =======================================================================================

def calculate_heikin_ashi(df):
    if not all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        raise ValueError("DataFrame must contain Open, High, Low, Close columns")

    open_prices = df["Open"].to_numpy()
    high_prices = df["High"].to_numpy()
    low_prices = df["Low"].to_numpy()
    close_prices = df["Close"].to_numpy()

    ha_open, ha_high, ha_low, ha_close = _calculate_heikin_ashi_numba(open_prices, high_prices, low_prices, close_prices)

    ha_df = pd.DataFrame({
        "HA_Open": ha_open,
        "HA_High": ha_high,
        "HA_Low": ha_low,
        "HA_Close": ha_close,
    }, index=df.index)

    return ha_df

def calculate_supertrend(df, period=14, base_multiplier=2.0, rolling_period=20, mcginley_length=14, suffix=""):
    original_index = df.index
    df_reset = df.reset_index(drop=True)

    if not all(col in df_reset.columns for col in ["HA_Open", "HA_High", "HA_Low", "HA_Close"]):
        raise ValueError("DataFrame must contain HA_Open, HA_High, HA_Low, HA_Close columns")

    ha_high = df_reset["HA_High"].to_numpy()
    ha_low = df_reset["HA_Low"].to_numpy()
    ha_close = df_reset["HA_Close"].to_numpy()

    mg = _calculate_mcginley_numba(ha_close, mcginley_length)

    tr = np.maximum(ha_high - ha_low, np.abs(ha_high - np.roll(ha_close, 1)))
    tr = np.maximum(tr, np.abs(ha_low - np.roll(ha_close, 1)))
    tr[0] = ha_high[0] - ha_low[0]

    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, len(tr)):
        atr[i] = (tr[i] * alpha) + (atr[i-1] * (1.0 - alpha))

    rolling_atr = np.zeros_like(atr)
    rolling_atr[0] = atr[0]
    beta = 2.0 / (rolling_period + 1.0)
    for i in range(1, len(atr)):
        rolling_atr[i] = (atr[i] * beta) + (rolling_atr[i-1] * (1.0 - beta))

    with np.errstate(divide='ignore', invalid='ignore'):
        multipliers = base_multiplier * (atr / rolling_atr)
    multipliers = np.nan_to_num(multipliers, nan=base_multiplier)
    multipliers = np.clip(multipliers, 0.5, 5.0)

    basic_upper_band = ((ha_high + ha_low) / 2.0) + (multipliers * atr)
    basic_lower_band = ((ha_high + ha_low) / 2.0) - (multipliers * atr)

    first_valid_atr_idx = np.where(~np.isnan(atr))[0][0] if np.any(~np.isnan(atr)) else -1

    trending_up_arr, trending_down_arr, direction_arr, supertrend_arr = _calculate_supertrend_numba(
        ha_high, ha_low, mg, tr, atr, period, multipliers, basic_upper_band, basic_lower_band, first_valid_atr_idx
    )

    st_col_name = f"ST_{period}_{str(base_multiplier).replace('.', '_')}_{rolling_period}_{mcginley_length}{suffix}"
    st_dir_col_name = f"ST_Dir_{period}_{str(base_multiplier).replace('.', '_')}_{rolling_period}_{mcginley_length}{suffix}"
    atr_col_name = f"ATR_{period}"
    mg_col_name = f"McGinley_{mcginley_length}"

    result_df = pd.DataFrame({
        st_col_name: supertrend_arr,
        st_dir_col_name: direction_arr,
        atr_col_name: atr,
        mg_col_name: mg,
    }, index=original_index)

    return result_df, atr

# =======================================================================================
# BROKERAGE & TRANSACTION COSTS (Indian Equity)
# =======================================================================================

def calculate_brokerage(total_sell_value: float, total_buy_value: float,
                        brokerage_per_order: float = 20.0, brokerage_pct: float = 0.0,
                        stt_pct: float = 0.001, transaction_pct: float = 0.0000345,
                        sebi_pct: float = 1e-7, stamp_duty_pct: float = 0.00015,
                        gst_pct: float = 0.18) -> float:
    """
    Calculate total brokerage and taxes for equity delivery trades in India.
    """
    brokerage = 0.0
    if brokerage_per_order > 0:
        brokerage = brokerage_per_order * 2  # buy + sell
    else:
        brokerage = (total_sell_value + total_buy_value) * brokerage_pct

    stt = total_sell_value * stt_pct
    transaction_charges = (total_sell_value + total_buy_value) * 2 * transaction_pct
    sebi_charges = (total_sell_value + total_buy_value) * 2 * sebi_pct
    gst = (brokerage + sebi_charges + transaction_charges) * gst_pct
    stamp_duty = total_buy_value * stamp_duty_pct

    total_cost = brokerage + stt + transaction_charges + sebi_charges + gst + stamp_duty
    return round(total_cost, 2)

# =======================================================================================
# TRADE DATA STRUCTURES
# =======================================================================================

class TradeType:
    LONG = "long"

@dataclass
class Trade:
    trade_type: str
    entry_bar: int
    entry_date: datetime.datetime
    entry_price: float
    quantity: int
    exit_bar: Optional[int] = None
    exit_date: Optional[datetime.datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    profit: float = 0.0
    commission: float = 0.0
    bars_held: int = 0

# =======================================================================================
# RISK MANAGER (Stock Edition - Long Only)
# =======================================================================================

class RiskManager:
    def __init__(self, risk_params: RiskParams, equity_params: EquityParams, brokerage_params: BrokerageParams):
        self.risk = risk_params
        self.equity = equity_params
        self.brokerage = brokerage_params
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trading_halted = False
        self.halt_reason = ""

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.trading_halted = False
        self.halt_reason = ""

    def check_entry_allowed(self, current_date) -> Tuple[bool, str]:
        if self.trading_halted:
            return False, f"Trading halted: {self.halt_reason}"
        return True, ""

    def apply_entry_slippage(self, price: float, direction: int = 1) -> float:
        """Apply slippage when entering. For long: pay slightly more."""
        return price * 1.001  # 0.1% slippage for market buy

    def apply_exit_slippage(self, price: float, direction: int = 1) -> float:
        """Apply slippage when exiting. For long: receive slightly less."""
        return price * 0.999  # 0.1% slippage for market sell

    def check_exit(self, trade: Trade, current_bar: int, current_index: float, atr: float) -> Tuple[bool, str, float]:
        """Check if long position should be exited."""
        if trade is None:
            return False, "", 0.0

        bars_held = current_bar - trade.entry_bar

        # 1. Time-based exit
        if bars_held >= self.risk.time_stop_bars:
            return True, "time_stop", current_index

        # 2. ATR-based stop loss (for long: exit if price drops by N x ATR)
        stop_price = trade.entry_price - (self.risk.atr_sl_multiplier * atr)
        if current_index <= stop_price:
            return True, "atr_stop_loss", current_index

        # 3. Max loss in rupee terms
        unrealized_loss = (trade.entry_price - current_index) * trade.quantity
        if unrealized_loss >= self.risk.max_loss_per_trade:
            return True, "max_loss", current_index

        return False, "", 0.0

    def calculate_trade_pnl(self, trade: Trade, exit_price_raw: float) -> Tuple[float, float]:
        """Calculate final PnL and commission."""
        exit_price = self.apply_exit_slippage(exit_price_raw)
        gross_pnl = (exit_price - trade.entry_price) * trade.quantity

        total_buy_value = trade.entry_price * trade.quantity
        total_sell_value = exit_price * trade.quantity

        commission = calculate_brokerage(
            total_sell_value=total_sell_value,
            total_buy_value=total_buy_value,
            brokerage_per_order=self.brokerage.brokerage_per_order,
            brokerage_pct=self.brokerage.brokerage_pct,
            stt_pct=self.brokerage.stt_pct,
            transaction_pct=self.brokerage.transaction_pct,
            sebi_pct=self.brokerage.sebi_pct,
            stamp_duty_pct=self.brokerage.stamp_duty_pct,
            gst_pct=self.brokerage.gst_pct
        )

        net_pnl = gross_pnl - commission
        return net_pnl, commission

    def update_after_trade(self, pnl: float):
        self.daily_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.daily_pnl <= -self.risk.max_daily_loss:
            self.trading_halted = True
            self.halt_reason = f"Daily loss limit: {self.daily_pnl:.2f}"
        elif self.consecutive_losses >= self.risk.max_consecutive_losses:
            self.trading_halted = True
            self.halt_reason = f"Consecutive losses: {self.consecutive_losses}"

# =======================================================================================
# STREAKS
# =======================================================================================

@jit(nopython=True)
def _calculate_streaks_numba(profits):
    winning_streaks = []
    losing_streaks = []
    current_streak = 0
    current_type = 0

    for profit in profits:
        if profit > 0:
            if current_type == 1:
                current_streak += 1
            else:
                if current_type != 0:
                    winning_streaks.append(current_streak) if current_type == 1 else losing_streaks.append(current_streak)
                current_type = 1
                current_streak = 1
        elif profit < 0:
            if current_type == -1:
                current_streak += 1
            else:
                if current_type != 0:
                    winning_streaks.append(current_streak) if current_type == 1 else losing_streaks.append(current_streak)
                current_type = -1
                current_streak = 1
        else:
            if current_type != 0:
                winning_streaks.append(current_streak) if current_type == 1 else losing_streaks.append(current_streak)
            current_type = 0
            current_streak = 0

    if current_type != 0:
        winning_streaks.append(current_streak) if current_type == 1 else losing_streaks.append(current_streak)

    max_winning_streak = max(winning_streaks) if winning_streaks else 0
    max_losing_streak = max(losing_streaks) if losing_streaks else 0
    return max_winning_streak, max_losing_streak

# =======================================================================================
# METRICS
# =======================================================================================

def calculate_metrics(trades_df: pd.DataFrame) -> Dict:
    metrics = {
        'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0,
        'avg_win': 0, 'avg_loss': 0, 'risk_reward_ratio': np.nan, 'profit_factor': np.nan,
        'expectancy': 0, 'net_profit': 0, 'net_profit_percent': 0, 'max_drawdown': 0,
        'return_drawdown_ratio': np.nan, 'max_winning_streak': 0, 'max_losing_streak': 0,
        'sharpe_ratio': np.nan, 'sortino_ratio': np.nan, 'avg_bars_held': 0
    }

    if trades_df.empty:
        return metrics

    total_profit = trades_df['profit'].sum()
    winning_trades = trades_df[trades_df['profit'] > 0]
    losing_trades = trades_df[trades_df['profit'] < 0]

    num_total_trades = len(trades_df)
    num_winning_trades = len(winning_trades)
    num_losing_trades = len(losing_trades)

    win_rate = (num_winning_trades / num_total_trades) * 100 if num_total_trades > 0 else 0

    avg_win = winning_trades['profit'].mean() if num_winning_trades > 0 else 0
    avg_loss = losing_trades['profit'].mean() if num_losing_trades > 0 else 0

    risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

    gross_profit = winning_trades['profit'].sum() if num_winning_trades > 0 else 0
    gross_loss = abs(losing_trades['profit'].sum()) if num_losing_trades > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    trades_df = trades_df.copy()
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    returns_series = trades_df.set_index('exit_date')['profit'].cumsum()
    net_profit = returns_series.iloc[-1] if not returns_series.empty else 0

    initial_capital = 100000
    if not returns_series.empty:
        equity_curve = initial_capital + returns_series.fillna(0)
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0
    else:
        max_drawdown = 0

    net_profit_percent = (net_profit / initial_capital) * 100 if initial_capital != 0 else 0
    return_drawdown_ratio = (net_profit_percent / abs(max_drawdown)) if max_drawdown != 0 else np.nan

    profits_array = trades_df['profit'].to_numpy()
    max_winning_streak, max_losing_streak = _calculate_streaks_numba(profits_array)

    if not trades_df.empty:
        daily_returns = trades_df.set_index('exit_date')['profit'].resample('D').sum().fillna(0)
        if len(daily_returns) > 1:
            annualized_returns = daily_returns.mean() * 252
            annualized_std_dev = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_returns / annualized_std_dev if annualized_std_dev != 0 else np.nan

            downside_returns = daily_returns[daily_returns < 0]
            downside_std_dev = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
            sortino_ratio = annualized_returns / downside_std_dev if downside_std_dev != 0 else np.nan
        else:
            sharpe_ratio = np.nan
            sortino_ratio = np.nan
    else:
        sharpe_ratio = np.nan
        sortino_ratio = np.nan

    avg_bars_held = trades_df['bars_held'].mean()

    metrics = {
        'total_trades': num_total_trades,
        'winning_trades': num_winning_trades,
        'losing_trades': num_losing_trades,
        'win_rate': round(win_rate, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'risk_reward_ratio': round(risk_reward_ratio, 2) if not np.isnan(risk_reward_ratio) else np.nan,
        'profit_factor': round(profit_factor, 2) if not np.isnan(profit_factor) else np.nan,
        'expectancy': round(expectancy, 2),
        'net_profit': round(net_profit, 2),
        'net_profit_percent': round(net_profit_percent, 2),
        'max_drawdown': round(max_drawdown, 2),
        'return_drawdown_ratio': round(return_drawdown_ratio, 2) if not np.isnan(return_drawdown_ratio) else np.nan,
        'max_winning_streak': max_winning_streak,
        'max_losing_streak': max_losing_streak,
        'sharpe_ratio': round(sharpe_ratio, 2) if not np.isnan(sharpe_ratio) else np.nan,
        'sortino_ratio': round(sortino_ratio, 2) if not np.isnan(sortino_ratio) else np.nan,
        'avg_bars_held': round(avg_bars_held, 2)
    }

    return metrics

# =======================================================================================
# CORE STRATEGY RUNNER (Long-Only Stock Edition)
# =======================================================================================

def compute_htf_direction(htf_df: pd.DataFrame, period: int, base_multiplier: float,
                          rolling_period: int, mcginley_length: int) -> pd.DataFrame:
    """Compute higher-timeframe Supertrend direction for filtering."""
    htf = htf_df.copy()
    if not all(col in htf.columns for col in ["Open", "High", "Low", "Close"]):
        return pd.DataFrame()

    ha_df = calculate_heikin_ashi(htf)
    htf = pd.concat([htf, ha_df], axis=1)
    st_result, _ = calculate_supertrend(htf, period=period, base_multiplier=base_multiplier,
                                        rolling_period=rolling_period, mcginley_length=mcginley_length)

    st_dir_col = f"ST_Dir_{period}_{str(base_multiplier).replace('.', '_')}_{rolling_period}_{mcginley_length}"
    if st_dir_col not in st_result.columns:
        return pd.DataFrame()

    return st_result[[st_dir_col]].rename(columns={st_dir_col: "HTF_Dir"})

# =======================================================================================
# CORE STRATEGY RUNNER (Long-Only Stock Edition)
# =======================================================================================

def run_strategy(df_original: pd.DataFrame,
                 strategy_params: StrategyParams = None,
                 risk_params: RiskParams = None,
                 equity_params: EquityParams = None,
                 brokerage_params: BrokerageParams = None,
                 start_date: Optional[datetime.datetime] = None,
                 end_date: Optional[datetime.datetime] = None,
                 warmup_bars: int = 100) -> Tuple[Dict, pd.DataFrame]:

    if strategy_params is None:
        strategy_params = StrategyParams()
    if risk_params is None:
        risk_params = RiskParams()
    if equity_params is None:
        equity_params = EquityParams()
    if brokerage_params is None:
        brokerage_params = BrokerageParams()

    df = df_original.copy()
    risk_manager = RiskManager(risk_params, equity_params, brokerage_params)

    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    if len(df) < warmup_bars + 50:
        raise ValueError(f"Not enough data. Need at least {warmup_bars + 50} bars, got {len(df)}")

    # 1. Heikin Ashi
    ha_df = calculate_heikin_ashi(df)
    df = pd.concat([df, ha_df], axis=1)

    # 2. Supertrend indicators
    st_result, atr = calculate_supertrend(
        df.copy(),
        period=strategy_params.period,
        base_multiplier=strategy_params.base_multiplier,
        rolling_period=strategy_params.rolling_period,
        mcginley_length=strategy_params.mcginley_length,
        suffix=""
    )

    st_col_name = f"ST_{strategy_params.period}_{str(strategy_params.base_multiplier).replace('.', '_')}_{strategy_params.rolling_period}_{strategy_params.mcginley_length}"
    st_dir_col_name = f"ST_Dir_{strategy_params.period}_{str(strategy_params.base_multiplier).replace('.', '_')}_{strategy_params.rolling_period}_{strategy_params.mcginley_length}"
    atr_col_name = f"ATR_{strategy_params.period}"
    mg_col_name = f"McGinley_{strategy_params.mcginley_length}"

    df = pd.concat([df, st_result[[st_col_name, st_dir_col_name, atr_col_name, mg_col_name]]], axis=1)
    df = df.dropna(subset=[st_dir_col_name, atr_col_name, mg_col_name]).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Insufficient data after warmup")

    # 3. Higher-timeframe filter
    htf_dir_map = {}
    if equity_params.use_htf_filter and equity_params.htf_file_path:
        try:
            htf_df = pd.read_csv(equity_params.htf_file_path)
            htf_df = htf_df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
            htf_df = htf_df.rename(columns={'Date': 'DateTime'})
            htf_df = htf_df.set_index('DateTime')
            htf_df.index = pd.to_datetime(htf_df.index)
            htf_df = htf_df[~htf_df.index.duplicated(keep='first')]
            htf_df = htf_df[htf_df.index.year >= 2021]

            htf_st = compute_htf_direction(
                htf_df,
                period=equity_params.htf_period,
                base_multiplier=equity_params.htf_base_multiplier,
                rolling_period=equity_params.htf_rolling_period,
                mcginley_length=equity_params.htf_mcginley_length
            )

            if not htf_st.empty:
                htf_st.index = pd.to_datetime(htf_st.index)
                df.index = pd.to_datetime(df.index)
                htf_st = htf_st.reindex(df.index, method='ffill')
                htf_dir_map = htf_st['HTF_Dir'].fillna(0).to_dict()
        except Exception as e:
            print(f"Warning: HTF filter failed ({e}). Continuing without filter.")

    # 4. Simulate trades
    trades: List[Trade] = []
    current_trade: Optional[Trade] = None
    same_bar_lockout = False

    current_day = None

    close_prices = df['Close'].to_numpy()
    open_prices = df['Open'].to_numpy() if 'Open' in df.columns else close_prices
    high_prices = df['High'].to_numpy() if 'High' in df.columns else close_prices
    low_prices = df['Low'].to_numpy() if 'Low' in df.columns else close_prices
    st_dir = df[st_dir_col_name].to_numpy()
    atr_vals = df[atr_col_name].to_numpy()
    mg_vals = df[mg_col_name].to_numpy()
    dates = df.index.to_numpy()
    # Live-safe: decide on prior close signal, act on current open/path
    signal_offset = 1 if USE_NEXT_OPEN_FILL else 0

    quantity = equity_params.quantity

    for i in range(1 + signal_offset, len(df)):
        sig_i = i - signal_offset
        today = pd.Timestamp(dates[i]).date() if hasattr(dates[i], 'date') else dates[i]
        if isinstance(today, datetime.datetime):
            today = today.date()

        if current_day != today:
            if current_day is not None:
                risk_manager.reset_daily()
            current_day = today

        allowed, _ = risk_manager.check_entry_allowed(dates[i])
        if not allowed:
            if current_trade is not None:
                pnl, commission = risk_manager.calculate_trade_pnl(current_trade, open_prices[i])
                current_trade.exit_bar = i
                current_trade.exit_date = dates[i]
                current_trade.exit_price = open_prices[i]
                current_trade.exit_reason = "trading_halted"
                current_trade.profit = pnl
                current_trade.commission = commission
                trades.append(current_trade)
                risk_manager.update_after_trade(pnl)
                current_trade = None
            continue

        # Check exit for existing trade (path-aware stop on Low, signal on prior bar)
        if current_trade is not None:
            should_exit, exit_reason, exit_idx = risk_manager.check_exit(
                current_trade, i, low_prices[i], atr_vals[sig_i]
            )
            if should_exit and exit_reason in ("atr_stop_loss", "max_loss"):
                # Gap-aware: if open already through stop, fill open; else stop level
                stop_price = current_trade.entry_price - (risk_params.atr_sl_multiplier * atr_vals[sig_i])
                exit_idx = min(float(open_prices[i]), float(stop_price))

            # Trend flip exit uses signal bar direction (known after prior close)
            if not should_exit and st_dir[sig_i] != 1:
                should_exit = True
                exit_reason = "signal_flip"
                exit_idx = open_prices[i] if USE_NEXT_OPEN_FILL else close_prices[i]

            if should_exit:
                pnl, commission = risk_manager.calculate_trade_pnl(current_trade, exit_idx)
                current_trade.exit_bar = i
                current_trade.exit_date = dates[i]
                current_trade.exit_price = exit_idx
                current_trade.exit_reason = exit_reason
                current_trade.profit = pnl
                current_trade.commission = commission
                current_trade.bars_held = i - current_trade.entry_bar
                trades.append(current_trade)

                risk_manager.update_after_trade(pnl)
                current_trade = None

                if risk_params.same_bar_cooldown:
                    same_bar_lockout = True
                continue

        # Entry logic: only long when prior signal bar trend is up
        if current_trade is None and not same_bar_lockout:
            # HTF filter: block only if HTF is in confirmed downtrend
            if equity_params.use_htf_filter and htf_dir_map and dates[sig_i] in htf_dir_map:
                htf_dir = htf_dir_map[dates[sig_i]]
                if htf_dir == -1:
                    if same_bar_lockout:
                        same_bar_lockout = False
                    continue

            # ATR volatility filter (from signal bar)
            atr_pct = (atr_vals[sig_i] / close_prices[sig_i]) if close_prices[sig_i] > 0 else 0
            if strategy_params.atr_threshold > 0 and atr_pct < strategy_params.atr_threshold:
                if same_bar_lockout:
                    same_bar_lockout = False
                continue

            # Trend strength filter on signal bar
            if sig_i > 2:
                mg_slope = mg_vals[sig_i] - mg_vals[sig_i - 1]
                if st_dir[sig_i] == 1 and mg_slope <= 0:
                    if same_bar_lockout:
                        same_bar_lockout = False
                    continue
                if st_dir[sig_i] == -1:
                    if same_bar_lockout:
                        same_bar_lockout = False
                    continue  # Long-only: skip downtrend

            if st_dir[sig_i] == 1:
                # Volatility-based position sizing from signal-bar ATR
                if equity_params.use_volatility_sizing and atr_vals[sig_i] > 0:
                    sl_distance = risk_params.atr_sl_multiplier * atr_vals[sig_i]
                    risk_amount = equity_params.capital * equity_params.risk_per_trade_pct
                    raw_qty = int(risk_amount / sl_distance)
                    # Round down to nearest 10, enforce min/max
                    raw_qty = max(1, (raw_qty // 10) * 10)
                    raw_qty = min(raw_qty, equity_params.quantity)
                    entry_qty = raw_qty
                else:
                    entry_qty = equity_params.quantity

                fill_px = open_prices[i] if USE_NEXT_OPEN_FILL else close_prices[i]
                entry_price = risk_manager.apply_entry_slippage(fill_px)
                trade = Trade(
                    trade_type=TradeType.LONG,
                    entry_bar=i,
                    entry_date=dates[i],
                    entry_price=entry_price,
                    quantity=entry_qty
                )
                current_trade = trade

        if same_bar_lockout:
            same_bar_lockout = False

    # Close remaining trade
    if current_trade is not None:
        last_idx = len(df) - 1
        pnl, commission = risk_manager.calculate_trade_pnl(current_trade, close_prices[-1])
        current_trade.exit_bar = last_idx
        current_trade.exit_date = dates[-1]
        current_trade.exit_price = close_prices[-1]
        current_trade.exit_reason = "end_of_data"
        current_trade.profit = pnl
        current_trade.commission = commission
        current_trade.bars_held = last_idx - current_trade.entry_bar
        trades.append(current_trade)

    # Build output
    trades_data = []
    for t in trades:
        trades_data.append({
            'trade_type': t.trade_type,
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_price': round(t.entry_price, 2),
            'exit_price': round(t.exit_price, 2) if t.exit_price else 0,
            'quantity': t.quantity,
            'bars_held': t.bars_held,
            'exit_reason': t.exit_reason,
            'profit': round(t.profit, 2),
            'commission': round(t.commission, 2),
            'gross_profit': round(t.profit + t.commission, 2),
        })
    trades_df = pd.DataFrame(trades_data)

    metrics = calculate_metrics(trades_df)
    return metrics, trades_df

# =======================================================================================
# WALK-FORWARD VALIDATION
# =======================================================================================

def walk_forward_analysis(df_original: pd.DataFrame, strategy_params: StrategyParams,
                          risk_params: RiskParams, equity_params: EquityParams,
                          brokerage_params: BrokerageParams,
                          train_months: int = 6, test_months: int = 1) -> pd.DataFrame:
    """Walk-forward analysis with rolling train window."""
    if len(df_original) == 0:
        return pd.DataFrame()

    df_original = df_original.sort_index()
    start_date = df_original.index.min()
    end_date = df_original.index.max()

    results = []
    current_train_start = start_date

    while True:
        train_end = current_train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end_date:
            break

        train_df = df_original.loc[current_train_start:train_end]
        test_df = df_original.loc[test_start:test_end]

        if len(train_df) < 100 or len(test_df) < 20:
            current_train_start = test_start
            continue

        train_metrics, _ = run_strategy(train_df, strategy_params, risk_params, equity_params, brokerage_params)
        test_metrics, _ = run_strategy(test_df, strategy_params, risk_params, equity_params, brokerage_params)

        results.append({
            'train_start': current_train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_win_rate': train_metrics.get('win_rate', 0),
            'train_net_profit': train_metrics.get('net_profit', 0),
            'train_profit_factor': train_metrics.get('profit_factor', np.nan),
            'test_win_rate': test_metrics.get('win_rate', 0),
            'test_net_profit': test_metrics.get('net_profit', 0),
            'test_profit_factor': test_metrics.get('profit_factor', np.nan),
            'test_max_drawdown': test_metrics.get('max_drawdown', 0),
        })

        current_train_start = test_start

    return pd.DataFrame(results)

# =======================================================================================
# MAIN
# =======================================================================================

if __name__ == '__main__':
    try:
        df_original = pd.read_csv(csv_file_path)
        df_original = df_original.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
        df_original = df_original.rename(columns={'Date': 'DateTime'})
        df_original = df_original.set_index('DateTime')
        df_original.index = pd.to_datetime(df_original.index)
        df_original = df_original[~df_original.index.duplicated(keep='first')]
        df_original = df_original[df_original.index.year >= 2021]

        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df_original.columns:
                df_original[col] = df_original[col].round(2)

        print(f"Data loaded: {len(df_original)} rows")

        strategy_params = StrategyParams(
            period=24,
            base_multiplier=1.2,
            rolling_period=40,
            mcginley_length=19,
            atr_threshold=0.0
        )
        risk_params = RiskParams(
            atr_sl_multiplier=2.0,
            max_loss_per_trade=2000.0,
            time_stop_bars=12,
            max_daily_loss=5000.0,
            max_consecutive_losses=3,
            same_bar_cooldown=True
        )
        equity_params = EquityParams(
            quantity=100,
            position_sizing="volatility",
            capital=100000.0,
            use_htf_filter=False,
            htf_file_path='BANKBEES/BANKBEES_EQ_15_Min_cleaned.csv',
            htf_period=14,
            htf_base_multiplier=2.0,
            htf_rolling_period=20,
            htf_mcginley_length=14,
            htf_tf_name="15min",
            use_volatility_sizing=True,
            risk_per_trade_pct=0.01,
            atr_sizing_multiplier=2.0
        )
        brokerage_params = BrokerageParams(
            brokerage_per_order=20.0,
            stt_pct=0.001,
            transaction_pct=0.0000345,
            sebi_pct=1e-7,
            stamp_duty_pct=0.00015,
            gst_pct=0.18
        )

        metrics, trades_df = run_strategy(
            df_original, strategy_params, risk_params, equity_params, brokerage_params
        )

        print("\n" + "="*50)
        print("STRATEGY RESULTS")
        print("="*50)
        print(f"Parameters: {strategy_params}")
        print(f"Risk: {risk_params}")
        print(f"Equity: {equity_params}")
        print(f"Brokerage: {brokerage_params}")
        print("\n--- Metrics ---")
        for key, value in metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        trades_csv_path = 'mcginley_dynamic_supertrend_trades_stock.csv'
        trades_df.to_csv(trades_csv_path, index=False)
        print(f"\nTrades saved to '{trades_csv_path}'")

        print("\nStrategy execution completed successfully!")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
