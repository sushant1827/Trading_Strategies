"""
Nadaraya_Watson_Live_Realistic_Backtest.py
==========================================
Live-realistic rebuild of the Nadaraya-Watson family of strategies.

What was wrong in the older scripts (and is fixed here)
------------------------------------------------------
1. Causal NW kernel with a fixed decaying lookback (no future bars, no x0 misuse).
2. Signals confirmed on bar CLOSE; orders filled on NEXT bar OPEN (no same-bar fill).
3. Indicators may use Heikin Ashi, but fills always use real OHLC.
4. Trade list comes from vectorbt (not "every long bar is a trade").
5. Correct bullish/bearish polarity for NW crosses and slope turns.
6. Standard Wilder SuperTrend (one definition only).
7. Costs: Rs 250 round-trip on Rs 1L capital => Rs 125 fixed fee per order.
8. No random Sharpe / streak metrics.
9. Walk-forward: train selects params + mistake filters; test is frozen OOS.
10. Mistake memory: train-only loss patterns -> frozen avoid-rules on test.

Strategies backtested
---------------------
A) NW_Cross          : yhat2 cross yhat1 (lagged bandwidth)
B) NW_SlopeTurn      : yhat1 slope turn (non-smooth Pine-style)
C) NW_ST_Confirm     : SuperTrend flip + NW regime confirmation
D) NW_ST_Confirm_SL  : same as C with prior-bar SuperTrend stop

Instrument: NIFTY50 30-min index as a continuous price proxy (whole units).
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, asdict
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR = r"D:\Sushant\Fyers_AlgoTrade\Fyers_Data"
CSV_PATH = os.path.join(BASE_DIR, "Nifty50_Index", "NIFTY50_INDEX_30_Min.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "PyScripts", "nw_live_realistic_outputs")
MEMORY_PATH = os.path.join(OUTPUT_DIR, "nw_mistake_memory.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BACKTEST_START = pd.Timestamp("2021-01-01")
# Walk-forward: train through end-2023, OOS from 2024 onward (frozen rules)
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
TEST_START = pd.Timestamp("2024-01-01")

INIT_CASH = 100_000.0
ROUND_TRIP_CHARGES = 250.0
FIXED_FEE_PER_ORDER = ROUND_TRIP_CHARGES / 2.0  # 125 per side
BARS_PER_DAY = 13  # 09:15-15:15 inclusive on 30m ≈ 13 bars
NSE_DAYS_PER_YEAR = 252
YEAR_FREQ = pd.Timedelta(days=NSE_DAYS_PER_YEAR)
VBT_FREQ = "30min"

# Compact grids (train only). Keep small so full run stays practical.
NW_H_GRID = [8, 12, 20]
NW_R_GRID = [4.0, 8.0, 12.0]
NW_LAG_GRID = [2]
ST_PERIOD_GRID = [14, 21]
ST_MULT_GRID = [2.0, 2.5, 3.0]
HA_GRID = [False, True]
DIRECTION_MODES = ["long_only", "long_short"]  # long_short = reverse on opposite signal
# SuperTrend combos explode quickly; use a tighter default ST search.
ST_SEARCH_H = [8, 16]
ST_SEARCH_R = [8.0, 12.0]
ST_SEARCH_LAG = [2]
ST_SEARCH_PERIOD = [14, 21]
ST_SEARCH_MULT = [2.0, 3.0]

MIN_TRAIN_TRADES = 20
MIN_TEST_TRADES_REPORT = 1

# Mistake-memory thresholds (fit on train losers only)
LOSS_CLUSTER_MIN_COUNT = 8
LOSS_CLUSTER_MIN_SHARE = 0.22  # rule must cover >=22% of train losses
AVOID_RULE_MAX = 4


# ==============================================================================
# DATA
# ==============================================================================

def load_ohlc(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "DateTime"})
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime").drop_duplicates("DateTime").set_index("DateTime")
    rename = {"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"}
    # normalize case
    colmap = {c: c.title() if c.lower() in ("open", "high", "low", "close", "volume") else c for c in df.columns}
    df = df.rename(columns=colmap)
    df = df.loc[df.index >= BACKTEST_START, ["Open", "High", "Low", "Close"]].astype(float)
    for c in df.columns:
        df[c] = df[c].round(2)
    return df


# ==============================================================================
# INDICATORS (causal)
# ==============================================================================

@njit(cache=True)
def _heikin_ashi_nb(open_, high, low, close):
    n = len(close)
    ha_o = np.empty(n)
    ha_h = np.empty(n)
    ha_l = np.empty(n)
    ha_c = np.empty(n)
    ha_c[0] = (open_[0] + high[0] + low[0] + close[0]) / 4.0
    ha_o[0] = (open_[0] + close[0]) / 2.0
    ha_h[0] = max(high[0], ha_o[0], ha_c[0])
    ha_l[0] = min(low[0], ha_o[0], ha_c[0])
    for i in range(1, n):
        ha_c[i] = (open_[i] + high[i] + low[i] + close[i]) / 4.0
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
        ha_h[i] = max(high[i], ha_o[i], ha_c[i])
        ha_l[i] = min(low[i], ha_o[i], ha_c[i])
    return ha_o, ha_h, ha_l, ha_c


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = _heikin_ashi_nb(
        df["Open"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    )
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=df.index)


@njit(cache=True)
def _nw_rq_kernel_nb(src: np.ndarray, h: float, r: float, max_window: int) -> np.ndarray:
    """
    Causal Nadaraya-Watson Rational Quadratic kernel.
    At bar k, only uses src[k], src[k-1], ... (past and present).
    Weight for lag i: (1 + i^2 / (h^2 * 2 * r)) ** (-r)
    Weights are precomputed once; window capped (distant weights ~0).
    """
    n = len(src)
    yhat = np.empty(n)
    denom = (h * h) * 2.0 * r
    # Precompute kernel weights for lags 0..max_window-1
    wtab = np.empty(max_window)
    for i in range(max_window):
        wtab[i] = (1.0 + (i * i) / denom) ** (-r)

    for k in range(n):
        cur = 0.0
        cum = 0.0
        last = k + 1 if (k + 1) < max_window else max_window
        for i in range(last):
            w = wtab[i]
            cur += src[k - i] * w
            cum += w
        if cum > 0.0:
            yhat[k] = cur / cum
        else:
            yhat[k] = np.nan
    return yhat


# Cache NW curves for repeated (h, r) on the same close series identity during a run.
_NW_CACHE: dict[tuple[int, float, float, int], np.ndarray] = {}


def nw_regression(close: pd.Series, h: float, r: float) -> pd.Series:
    if h <= 0 or r <= 0:
        raise ValueError("h and r must be > 0")
    # ~5*h covers nearly all RQ mass for typical r; much faster than full history.
    max_window = int(max(24, min(len(close), int(h * 5) + 8)))
    key = (id(close.to_numpy()), float(h), float(r), max_window)
    # Use values identity + length for cache (avoid holding huge keys on every call)
    arr_src = close.to_numpy(dtype=np.float64, copy=False)
    cache_key = (arr_src.ctypes.data, arr_src.shape[0], float(h), float(r), max_window)
    cached = _NW_CACHE.get(cache_key)
    if cached is None:
        cached = _nw_rq_kernel_nb(arr_src, float(h), float(r), max_window)
        _NW_CACHE[cache_key] = cached
    return pd.Series(cached, index=close.index, name="yhat")


def clear_nw_cache() -> None:
    _NW_CACHE.clear()


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().to_numpy()


@njit(cache=True)
def _supertrend_nb(high, low, close, atr, multiplier):
    """Standard SuperTrend: ratchet bands + direction from close vs bands."""
    n = len(close)
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    final_upper = np.empty(n)
    final_lower = np.empty(n)
    st = np.empty(n)
    direction = np.empty(n, dtype=np.int64)

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    direction[0] = 1
    st[0] = final_lower[0]

    for i in range(1, n):
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if direction[i - 1] == 1:
            if close[i] < final_lower[i]:
                direction[i] = -1
                st[i] = final_upper[i]
            else:
                direction[i] = 1
                st[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                direction[i] = 1
                st[i] = final_lower[i]
            else:
                direction[i] = -1
                st[i] = final_upper[i]
    return st, direction


def supertrend(df: pd.DataFrame, period: int, multiplier: float) -> pd.DataFrame:
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()
    atr = wilder_atr(high, low, close, period)
    n = len(close)
    st = np.full(n, np.nan)
    direction = np.zeros(n, dtype=np.int64)
    valid = np.flatnonzero(np.isfinite(atr))
    if len(valid) > 1:
        start = int(valid[0])
        st_v, d_v = _supertrend_nb(high[start:], low[start:], close[start:], atr[start:], float(multiplier))
        st[start:] = st_v
        direction[start:] = d_v
    return pd.DataFrame({"ST": st, "ST_Dir": direction, "ATR": atr}, index=df.index)


# ==============================================================================
# SIGNAL BUILDERS (event signals on bar close; execution shifts later)
# ==============================================================================

@dataclass
class StrategyParams:
    name: str
    h: float
    r: float
    lag: int = 2
    st_period: int = 14
    st_mult: float = 2.0
    use_ha: bool = False
    direction_mode: str = "long_only"  # long_only | long_short
    use_st_stop: bool = False


def _indicator_frame(raw: pd.DataFrame, use_ha: bool) -> pd.DataFrame:
    """Price series for indicators only. Never used for fill prices."""
    return heikin_ashi(raw) if use_ha else raw[["Open", "High", "Low", "Close"]].copy()


def build_nw_cross_signals(raw: pd.DataFrame, p: StrategyParams) -> dict[str, Any]:
    ind = _indicator_frame(raw, p.use_ha)
    h2 = float(p.h - p.lag)
    if h2 <= 0.5:
        raise ValueError(f"h-lag must be > 0.5 (h={p.h}, lag={p.lag})")
    y1 = nw_regression(ind["Close"], p.h, p.r)
    y2 = nw_regression(ind["Close"], h2, p.r)
    valid = y1.notna() & y2.notna() & y1.shift(1).notna() & y2.shift(1).notna()

    # Bullish: lag line crosses ABOVE slow line (y2 crosses above y1)
    long_entry = valid & y2.gt(y1) & y2.shift(1).le(y1.shift(1))
    long_exit = valid & y2.lt(y1) & y2.shift(1).ge(y1.shift(1))
    short_entry = long_exit.copy()
    short_exit = long_entry.copy()

    atr = wilder_atr(raw["High"].to_numpy(), raw["Low"].to_numpy(), raw["Close"].to_numpy(), 14)
    return {
        "long_entry": long_entry.fillna(False),
        "long_exit": long_exit.fillna(False),
        "short_entry": short_entry.fillna(False),
        "short_exit": short_exit.fillna(False),
        "yhat1": y1,
        "yhat2": y2,
        "stop_level": None,
        "atr": pd.Series(atr, index=raw.index),
        "nw_bull": (y2 > y1).fillna(False),
    }


def build_nw_slope_signals(raw: pd.DataFrame, p: StrategyParams) -> dict[str, Any]:
    """Slope-turn mode: buy when yhat1 turns up, sell when yhat1 turns down."""
    ind = _indicator_frame(raw, p.use_ha)
    y1 = nw_regression(ind["Close"], p.h, p.r)
    # slope: y1 - y1.shift(1); turn up = was down, now up
    d1 = y1.diff()
    d2 = d1.shift(1)
    valid = d1.notna() & d2.notna()
    long_entry = valid & d1.gt(0) & d2.le(0)
    long_exit = valid & d1.lt(0) & d2.ge(0)
    short_entry = long_exit.copy()
    short_exit = long_entry.copy()
    atr = wilder_atr(raw["High"].to_numpy(), raw["Low"].to_numpy(), raw["Close"].to_numpy(), 14)
    return {
        "long_entry": long_entry.fillna(False),
        "long_exit": long_exit.fillna(False),
        "short_entry": short_entry.fillna(False),
        "short_exit": short_exit.fillna(False),
        "yhat1": y1,
        "yhat2": y1,  # unused
        "stop_level": None,
        "atr": pd.Series(atr, index=raw.index),
        "nw_bull": d1.gt(0).fillna(False),
    }


def build_nw_st_signals(raw: pd.DataFrame, p: StrategyParams) -> dict[str, Any]:
    ind = _indicator_frame(raw, p.use_ha)
    h2 = float(p.h - p.lag)
    if h2 <= 0.5:
        raise ValueError(f"h-lag must be > 0.5 (h={p.h}, lag={p.lag})")
    y1 = nw_regression(ind["Close"], p.h, p.r)
    y2 = nw_regression(ind["Close"], h2, p.r)
    st = supertrend(ind, p.st_period, p.st_mult)

    nw_bull = y2 > y1
    nw_bear = y2 < y1
    st_up = st["ST_Dir"] == 1
    st_dn = st["ST_Dir"] == -1
    st_flip_up = st_up & ~st_up.shift(1).fillna(False)
    st_flip_dn = st_dn & ~st_dn.shift(1).fillna(False)
    ready = y1.notna() & y2.notna() & st["ST"].notna()

    long_entry = ready & st_flip_up & nw_bull
    long_exit = ready & (st_flip_dn | nw_bear)
    short_entry = ready & st_flip_dn & nw_bear
    short_exit = ready & (st_flip_up | nw_bull)

    stop_level = None
    if p.use_st_stop:
        # trail with SuperTrend line while in respective direction (on indicator frame)
        stop_level = st["ST"]

    atr = st["ATR"]
    return {
        "long_entry": long_entry.fillna(False),
        "long_exit": long_exit.fillna(False),
        "short_entry": short_entry.fillna(False),
        "short_exit": short_exit.fillna(False),
        "yhat1": y1,
        "yhat2": y2,
        "stop_level": stop_level,
        "st": st["ST"],
        "st_dir": st["ST_Dir"],
        "atr": atr,
        "nw_bull": nw_bull.fillna(False),
    }


SIGNAL_BUILDERS = {
    "NW_Cross": build_nw_cross_signals,
    "NW_SlopeTurn": build_nw_slope_signals,
    "NW_ST_Confirm": build_nw_st_signals,
    "NW_ST_Confirm_SL": build_nw_st_signals,
}


# ==============================================================================
# BACKTEST ENGINE (next-open fills, costs, optional ST stop)
# ==============================================================================

@njit(cache=True)
def _causal_percentile_rank_nb(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.nan
        start = i - window + 1
        if start < 0:
            start = 0
        # count finite in window
        cnt = 0
        le = 0
        x = arr[i]
        if not np.isfinite(x):
            continue
        for j in range(start, i + 1):
            v = arr[j]
            if np.isfinite(v):
                cnt += 1
                if v <= x:
                    le += 1
        if cnt >= 20:
            out[i] = le / cnt
    return out


def causal_percentile_rank(series: pd.Series, window: int = 100) -> pd.Series:
    """Rolling percentile of the last value vs prior window (causal)."""
    arr = series.to_numpy(dtype=np.float64)
    return pd.Series(_causal_percentile_rank_nb(arr, int(window)), index=series.index)


def apply_avoid_mask(
    entries: pd.Series,
    raw: pd.DataFrame,
    sig: dict[str, Any],
    avoid_rules: list[dict[str, Any]],
) -> pd.Series:
    """Zero-out entries that match frozen avoid rules (no lookahead)."""
    if not avoid_rules:
        return entries
    out = entries.copy()
    idx = raw.index
    hour = idx.hour
    dow = idx.dayofweek
    atr = sig.get("atr")
    atr_s = atr if isinstance(atr, pd.Series) else pd.Series(atr, index=idx)
    atr_rank = causal_percentile_rank(atr_s, window=100)

    for rule in avoid_rules:
        rtype = rule["type"]
        if rtype == "avoid_hour":
            out = out & ~(hour == int(rule["hour"]))
        elif rtype == "avoid_dow":
            out = out & ~(dow == int(rule["dow"]))
        elif rtype == "avoid_hour_dow":
            out = out & ~((hour == int(rule["hour"])) & (dow == int(rule["dow"])))
        elif rtype == "avoid_high_atr":
            thr = float(rule["atr_rank_min"])
            out = out & ~(atr_rank.fillna(0) >= thr)
        elif rtype == "avoid_low_atr":
            thr = float(rule["atr_rank_max"])
            out = out & ~(atr_rank.fillna(1) <= thr)
    return out.fillna(False)


def run_portfolio(
    raw: pd.DataFrame,
    sig: dict[str, Any],
    direction_mode: str = "long_only",
    use_st_stop: bool = False,
    avoid_rules: list[dict[str, Any]] | None = None,
) -> vbt.Portfolio:
    """
    Live rules:
    - Entry/exit intents known at bar t close -> shift(1) -> fill at bar t+1 open.
    - Stop: prior bar stop level vs current bar low/high; fill min/max(open, stop).
    - Fees: fixed Rs 125 per order (Rs 250 round trip on Rs 1L).
    """
    avoid_rules = avoid_rules or []

    long_entry = apply_avoid_mask(sig["long_entry"], raw, sig, avoid_rules)
    long_exit = sig["long_exit"].fillna(False)
    short_entry = apply_avoid_mask(sig["short_entry"], raw, sig, avoid_rules)
    short_exit = sig["short_exit"].fillna(False)

    # Confirm on close -> next open
    le = long_entry.shift(1, fill_value=False)
    lx = long_exit.shift(1, fill_value=False)
    se = short_entry.shift(1, fill_value=False)
    sx = short_exit.shift(1, fill_value=False)

    execution_price = raw["Open"].astype(float).copy()

    if use_st_stop and sig.get("stop_level") is not None:
        prev_stop = sig["stop_level"].shift(1)
        # Long stop: low through prior stop
        long_stop_hit = prev_stop.notna() & raw["Low"].le(prev_stop)
        # Short stop: high through prior stop
        short_stop_hit = prev_stop.notna() & raw["High"].ge(prev_stop)

        # Gap-aware fill
        long_fill = np.minimum(raw["Open"].to_numpy(), prev_stop.to_numpy())
        short_fill = np.maximum(raw["Open"].to_numpy(), prev_stop.to_numpy())
        px = execution_price.to_numpy().copy()
        ls = long_stop_hit.fillna(False).to_numpy()
        ss = short_stop_hit.fillna(False).to_numpy()
        px = np.where(ls, long_fill, px)
        px = np.where(ss & ~ls, short_fill, px)
        execution_price = pd.Series(px, index=raw.index)
        lx = lx | long_stop_hit.fillna(False)
        sx = sx | short_stop_hit.fillna(False)

    common = dict(
        close=raw["Close"],
        price=execution_price,
        open=raw["Open"],
        high=raw["High"],
        low=raw["Low"],
        init_cash=INIT_CASH,
        fixed_fees=FIXED_FEE_PER_ORDER,
        freq=VBT_FREQ,
        size_granularity=1.0,  # whole index units as proxy
        fees=0.0,
        slippage=0.0,  # bundled into the Rs 250 round-trip assumption
    )

    if direction_mode == "long_only":
        return vbt.Portfolio.from_signals(
            entries=le,
            exits=lx,
            short_entries=False,
            short_exits=False,
            **common,
        )

    # long_short: allow both sides (vectorbt handles flips)
    return vbt.Portfolio.from_signals(
        entries=le,
        exits=lx,
        short_entries=se,
        short_exits=sx,
        **common,
    )


def portfolio_metrics(pf: vbt.Portfolio) -> dict[str, float]:
    trades = pf.trades.records_readable
    n = len(trades)
    return {
        "total_return_pct": float(pf.total_return() * 100),
        "cagr_pct": float(pf.annualized_return(year_freq=YEAR_FREQ) * 100) if n else 0.0,
        "sharpe_ratio": float(pf.sharpe_ratio(year_freq=YEAR_FREQ)) if n else np.nan,
        "max_drawdown_pct": float(pf.max_drawdown() * 100),
        "num_trades": int(n),
        "win_rate_pct": float(pf.trades.win_rate() * 100) if n else 0.0,
        "profit_factor": float(pf.trades.profit_factor()) if n else np.nan,
        "final_value": float(pf.final_value()),
    }


def composite_score(m: dict[str, float]) -> float:
    if m["num_trades"] < MIN_TRAIN_TRADES or not np.isfinite(m.get("sharpe_ratio", np.nan)):
        return -1e9
    # Prefer stable edge after costs; penalize deep DD and sparse trading lightly
    return (
        0.55 * m["sharpe_ratio"]
        + 0.25 * (m["cagr_pct"] / 15.0)
        - 0.20 * (abs(m["max_drawdown_pct"]) / 15.0)
    )


# ==============================================================================
# MISTAKE MEMORY (train-only learning, frozen on test)
# ==============================================================================

def _trade_entry_context(raw: pd.DataFrame, sig: dict[str, Any], trades_df: pd.DataFrame) -> pd.DataFrame:
    """Attach entry-time features to each closed trade (no future info)."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    # vectorbt readable trades use Entry Timestamp / Exit Timestamp / PnL
    cols = {c.lower().replace(" ", "_"): c for c in trades_df.columns}
    entry_col = cols.get("entry_timestamp") or cols.get("entry_idx")
    pnl_col = cols.get("pnl") or cols.get("return")
    if entry_col is None:
        for c in trades_df.columns:
            if "Entry" in c and "Time" in c:
                entry_col = c
                break
    if pnl_col is None:
        for c in trades_df.columns:
            if c.lower() == "pnl" or "PnL" in c:
                pnl_col = c
                break
    if entry_col is None or pnl_col is None:
        return pd.DataFrame()

    rows = []
    atr = sig.get("atr")
    atr_s = atr if isinstance(atr, pd.Series) else pd.Series(atr, index=raw.index)
    atr_rank = causal_percentile_rank(atr_s, window=100)

    for _, tr in trades_df.iterrows():
        ts = pd.Timestamp(tr[entry_col])
        # entry was filled at this bar open; signal came from prior close.
        # Features must be known before/at fill: use prior bar (shift) context.
        if ts not in raw.index:
            # nearest previous label
            loc = raw.index.searchsorted(ts)
            if loc == 0:
                continue
            ts = raw.index[loc - 1 if raw.index[loc] != ts else loc]
            if ts not in raw.index:
                continue
        i = raw.index.get_loc(ts)
        prev_i = max(0, i - 1)
        prev_ts = raw.index[prev_i]
        pnl = float(tr[pnl_col])
        rows.append(
            {
                "entry_time": ts,
                "signal_time": prev_ts,
                "pnl": pnl,
                "is_loss": pnl < 0,
                "hour": prev_ts.hour,
                "dow": prev_ts.dayofweek,
                "atr_rank": float(atr_rank.loc[prev_ts]) if np.isfinite(atr_rank.loc[prev_ts]) else np.nan,
                "nw_bull": bool(sig["nw_bull"].loc[prev_ts]) if "nw_bull" in sig else False,
            }
        )
    return pd.DataFrame(rows)


def learn_mistake_rules(ctx: pd.DataFrame) -> list[dict[str, Any]]:
    """
    From train losing trades, propose simple avoid filters.
    Only keep rules that concentrate losses and are not the majority of all trades
    (avoid banning everything).
    """
    if ctx.empty:
        return []
    losses = ctx[ctx["is_loss"]]
    if len(losses) < LOSS_CLUSTER_MIN_COUNT:
        return []

    candidates: list[tuple[float, dict[str, Any]]] = []
    n_loss = len(losses)
    n_all = len(ctx)

    # Hour clusters
    for hour, g in losses.groupby("hour"):
        share = len(g) / n_loss
        all_share = (ctx["hour"] == hour).mean()
        if len(g) >= LOSS_CLUSTER_MIN_COUNT and share >= LOSS_CLUSTER_MIN_SHARE and all_share < 0.35:
            # score: loss concentration vs how much trading we kill
            score = share / max(all_share, 1e-6)
            candidates.append((score, {"type": "avoid_hour", "hour": int(hour),
                                       "train_loss_share": round(share, 3),
                                       "train_trade_share": round(float(all_share), 3)}))

    # Day-of-week clusters
    for dow, g in losses.groupby("dow"):
        share = len(g) / n_loss
        all_share = (ctx["dow"] == dow).mean()
        if len(g) >= LOSS_CLUSTER_MIN_COUNT and share >= LOSS_CLUSTER_MIN_SHARE and all_share < 0.40:
            score = share / max(all_share, 1e-6)
            candidates.append((score, {"type": "avoid_dow", "dow": int(dow),
                                       "train_loss_share": round(share, 3),
                                       "train_trade_share": round(float(all_share), 3)}))

    # Hour x DOW fine clusters
    for (hour, dow), g in losses.groupby(["hour", "dow"]):
        share = len(g) / n_loss
        mask = (ctx["hour"] == hour) & (ctx["dow"] == dow)
        all_share = mask.mean()
        if len(g) >= max(5, LOSS_CLUSTER_MIN_COUNT // 2) and share >= 0.12 and all_share < 0.15:
            score = share / max(all_share, 1e-6)
            candidates.append((score, {"type": "avoid_hour_dow", "hour": int(hour), "dow": int(dow),
                                       "train_loss_share": round(share, 3),
                                       "train_trade_share": round(float(all_share), 3)}))

    # High ATR regime losses
    la = losses["atr_rank"].dropna()
    if len(la) >= LOSS_CLUSTER_MIN_COUNT:
        thr = float(la.quantile(0.75))
        if thr >= 0.70:
            g = losses[losses["atr_rank"] >= thr]
            share = len(g) / n_loss
            all_share = (ctx["atr_rank"] >= thr).mean()
            if share >= LOSS_CLUSTER_MIN_SHARE and all_share < 0.40:
                candidates.append((share / max(all_share, 1e-6),
                                   {"type": "avoid_high_atr", "atr_rank_min": round(thr, 3),
                                    "train_loss_share": round(share, 3),
                                    "train_trade_share": round(float(all_share), 3)}))

    candidates.sort(key=lambda x: x[0], reverse=True)
    # Greedy non-overlapping-ish pick
    picked: list[dict[str, Any]] = []
    used_hours = set()
    used_dows = set()
    for _, rule in candidates:
        if len(picked) >= AVOID_RULE_MAX:
            break
        if rule["type"] == "avoid_hour" and rule["hour"] in used_hours:
            continue
        if rule["type"] == "avoid_dow" and rule["dow"] in used_dows:
            continue
        if rule["type"] == "avoid_hour":
            used_hours.add(rule["hour"])
        if rule["type"] == "avoid_dow":
            used_dows.add(rule["dow"])
        picked.append(rule)
    return picked


def save_memory(memory: dict[str, Any], path: str = MEMORY_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, default=str)


def load_memory(path: str = MEMORY_PATH) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# OPTIMIZATION + WALK-FORWARD
# ==============================================================================

def param_grid_for(strategy_name: str) -> list[StrategyParams]:
    grid: list[StrategyParams] = []
    if strategy_name in ("NW_Cross", "NW_SlopeTurn"):
        for h, r, lag, ha, mode in product(NW_H_GRID, NW_R_GRID, NW_LAG_GRID, HA_GRID, DIRECTION_MODES):
            if strategy_name == "NW_SlopeTurn" and lag != NW_LAG_GRID[0]:
                continue  # lag unused for slope
            if h - lag <= 0.5:
                continue
            grid.append(StrategyParams(
                name=strategy_name, h=h, r=r, lag=lag, use_ha=ha,
                direction_mode=mode, use_st_stop=False,
            ))
    else:
        use_stop = strategy_name == "NW_ST_Confirm_SL"
        for h, r, lag, sp, sm, ha, mode in product(
            ST_SEARCH_H, ST_SEARCH_R, ST_SEARCH_LAG,
            ST_SEARCH_PERIOD, ST_SEARCH_MULT, HA_GRID, DIRECTION_MODES,
        ):
            if h - lag <= 0.5:
                continue
            grid.append(StrategyParams(
                name=strategy_name, h=h, r=r, lag=lag,
                st_period=sp, st_mult=sm, use_ha=ha,
                direction_mode=mode, use_st_stop=use_stop,
            ))
    return grid


def evaluate_params(
    raw: pd.DataFrame,
    p: StrategyParams,
    avoid_rules: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, float], vbt.Portfolio | None, dict[str, Any] | None]:
    builder = SIGNAL_BUILDERS[p.name]
    try:
        # ST strategies need use_st_stop flag on params object
        if p.name.startswith("NW_ST"):
            sig = builder(raw, p)
        else:
            sig = builder(raw, p)
        pf = run_portfolio(
            raw, sig,
            direction_mode=p.direction_mode,
            use_st_stop=p.use_st_stop,
            avoid_rules=avoid_rules,
        )
        return portfolio_metrics(pf), pf, sig
    except Exception as exc:
        print(f"  [WARN] evaluate failed for {asdict(p)}: {exc}", flush=True)
        return {
            "total_return_pct": np.nan, "cagr_pct": np.nan, "sharpe_ratio": np.nan,
            "max_drawdown_pct": np.nan, "num_trades": 0, "win_rate_pct": 0.0,
            "profit_factor": np.nan, "final_value": INIT_CASH,
        }, None, None


def optimize_on_train(train: pd.DataFrame, strategy_name: str) -> tuple[StrategyParams | None, pd.DataFrame]:
    rows = []
    best_p = None
    best_score = -1e18
    clear_nw_cache()
    grid = param_grid_for(strategy_name)

    print(f"  Train grid size: {len(grid)}", flush=True)
    for i, p in enumerate(grid, 1):
        m, _, _ = evaluate_params(train, p, avoid_rules=None)
        score = composite_score(m)
        rows.append({**asdict(p), **m, "score": score})
        if score > best_score:
            best_score = score
            best_p = p
        if i % 10 == 0 or i == len(grid):
            print(f"    ... {i}/{len(grid)}  best_score={best_score:.3f}", flush=True)

    return best_p, pd.DataFrame(rows)


def run_strategy_walkforward(raw: pd.DataFrame, strategy_name: str) -> dict[str, Any]:
    train = raw.loc[:TRAIN_END].copy()
    test = raw.loc[TEST_START:].copy()
    clear_nw_cache()
    print(f"\n===== {strategy_name} =====", flush=True)
    print(f"  Train bars: {len(train)}  ({train.index.min()} -> {train.index.max()})", flush=True)
    print(f"  Test  bars: {len(test)}  ({test.index.min()} -> {test.index.max()})", flush=True)

    best_p, train_grid = optimize_on_train(train, strategy_name)
    train_grid.to_csv(os.path.join(OUTPUT_DIR, f"{strategy_name}_train_grid.csv"), index=False)

    if best_p is None:
        print("  No viable params on train.")
        return {"strategy": strategy_name, "error": "no_params"}

    print(f"  Best train params: {asdict(best_p)}")

    # Baseline train/test without mistake filters
    m_tr, pf_tr, sig_tr = evaluate_params(train, best_p, None)
    m_te, pf_te, sig_te = evaluate_params(test, best_p, None)
    print(f"  Train metrics (no filter): {m_tr}")
    print(f"  Test  metrics (no filter): {m_te}")

    # Learn mistakes on train trades only
    avoid_rules: list[dict[str, Any]] = []
    if pf_tr is not None and sig_tr is not None:
        tdf = pf_tr.trades.records_readable
        ctx = _trade_entry_context(train, sig_tr, tdf)
        if not ctx.empty:
            ctx.to_csv(os.path.join(OUTPUT_DIR, f"{strategy_name}_train_trade_context.csv"), index=False)
            avoid_rules = learn_mistake_rules(ctx)
            print(f"  Mistake rules learned: {avoid_rules}")

    # Re-evaluate with frozen avoid rules
    m_tr_f, pf_tr_f, _ = evaluate_params(train, best_p, avoid_rules)
    m_te_f, pf_te_f, _ = evaluate_params(test, best_p, avoid_rules)
    print(f"  Train metrics (filtered): {m_tr_f}")
    print(f"  Test  metrics (filtered): {m_te_f}")

    # Full-sample equity with frozen params+rules (for charting only; still report OOS separately)
    m_full, pf_full, _ = evaluate_params(raw, best_p, avoid_rules)

    # Persist trades OOS
    if pf_te_f is not None:
        pf_te_f.trades.records_readable.to_csv(
            os.path.join(OUTPUT_DIR, f"{strategy_name}_oos_trades_filtered.csv"), index=False
        )
    if pf_te is not None:
        pf_te.trades.records_readable.to_csv(
            os.path.join(OUTPUT_DIR, f"{strategy_name}_oos_trades_baseline.csv"), index=False
        )

    result = {
        "strategy": strategy_name,
        "params": asdict(best_p),
        "avoid_rules": avoid_rules,
        "train_baseline": m_tr,
        "test_baseline": m_te,
        "train_filtered": m_tr_f,
        "test_filtered": m_te_f,
        "full_filtered": m_full,
        "sign_stable_sharpe": (
            np.isfinite(m_tr.get("sharpe_ratio", np.nan))
            and np.isfinite(m_te_f.get("sharpe_ratio", np.nan))
            and (m_tr["sharpe_ratio"] > 0)
            and (m_te_f["sharpe_ratio"] > 0)
        ),
    }

    # Save equity curve OOS filtered
    if pf_te_f is not None:
        eq = pf_te_f.value()
        eq.to_csv(os.path.join(OUTPUT_DIR, f"{strategy_name}_oos_equity_filtered.csv"), header=["equity"])

    return result


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("Loading data...")
    raw = load_ohlc(CSV_PATH)
    print(f"Bars: {len(raw)}  {raw.index.min()} -> {raw.index.max()}")
    print(f"Costs: Rs {FIXED_FEE_PER_ORDER:.0f}/order  (Rs {ROUND_TRIP_CHARGES:.0f} round-trip)")
    print(f"Cash: Rs {INIT_CASH:,.0f}")
    print(f"Execution: signal on close -> fill next open | no random metrics | walk-forward OOS")

    # Warm up numba
    _ = nw_regression(raw["Close"].iloc[:300], 8.0, 8.0)
    _ = supertrend(raw.iloc[:300], 14, 2.0)

    strategies = ["NW_Cross", "NW_SlopeTurn", "NW_ST_Confirm", "NW_ST_Confirm_SL"]
    all_results = []
    memory_blob: dict[str, Any] = {
        "version": 1,
        "instrument": "NIFTY50_INDEX_30m",
        "train_end": str(TRAIN_END),
        "test_start": str(TEST_START),
        "cost_model": {
            "init_cash": INIT_CASH,
            "round_trip_inr": ROUND_TRIP_CHARGES,
            "fee_per_order_inr": FIXED_FEE_PER_ORDER,
            "slippage": "included_in_round_trip",
        },
        "execution": "next_bar_open_after_close_signal",
        "strategies": {},
    }

    for name in strategies:
        res = run_strategy_walkforward(raw, name)
        all_results.append(res)
        if "params" in res:
            memory_blob["strategies"][name] = {
                "params": res["params"],
                "avoid_rules": res["avoid_rules"],
                "train_baseline": res["train_baseline"],
                "test_baseline": res["test_baseline"],
                "train_filtered": res["train_filtered"],
                "test_filtered": res["test_filtered"],
                "sign_stable_sharpe": res["sign_stable_sharpe"],
                "updated_at": str(pd.Timestamp.utcnow()),
            }

    save_memory(memory_blob)
    print(f"\nMistake memory saved -> {MEMORY_PATH}")

    # Summary table
    summary_rows = []
    for res in all_results:
        if "error" in res:
            continue
        for phase, key in [
            ("train_base", "train_baseline"),
            ("test_base", "test_baseline"),
            ("train_filt", "train_filtered"),
            ("test_filt", "test_filtered"),
        ]:
            m = res[key]
            summary_rows.append({
                "strategy": res["strategy"],
                "phase": phase,
                "h": res["params"]["h"],
                "r": res["params"]["r"],
                "lag": res["params"]["lag"],
                "use_ha": res["params"]["use_ha"],
                "direction_mode": res["params"]["direction_mode"],
                "st_period": res["params"].get("st_period"),
                "st_mult": res["params"].get("st_mult"),
                "use_st_stop": res["params"].get("use_st_stop"),
                "n_avoid_rules": len(res["avoid_rules"]),
                "sign_stable": res["sign_stable_sharpe"],
                **m,
            })

    summary = pd.DataFrame(summary_rows)
    out_csv = os.path.join(OUTPUT_DIR, "walkforward_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"Summary -> {out_csv}")

    # Console scoreboard (OOS only)
    print("\n" + "=" * 78)
    print("OUT-OF-SAMPLE SCOREBOARD (2024+), costs included")
    print("=" * 78)
    oos = summary[summary["phase"].isin(["test_base", "test_filt"])].copy()
    if not oos.empty:
        show_cols = [
            "strategy", "phase", "num_trades", "win_rate_pct", "profit_factor",
            "total_return_pct", "cagr_pct", "sharpe_ratio", "max_drawdown_pct",
            "final_value", "n_avoid_rules", "sign_stable",
        ]
        print(oos[show_cols].to_string(index=False, float_format=lambda x: f"{x: .3f}"))
    print("=" * 78)
    print(
        "\nHow to read this:\n"
        "- test_base  = best train params, no mistake filter (true OOS)\n"
        "- test_filt  = same params + avoid rules learned ONLY from train losses\n"
        "- sign_stable = train Sharpe>0 AND filtered test Sharpe>0\n"
        "- If test collapses vs train, treat as overfit (do not go live).\n"
        "- Mistake memory is NOT magic; it is a frozen train filter. Re-fit only on new train folds.\n"
    )


if __name__ == "__main__":
    main()
