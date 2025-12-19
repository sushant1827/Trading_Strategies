"""
multi_kernel_regression.py

Importable module that implements the Pine Script "Multi Kernel Regression" logic
without look-ahead (no forward-looking bias). Includes multiple kernels and
a built-in "check twice" numeric validation.

Usage:
    from multi_kernel_regression import MultiKernelRegression
    mkr = MultiKernelRegression(bandwidth=14, kernel="Laplace", deviations=2.0)
    out_df = mkr.fit_transform(df['Close'])  # returns DataFrame with signals
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
import math
import warnings

# ---------- Kernel functions ----------
def _sq(x: float) -> float:
    return x * x

def gaussian(x: float, bw: float) -> float:
    return math.exp(-_sq(x / bw) / 2.0) / math.sqrt(2.0 * math.pi)

def triangular(x: float, bw: float) -> float:
    a = abs(x / bw)
    return 1.0 - a if a <= 1.0 else 0.0

def epanechnikov(x: float, bw: float) -> float:
    a = abs(x / bw)
    return 0.75 * (1.0 - _sq(a)) if a <= 1.0 else 0.0

def quartic(x: float, bw: float) -> float:
    a = abs(x / bw)
    return (15.0 / 16.0) * pow(1.0 - _sq(a), 2) if a <= 1.0 else 0.0

def logistic(x: float, bw: float) -> float:
    return 1.0 / (math.exp(x / bw) + 2.0 + math.exp(-x / bw))

def loglogistic(x: float, bw: float) -> float:
    return 1.0 / pow(1.0 + abs(x / bw), 2)

def cosine(x: float, bw: float) -> float:
    a = abs(x / bw)
    return (math.pi / 4.0) * math.cos((math.pi / 2.0) * (x / bw)) if a <= 1.0 else 0.0

def sinc(x: float, bw: float) -> float:
    if x == 0.0:
        return 1.0
    y = math.pi * x / bw
    return math.sin(y) / y

def laplace(x: float, bw: float) -> float:
    return (1.0 / (2.0 * bw)) * math.exp(-abs(x / bw))

def exponential(x: float, bw: float) -> float:
    return (1.0 / bw) * math.exp(-abs(x / bw))

def silverman(x: float, bw: float) -> float:
    a = abs(x / bw)
    if a <= 0.5:
        return 0.5 * math.exp(-(x / bw) / 2.0) * math.sin((x / bw) / 2.0 + math.pi / 4.0)
    return 0.0

def cauchy(x: float, bw: float) -> float:
    return 1.0 / (math.pi * bw * (1.0 + _sq(x / bw)))

def tent(x: float, bw: float) -> float:
    a = abs(x / bw)
    return 1.0 - a if a <= 1.0 else 0.0

def wave(x: float, bw: float) -> float:
    a = abs(x / bw)
    if a <= 1.0:
        return (1.0 - a) * math.cos((math.pi * x) / bw)
    return 0.0

def parabolic(x: float, bw: float) -> float:
    a = abs(x / bw)
    return 1.0 - _sq(x / bw) if a <= 1.0 else 0.0

def power(x: float, bw: float) -> float:
    a = abs(x / bw)
    if a <= 1.0:
        return pow(1.0 - pow(a, 3.0), 3.0)
    return 0.0

def morters(x: float, bw: float) -> float:
    a = abs(x / bw)
    if a <= math.pi:
        return (1.0 + math.cos(x / bw)) / (2.0 * math.pi * bw)
    return 0.0

# Map kernel name -> function
_KERNEL_MAP = {
    "Triangular": triangular,
    "Gaussian": gaussian,
    "Epanechnikov": epanechnikov,
    "Logistic": logistic,
    "Log Logistic": loglogistic,
    "Cosine": cosine,
    "Sinc": sinc,
    "Laplace": laplace,
    "Quartic": quartic,
    "Parabolic": parabolic,
    "Exponential": exponential,
    "Silverman": silverman,
    "Cauchy": cauchy,
    "Tent": tent,
    "Wave": wave,
    "Power": power,
    "Morters": morters
}

# ---------- Core class ----------
class MultiKernelRegression:
    """
    Multi Kernel Regression class.

    Example:
        mkr = MultiKernelRegression(bandwidth=14, kernel="Laplace", deviations=2.0)
        out = mkr.fit_transform(series)   # series is a pd.Series of prices
    """

    def __init__(self, bandwidth: int = 14, kernel: str = "Laplace", deviations: float = 2.0, repaint: bool = False):
        if kernel not in _KERNEL_MAP:
            raise ValueError(f"Kernel '{kernel}' not supported. Supported: {list(_KERNEL_MAP.keys())}")
        if bandwidth < 1:
            raise ValueError("bandwidth must be >= 1")
        self.bandwidth = int(bandwidth)
        self.kernel_name = kernel
        self.kernel = _KERNEL_MAP[kernel]
        self.deviations = float(deviations)
        self.repaint = bool(repaint)
        if self.repaint:
            # We strongly discourage repaint. We keep API but do not implement forward-looking results.
            warnings.warn("repaint=True is not supported for forward-looking behavior. Module will run non-repainting calculations (past+current only).", UserWarning)

    def _regression_single_pass(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Core calculation using explicit loop (one method).
        For each index i, use data indices: max(0, i-bandwidth+1) .. i (past and current only).
        Returns (regression_values, std_devs).
        """
        n = len(prices)
        reg = np.zeros(n, dtype=float)
        stdev = np.zeros(n, dtype=float)

        bw = self.bandwidth
        for i in range(n):
            start = max(0, i - bw + 1)
            sum_w = 0.0
            weighted_sum = 0.0
            # weights over distance (i - j) where j <= i
            for j in range(start, i + 1):
                diff = float(i - j)        # distance (0 for current bar)
                w = self.kernel(diff, float(self.bandwidth))
                sum_w += w
                weighted_sum += w * prices[j]
            if sum_w > 0.0:
                mu = weighted_sum / sum_w
                reg[i] = mu
                # variance / stddev (weighted)
                var_sum = 0.0
                for j in range(start, i + 1):
                    diff = float(i - j)
                    w = self.kernel(diff, float(self.bandwidth))
                    var_sum += w * ((prices[j] - mu) ** 2)
                # Use sample-like normalization consistent with Pine's version:
                # Pine used sqrt(sumsq / sumw - sq(mean)) in repaint branch; for non-repaint it used sqrt(sumsq/(n-1)) * deviations.
                # We choose a stable weighted variance equivalent: var = var_sum / sum_w
                variance = var_sum / sum_w if sum_w != 0.0 else 0.0
                stdev[i] = math.sqrt(max(0.0, variance)) * self.deviations
            else:
                reg[i] = prices[i]
                stdev[i] = 0.0

        return reg, stdev

    def _regression_vectorized_check(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        A second independent calculation (vectorized over the window using numpy) used for numerical validation.
        This implements the same logic but computes weights vector for each i and applies dot products.
        """
        n = len(prices)
        reg = np.zeros(n, dtype=float)
        stdev = np.zeros(n, dtype=float)
        bw = self.bandwidth

        for i in range(n):
            start = max(0, i - bw + 1)
            window = prices[start:i + 1]
            # distances: i - j, where j runs start..i
            dists = np.arange(i - start, -1, -1, dtype=float)  # e.g. [i-start, ..., 0]
            # compute weights vectorized
            weights = np.array([self.kernel(d, float(self.bandwidth)) for d in dists], dtype=float)
            sum_w = weights.sum()
            if sum_w > 0.0:
                mu = float(np.dot(weights, window) / sum_w)
                reg[i] = mu
                var_sum = float(np.dot(weights, (window - mu) ** 2))
                variance = var_sum / sum_w
                stdev[i] = math.sqrt(max(0.0, variance)) * self.deviations
            else:
                reg[i] = prices[i]
                stdev[i] = 0.0

        return reg, stdev

    def _double_check(self, prices: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run both implementations and assert they match within tolerance.
        Returns regression (using looped implementation) and stdev.
        """
        r1, s1 = self._regression_single_pass(prices)
        r2, s2 = self._regression_vectorized_check(prices)

        # Numeric safety: use relative tolerance and absolute tolerance
        if not (np.allclose(r1, r2, atol=tol, rtol=1e-6) and np.allclose(s1, s2, atol=tol, rtol=1e-6)):
            # Prepare diagnostics for debugging
            idx = np.where(~np.isclose(r1, r2, atol=tol, rtol=1e-6))[0]
            # Raise a useful error rather than silently proceed
            raise AssertionError(f"Regression implementations disagree at indices {idx[:10]} (showing up to 10). "
                                 "This indicates a numerical bug. Values (r1 vs r2) at first mismatch: "
                                 f"{r1[idx[0]]} vs {r2[idx[0]]}")
        return r1, s1

    def fit_transform(self, price_series: pd.Series) -> pd.DataFrame:
        """
        Compute kernel regression, bands and signals for the given price series.

        Args:
            price_series: pd.Series (index preserved) of price values (e.g., close prices)

        Returns:
            pd.DataFrame with columns:
                'price', 'kernel_regression', 'upper_band', 'lower_band', 'regression_change', 'signal', 'signal_label'
        """
        if not isinstance(price_series, pd.Series):
            price_series = pd.Series(price_series)

        # prices = price_series.astype(float).fillna(method="ffill").fillna(method="bfill").to_numpy()
        prices = price_series.astype(float).ffill().bfill().to_numpy()


        # Calculate regression and stdev, with a double-check
        reg, stdev = self._double_check(prices)

        df = pd.DataFrame(index=price_series.index)
        df['price'] = price_series.astype(float).values
        df['kernel_regression'] = reg
        df['upper_band'] = reg + stdev
        df['lower_band'] = reg - stdev

        # Signals: when regression turns up vs previous -> 1 (BUY), turns down -> -1 (SELL)
        df['regression_change'] = df['kernel_regression'].diff()
        df['signal'] = 0
        df.loc[df['regression_change'] > 0, 'signal'] = 1
        df.loc[df['regression_change'] < 0, 'signal'] = -1
        df['signal_label'] = df['signal'].map({1: 'BUY', -1: 'SELL', 0: 'HOLD'})

        return df
    
    def generate_signals(self, df: pd.DataFrame, price_column: str = "Close") -> pd.DataFrame:
        """
        Generate buy/sell signals for an OHLC DataFrame using this MultiKernelRegression.

        Args:
            df: pd.DataFrame containing at least `price_column` (e.g. 'Close').
                Index is preserved.
            price_column: name of the price column to use (default: "Close").

        Returns:
            pd.DataFrame: A copy of `df` with the following new columns added:
                - 'kernel_regression'
                - 'upper_band'
                - 'lower_band'
                - 'regression_change'
                - 'signal' (1 buy, -1 sell, 0 hold)
                - 'signal_label' ("BUY","SELL","HOLD")
        """
        if price_column not in df.columns:
            raise KeyError(f"price_column '{price_column}' not found in DataFrame columns: {list(df.columns)}")

        # work on a copy so we don't mutate user data
        out = df.copy()

        # compute regression + bands + signals using the class' safe implementation
        series = out[price_column].astype(float).ffill().bfill()  # future-proof fill
        result = self.fit_transform(series)

        # result contains index same as series; merge columns back into out
        # avoid overwriting existing user columns except those intentionally added
        out['kernel_regression'] = result['kernel_regression'].values
        out['upper_band'] = result['upper_band'].values
        out['lower_band'] = result['lower_band'].values
        out['regression_change'] = result['regression_change'].values
        out['signal'] = result['signal'].values
        out['signal_label'] = result['signal_label'].values

        return out


# ---------- Example quick test ----------
def _quick_unit_test():
    # small synthetic dataset for a deterministic test
    s = pd.Series([1.0, 1.1, 0.9, 1.2, 1.15, 1.18, 1.25, 1.2, 1.3, 1.4])
    mkr = MultiKernelRegression(bandwidth=3, kernel="Laplace", deviations=1.5)
    out = mkr.fit_transform(s)
    # basic sanity checks
    assert 'kernel_regression' in out.columns
    assert len(out) == len(s)
    # check that first reg equals first price (no prior)
    assert abs(out['kernel_regression'].iloc[0] - s.iloc[0]) < 1e-12
    # ensure signals are only -1,0,1
    assert set(out['signal'].unique()).issubset({-1, 0, 1})
    print("Quick unit test passed.")

if __name__ == "__main__":
    # Run a simple self-test when executed directly:
    _quick_unit_test()
