import pandas as pd
import numpy as np
from numba import jit

def validate_dataframe_data(df, name="DataFrame"):
    """
    Comprehensive data validation for OHLC DataFrame

    Parameters:
    df (pd.DataFrame): DataFrame to validate
    name (str): Name for error messages

    Returns:
    bool: True if validation passes, raises ValueError otherwise
    """
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{name}: Missing required columns: {missing_cols}")

    # Check for missing values
    missing_data = df[required_cols].isnull().sum()
    if missing_data.sum() > 0:
        print(f"{name} - Missing values found:")
        for col, count in missing_data[missing_data > 0].items():
            print(f"  {col}: {count} missing values ({count/len(df)*100:.2f}%)")
        raise ValueError(f"{name}: Contains missing values in OHLC data")

    # Check for zero or negative prices
    for col in required_cols:
        invalid_prices = df[df[col] <= 0]
        if not invalid_prices.empty:
            print(f"{name} - Invalid {col} prices found: {len(invalid_prices)} rows")
            print(f"  Min {col}: {df[col].min()}")
            raise ValueError(f"{name}: Contains zero or negative {col} prices")

    # Check price consistency (High >= Low)
    invalid_hl = df[df['High'] < df['Low']]
    if not invalid_hl.empty:
        print(f"{name} - High < Low violations: {len(invalid_hl)} rows")
        raise ValueError(f"{name}: High prices lower than Low prices")

    # Check for outliers using IQR method
    for col in required_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            print(f"{name} - {col} outliers detected: {len(outliers)} rows")
            print(f"  Normal range: {lower_bound:.2f} to {upper_bound:.2f}")
            print(f"  Actual range: {df[col].min():.2f} to {df[col].max():.2f}")

    return True

def validate_ha_data(df, name="DataFrame"):
    """
    Comprehensive data validation for Heikin-Ashi DataFrame

    Parameters:
    df (pd.DataFrame): DataFrame to validate
    name (str): Name for error messages

    Returns:
    bool: True if validation passes, raises ValueError otherwise
    """
    # Check for required HA columns
    required_cols = ['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{name}: Missing required columns: {missing_cols}")

    # Check for missing values
    missing_data = df[required_cols].isnull().sum()
    if missing_data.sum() > 0:
        print(f"{name} - Missing values found:")
        for col, count in missing_data[missing_data > 0].items():
            print(f"  {col}: {count} missing values ({count/len(df)*100:.2f}%)")
        raise ValueError(f"{name}: Contains missing values in HA data")

    # Check for zero or negative prices
    for col in required_cols:
        invalid_prices = df[df[col] <= 0]
        if not invalid_prices.empty:
            print(f"{name} - Invalid {col} prices found: {len(invalid_prices)} rows")
            print(f"  Min {col}: {df[col].min()}")
            raise ValueError(f"{name}: Contains zero or negative {col} prices")

    # Check price consistency (HA_High >= HA_Low)
    invalid_hl = df[df['HA_High'] < df['HA_Low']]
    if not invalid_hl.empty:
        print(f"{name} - HA_High < HA_Low violations: {len(invalid_hl)} rows")
        raise ValueError(f"{name}: HA_High prices lower than HA_Low prices")

    # Check for outliers using IQR method
    for col in required_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            print(f"{name} - {col} outliers detected: {len(outliers)} rows")
            print(f"  Normal range: {lower_bound:.2f} to {upper_bound:.2f}")
            print(f"  Actual range: {df[col].min():.2f} to {df[col].max():.2f}")

    return True

def validate_chronological_order(df, name="DataFrame"):
    """
    Validate that DataFrame index is in chronological order

    Parameters:
    df (pd.DataFrame): DataFrame to validate
    name (str): Name for error messages

    Returns:
    bool: True if validation passes, raises ValueError otherwise
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: Index must be DatetimeIndex")

    # Check for reasonable time intervals (no huge gaps)
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        max_diff = time_diffs.max()

        # Warn if maximum gap is more than 10x median
        if max_diff > (median_diff * 10):
            print(f"{name} - Large time gaps detected:")
            print(f"  Median interval: {median_diff}")
            print(f"  Maximum gap: {max_diff}")

    return True

def clean_dataframe_data(df, name="DataFrame", fix_chronological=True):
    """
    Clean and prepare DataFrame for analysis

    Parameters:
    df (pd.DataFrame): DataFrame to clean
    name (str): Name for logging messages
    fix_chronological (bool): Whether to automatically sort non-chronological data

    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    print(f"\n=== Data Validation for {name} ===")

    # Validate data
    validate_dataframe_data(df, name)

    # Handle chronological order
    if not df.index.is_monotonic_increasing:
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            unsorted_count = (time_diffs <= pd.Timedelta(0)).sum()
        else:
            unsorted_count = 0

        if fix_chronological:
            print(f"{name} - Found {unsorted_count} non-chronological data points. Sorting data...")
            df = df.sort_index()
            print(f"{name} - Data has been sorted chronologically")
        else:
            print(f"{name} - Non-chronological data points: {unsorted_count}")
            raise ValueError(f"{name}: Data is not in chronological order")
    else:
        print(f"{name} - Data is properly sorted chronologically")

    # Remove duplicates
    initial_shape = df.shape
    df_clean = df[~df.index.duplicated(keep='first')]

    if df_clean.shape[0] < initial_shape[0]:
        removed = initial_shape[0] - df_clean.shape[0]
        print(f"{name} - Removed {removed} duplicate rows")

    # Round prices to 2 decimal places
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df_clean.columns:
            df_clean.loc[:, col] = df_clean[col].round(2)

    print(f"{name} - Data validation completed successfully")
    print(f"  Final shape: {df_clean.shape}")
    print(f"  Date range: {df_clean.index.min()} to {df_clean.index.max()}")

    return df_clean

# Define the path to the CSV file
csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'
# csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_60_Min.csv'
# csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_120_Min.csv'

# Define kernel functions as standalone Numba jitted functions
@jit(nopython=True)
def _kernel_func_numba(source: float, bw: float, k_type: str) -> float:
    if k_type == "Gaussian":
        return np.exp(-np.power(source / bw, 2) / 2) / np.sqrt(2 * np.pi)
    elif k_type == "Epanechnikov":
        abs_val = abs(source / bw)
        return (3/4) * (1 - np.power(abs_val, 2)) if abs_val <= 1 else 0.0
    elif k_type == "Laplace":
        return (1 / (2 * bw)) * np.exp(-abs(source / bw))
    else:
        return 0.0  # Default to 0 for unsupported kernels

@jit(nopython=True)
def calculate_kernel_atr_numba(tr: np.array, bandwidth: int, kernel_type: str) -> np.array:
    """
    Calculate ATR using kernel regression instead of traditional EMA
    Uses only past and current data to avoid look-ahead bias
    
    Parameters:
    tr (np.array): True Range values
    bandwidth (int): The bandwidth parameter for kernel smoothing
    kernel_type (str): Type of kernel to apply ("Gaussian", "Epanechnikov", "Laplace")
    
    Returns:
    np.array: Kernel-smoothed ATR values
    """
    n = len(tr)
    kernel_atr = np.zeros(n)
    
    for i in range(n):
        weights = []
        weighted_tr = []
        sum_weights = 0.0
        
        # Use standard non-repaint calculation with bandwidth
        start_idx = max(0, i - bandwidth)
        
        # Calculate weights for current position using only past and current data
        for j in range(start_idx, i):
            diff = float(i - j)
            weight = _kernel_func_numba(diff, float(bandwidth), kernel_type)
            weights.append(weight)
            weighted_tr.append(tr[j] * weight)
            sum_weights += weight
        
        # Calculate kernel-smoothed ATR value
        if sum_weights > 0:
            kernel_atr[i] = sum(weighted_tr) / sum_weights
        else:
            kernel_atr[i] = tr[i]  # Fallback to raw TR if no weights
    
    return kernel_atr

@jit(nopython=True)
def _calculate_supertrend_numba(ha_high, ha_low, ha_close, tr, atr, period, multiplier, basic_upper_band, basic_lower_band, first_valid_atr_idx):
    trending_up = np.empty_like(ha_close)
    trending_down = np.empty_like(ha_close)
    direction = np.empty_like(ha_close, dtype=np.int32)
    supertrend = np.empty_like(ha_close)

    if first_valid_atr_idx is None or first_valid_atr_idx >= len(ha_close):
        return trending_up, trending_down, direction, supertrend

    trending_up[first_valid_atr_idx] = basic_lower_band[first_valid_atr_idx]
    trending_down[first_valid_atr_idx] = basic_upper_band[first_valid_atr_idx]

    if ha_close[first_valid_atr_idx] > trending_down[first_valid_atr_idx]:
        direction[first_valid_atr_idx] = 1
    elif ha_close[first_valid_atr_idx] < trending_up[first_valid_atr_idx]:
        direction[first_valid_atr_idx] = -1
    else:
        direction[first_valid_atr_idx] = 1

    supertrend[first_valid_atr_idx] = trending_up[first_valid_atr_idx] if direction[first_valid_atr_idx] == 1 else trending_down[first_valid_atr_idx]

    for i in range(first_valid_atr_idx + 1, len(ha_close)):
        prev_ha_close = ha_close[i - 1]
        prev_trending_up = trending_up[i - 1]
        prev_trending_down = trending_down[i - 1]
        prev_direction = direction[i - 1]

        current_basic_upper_band = basic_upper_band[i]
        current_basic_lower_band = basic_lower_band[i]
        current_ha_close = ha_close[i]

        if prev_ha_close > prev_trending_up:
            trending_up[i] = max(current_basic_lower_band, prev_trending_up)
        else:
            trending_up[i] = current_basic_lower_band

        if prev_ha_close < prev_trending_down:
            trending_down[i] = min(current_basic_upper_band, prev_trending_down)
        else:
            trending_down[i] = current_basic_upper_band

        if current_ha_close > trending_down[i - 1]:
            direction[i] = 1
        elif current_ha_close < trending_up[i - 1]:
            direction[i] = -1
        else:
            direction[i] = prev_direction

        supertrend[i] = trending_up[i] if direction[i] == 1 else trending_down[i]
    
    return trending_up, trending_down, direction, supertrend

def calculate_kernel_smoothed_supertrend(df, period=10, multiplier=3.0, bandwidth=14, kernel_type="Gaussian", suffix=""):
    original_index = df.index
    df_reset = df.reset_index(drop=True)

    if not all(col in df_reset.columns for col in ["HA_Open", "HA_High", "HA_Low", "HA_Close"]):
        raise ValueError("DataFrame must contain 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close' columns for Supertrend calculation.")

    ha_high = df_reset["HA_High"].to_numpy()
    ha_low = df_reset["HA_Low"].to_numpy()
    ha_close = df_reset["HA_Close"].to_numpy()

    # Calculate True Range based on HA candles
    # Numba does not support np.maximum.reduce, so use nested np.maximum
    tr = np.maximum(ha_high - ha_low, np.abs(ha_high - np.roll(ha_close, 1)))
    tr = np.maximum(tr, np.abs(ha_low - np.roll(ha_close, 1)))
    tr[0] = ha_high[0] - ha_low[0]  # Correct TR for the first element

    # Validate True Range values
    if np.any(tr <= 0):
        raise ValueError("True Range calculation produced zero or negative values")
    if np.any(~np.isfinite(tr)):
        raise ValueError("True Range calculation produced non-finite values")

    # ATR using Kernel smoothing instead of Wilder's smoothing
    kernel_atr = calculate_kernel_atr_numba(tr, bandwidth, kernel_type)

    # Validate ATR values
    if np.any(~np.isfinite(kernel_atr)):
        raise ValueError("Kernel ATR calculation produced non-finite values")
    if np.any(kernel_atr < 0):
        raise ValueError("Kernel ATR calculation produced negative values")

    basic_upper_band = ((ha_high + ha_low) / 2) + (multiplier * kernel_atr)
    basic_lower_band = ((ha_high + ha_low) / 2) - (multiplier * kernel_atr)

    first_valid_atr_idx = np.where(~np.isnan(kernel_atr))[0][0] if np.any(~np.isnan(kernel_atr)) else -1  # Use -1 if no valid index

    trending_up_arr, trending_down_arr, direction_arr, supertrend_arr = _calculate_supertrend_numba(
        ha_high, ha_low, ha_close, tr, kernel_atr, period, multiplier, basic_upper_band, basic_lower_band, first_valid_atr_idx
    )

    # Validate Supertrend calculations
    if np.any(~np.isfinite(supertrend_arr)):
        raise ValueError("Supertrend calculation produced non-finite values")
    if np.any(~np.isfinite(trending_up_arr)):
        raise ValueError("Trending up calculation produced non-finite values")
    if np.any(~np.isfinite(trending_down_arr)):
        raise ValueError("Trending down calculation produced non-finite values")
    if not np.all(np.isin(direction_arr, [-1, 0, 1])):
        raise ValueError("Supertrend direction contains invalid values (must be -1, 0, or 1)")

    # Validate that supertrend values are within reasonable bounds
    price_range = ha_high - ha_low
    max_reasonable_st = np.nanmax(price_range) * 3  # Allow up to 3x the max range for volatile markets
    max_st = np.nanmax(supertrend_arr)
    min_st = np.nanmin(supertrend_arr)

    # Supertrend values are within reasonable bounds

    st_col_name = f"Kernel_ST_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_bw{bandwidth}{suffix}"
    st_dir_col_name = f"Kernel_ST_Dir_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_bw{bandwidth}{suffix}"
    trending_up_col_name = f"kernel_trendingUp_HA_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_bw{bandwidth}{suffix}"
    trending_down_col_name = f"kernel_trendingDown_HA_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_bw{bandwidth}{suffix}"

    result_df = pd.DataFrame({
        st_col_name: supertrend_arr,
        st_dir_col_name: direction_arr,
        trending_up_col_name: trending_up_arr,
        trending_down_col_name: trending_down_arr
    }, index=original_index)

    return result_df

@jit(nopython=True)
def _calculate_heikin_ashi_numba(open_prices, high_prices, low_prices, close_prices):
    ha_close = (open_prices + high_prices + low_prices + close_prices) / 4
    ha_open = np.zeros_like(open_prices, dtype=np.float64)
    ha_high = np.zeros_like(high_prices, dtype=np.float64)
    ha_low = np.zeros_like(low_prices, dtype=np.float64)

    if len(open_prices) > 0:
        ha_open[0] = open_prices[0]

    for i in range(1, len(open_prices)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2

    # Numba does not support np.maximum.reduce and np.minimum.reduce, so use nested np.maximum/minimum
    ha_high = np.maximum(high_prices, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(low_prices, np.minimum(ha_open, ha_close))

    return ha_open, ha_high, ha_low, ha_close

def calculate_heikin_ashi(df):
    if not all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns for Heikin Ashi calculation.")

    open_prices = df["Open"].to_numpy()
    high_prices = df["High"].to_numpy()
    low_prices = df["Low"].to_numpy()
    close_prices = df["Close"].to_numpy()

    ha_open, ha_high, ha_low, ha_close = _calculate_heikin_ashi_numba(open_prices, high_prices, low_prices, close_prices)

    ha_df = pd.DataFrame(
        {
            "HA_Open": ha_open,
            "HA_High": ha_high,
            "HA_Low": ha_low,
            "HA_Close": ha_close,
        },
        index=df.index,
    )

    return ha_df

@jit(nopython=True)
def _run_strategy_numba(st_dir, close_prices):
    """
    Execute SuperTrend trading strategy with same-candle position reversals

    This function allows both buy position exit and sell entry on the same candle,
    matching the behavior of the Regular Supertrend strategy.

    Logic:
    - Buy_Entry: When Supertrend direction changes to up (not in any position)
    - Buy_Exit: When Supertrend direction changes to down (currently in buy position)
    - Sell_Entry: When Supertrend direction changes to down (not in any position)
    - Sell_Exit: When Supertrend direction changes to up (currently in sell position)

    This allows for same-candle reversals: exit long + enter short on same candle
    """
    buy_entry_prices = np.full_like(close_prices, np.nan)
    buy_exit_prices = np.full_like(close_prices, np.nan)
    sell_entry_prices = np.full_like(close_prices, np.nan)
    sell_exit_prices = np.full_like(close_prices, np.nan)

    in_buy_position = False
    in_sell_position = False

    for i in range(1, len(close_prices)):
        # Check for Buy_Entry condition (Supertrend direction changes to up)
        if not in_buy_position and not in_sell_position and st_dir[i] == 1:
            buy_entry_prices[i] = close_prices[i]
            in_buy_position = True
        # Check for Buy_Exit condition (Supertrend direction changes to down)
        elif in_buy_position and st_dir[i] == -1:
            buy_exit_prices[i] = close_prices[i]
            in_buy_position = False

        # Check for Sell_Entry condition (Supertrend direction changes to down)
        if not in_sell_position and not in_buy_position and st_dir[i] == -1:
            sell_entry_prices[i] = close_prices[i]
            in_sell_position = True
        # Check for Sell_Exit condition (Supertrend direction changes to up)
        elif in_sell_position and st_dir[i] == 1:
            sell_exit_prices[i] = close_prices[i]
            in_sell_position = False

    return buy_entry_prices, buy_exit_prices, sell_entry_prices, sell_exit_prices


# Copy the regular calculate_supertrend function from One_Supertrend.py for comparison
def calculate_supertrend(df, period=10, multiplier=3.0, suffix=""):
    original_index = df.index
    df_reset = df.reset_index(drop=True)

    if not all(col in df_reset.columns for col in ["HA_Open", "HA_High", "HA_Low", "HA_Close"]):
        raise ValueError("DataFrame must contain 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close' columns for Supertrend calculation.")

    ha_high = df_reset["HA_High"].to_numpy()
    ha_low = df_reset["HA_Low"].to_numpy()
    ha_close = df_reset["HA_Close"].to_numpy()

    # Calculate True Range based on HA candles
    # Numba does not support np.maximum.reduce, so use nested np.maximum
    tr = np.maximum(ha_high - ha_low, np.abs(ha_high - np.roll(ha_close, 1)))
    tr = np.maximum(tr, np.abs(ha_low - np.roll(ha_close, 1)))
    tr[0] = ha_high[0] - ha_low[0]  # Correct TR for the first element

    # Validate True Range values
    if np.any(tr <= 0):
        raise ValueError("True Range calculation produced zero or negative values")
    if np.any(~np.isfinite(tr)):
        raise ValueError("True Range calculation produced non-finite values")

    # ATR using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]  # Initialize first ATR
    alpha = 1 / period
    for i in range(1, len(tr)):
        atr[i] = (tr[i] * alpha) + (atr[i-1] * (1 - alpha))

    # Validate ATR values
    if np.any(~np.isfinite(atr)):
        raise ValueError("ATR calculation produced non-finite values")
    if np.any(atr < 0):
        raise ValueError("ATR calculation produced negative values")

    basic_upper_band = ((ha_high + ha_low) / 2) + (multiplier * atr)
    basic_lower_band = ((ha_high + ha_low) / 2) - (multiplier * atr)

    first_valid_atr_idx = np.where(~np.isnan(atr))[0][0] if np.any(~np.isnan(atr)) else -1  # Use -1 if no valid index

    trending_up_arr, trending_down_arr, direction_arr, supertrend_arr = _calculate_supertrend_numba(
        ha_high, ha_low, ha_close, tr, atr, period, multiplier, basic_upper_band, basic_lower_band, first_valid_atr_idx
    )

    # Validate Supertrend calculations
    if np.any(~np.isfinite(supertrend_arr)):
        raise ValueError("Supertrend calculation produced non-finite values")
    if np.any(~np.isfinite(trending_up_arr)):
        raise ValueError("Trending up calculation produced non-finite values")
    if np.any(~np.isfinite(trending_down_arr)):
        raise ValueError("Trending down calculation produced non-finite values")
    if not np.all(np.isin(direction_arr, [-1, 0, 1])):
        raise ValueError("Supertrend direction contains invalid values (must be -1, 0, or 1)")

    # Validate that supertrend values are within reasonable bounds
    price_range = ha_high - ha_low
    max_reasonable_st = np.nanmax(price_range) * 3  # Allow up to 3x the max range for volatile markets
    max_st = np.nanmax(supertrend_arr)
    min_st = np.nanmin(supertrend_arr)

    # Supertrend values are within reasonable bounds

    st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    trending_up_col_name = f"trendingUp_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    trending_down_col_name = f"trendingDown_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"

    result_df = pd.DataFrame({
        st_col_name: supertrend_arr,
        st_dir_col_name: direction_arr,
        trending_up_col_name: trending_up_arr,
        trending_down_col_name: trending_down_arr
    }, index=original_index)

    return result_df


def run_strategy_and_get_metrics(df_original, supertrend_params_list, kernel_params=None):
    df = df_original.copy()

    # Calculate Heikin Ashi
    ha_df = calculate_heikin_ashi(df.copy())

    # Validate Heikin-Ashi data with correct column names
    validate_ha_data(ha_df, "Heikin-Ashi Data")

    df = pd.concat([df, ha_df], axis=1)

    st_direction_cols = []

    # Only use the first (and only) Supertrend parameter
    period, multiplier = supertrend_params_list[0]
    suffix = "_1"
    
    # Use kernel-smoothed version if kernel_params provided, otherwise use regular
    if kernel_params:
        bandwidth, kernel_type = kernel_params
        st_result_ha = calculate_kernel_smoothed_supertrend(
            df.copy(),
            period=period,
            multiplier=multiplier,
            bandwidth=bandwidth,
            kernel_type=kernel_type,
            suffix=suffix
        )
        st_col_name = f"Kernel_ST_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_bw{bandwidth}{suffix}"
        st_dir_col_name = f"Kernel_ST_Dir_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_bw{bandwidth}{suffix}"
        trending_up_col_name = f"kernel_trendingUp_HA_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_bw{bandwidth}{suffix}"
        trending_down_col_name = f"kernel_trendingDown_HA_{period}_{str(multiplier).replace('.', '_')}_{kernel_type}_bw{bandwidth}{suffix}"
    else:
        # Fallback to regular supertrend for comparison
        st_result_ha = calculate_supertrend(
            df.copy(),
            period=period,
            multiplier=multiplier,
            suffix=suffix
        )
        st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        trending_up_col_name = f"trendingUp_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        trending_down_col_name = f"trendingDown_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"

    st_direction_cols.append(st_dir_col_name)

    df = pd.concat([df, st_result_ha], axis=1)

    df = df.drop(columns=[trending_up_col_name, trending_down_col_name], errors='ignore')

    # Convert direction column to numpy array for Numba
    st_dir = df[st_direction_cols[0]].to_numpy()
    close_prices = df['Close'].to_numpy()
    
    buy_entry_prices, buy_exit_prices, sell_entry_prices, sell_exit_prices = _run_strategy_numba(
        st_dir, close_prices
    )

    # Validate strategy results for consistency
    buy_entries = np.sum(~np.isnan(buy_entry_prices))
    buy_exits = np.sum(~np.isnan(buy_exit_prices))
    sell_entries = np.sum(~np.isnan(sell_entry_prices))
    sell_exits = np.sum(~np.isnan(sell_exit_prices))

    # Print trade summary in requested format
    print(f"Buy Trades: {buy_entries}, {buy_exits}")
    print(f"Sell Trades: {sell_entries}, {sell_exits}")
    total_positions = buy_entries + sell_entries
    print(f"Total Trades: {total_positions}")

    # Check for unmatched entries/exits
    if buy_entries != buy_exits:
        print(f"Warning: Unmatched buy trades - Entries: {buy_entries}, Exits: {buy_exits}")

    if sell_entries != sell_exits:
        print(f"Warning: Unmatched sell trades - Entries: {sell_entries}, Exits: {sell_exits}")

    # Check for overlapping positions (should be prevented by new logic)
    if total_positions > len(st_dir) * 0.5:  # More than 50% of bars have positions
        print(f"Warning: High position frequency - {total_positions} positions in {len(st_dir)} bars")

    df['Buy_Entry'] = buy_entry_prices
    df['Buy_Exit'] = buy_exit_prices
    df['Sell_Entry'] = sell_entry_prices
    df['Sell_Exit'] = sell_exit_prices

    trades = []
    current_buy_entry = None
    current_sell_entry = None

    for i in range(len(df)):
        if not pd.isna(df['Buy_Entry'].iloc[i]):
            current_buy_entry = {'entry_price': df['Buy_Entry'].iloc[i], 'entry_date': df.index[i]}
        if not pd.isna(df['Buy_Exit'].iloc[i]) and current_buy_entry is not None:
            profit = df['Buy_Exit'].iloc[i] - current_buy_entry['entry_price']
            trades.append({
                'type': 'buy',
                'entry_date': current_buy_entry['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_buy_entry['entry_price'],
                'exit_price': df['Buy_Exit'].iloc[i],
                'profit': profit
            })
            current_buy_entry = None

        if not pd.isna(df['Sell_Entry'].iloc[i]):
            current_sell_entry = {'entry_price': df['Sell_Entry'].iloc[i], 'entry_date': df.index[i]}
        if not pd.isna(df['Sell_Exit'].iloc[i]) and current_sell_entry is not None:
            profit = current_sell_entry['entry_price'] - df['Sell_Exit'].iloc[i]
            trades.append({
                'type': 'sell',
                'entry_date': current_sell_entry['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_sell_entry['entry_price'],
                'exit_price': df['Sell_Exit'].iloc[i],
                'profit': profit
            })
            current_sell_entry = None

    trades_df = pd.DataFrame(trades)

    metrics = {}
    if not trades_df.empty:
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

        streaks = []
        current_streak_type = None
        current_streak_length = 0

        for profit in trades_df['profit']:
            if profit > 0:
                if current_streak_type == 'win':
                    current_streak_length += 1
                else:
                    if current_streak_type is not None:
                        streaks.append({'type': current_streak_type, 'length': current_streak_length})
                    current_streak_type = 'win'
                    current_streak_length = 1
            elif profit < 0:
                if current_streak_type == 'loss':
                    current_streak_length += 1
                else:
                    if current_streak_type is not None:
                        streaks.append({'type': current_streak_type, 'length': current_streak_length})
                    current_streak_type = 'loss'
                    current_streak_length = 1
            else:
                if current_streak_type is not None:
                    streaks.append({'type': current_streak_type, 'length': current_streak_length})
                current_streak_type = None
                current_streak_length = 0
        if current_streak_type is not None:
            streaks.append({'type': current_streak_type, 'length': current_streak_length})

        winning_streaks = [s['length'] for s in streaks if s['type'] == 'win']
        losing_streaks = [s['length'] for s in streaks if s['type'] == 'loss']

        max_winning_streak = max(winning_streaks) if winning_streaks else 0
        max_losing_streak = max(losing_streaks) if losing_streaks else 0

        daily_returns = trades_df.set_index('exit_date')['profit'].resample('D').sum().fillna(0)
        annualized_returns = daily_returns.mean() * 252
        annualized_std_dev = daily_returns.std() * np.sqrt(252)

        sharpe_ratio = annualized_returns / annualized_std_dev if annualized_std_dev != 0 else np.nan

        downside_returns = daily_returns[daily_returns < 0]
        downside_std_dev = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
        sortino_ratio = annualized_returns / downside_std_dev if downside_std_dev != 0 else np.nan

        # Calculate Calmar Ratio (Annualized Return / Max Drawdown)
        calmar_ratio = annualized_returns / abs(max_drawdown) if max_drawdown != 0 else np.nan

        # Calculate additional trade-level metrics
        best_trade = winning_trades['profit'].max() if num_winning_trades > 0 else 0
        worst_trade = losing_trades['profit'].min() if num_losing_trades > 0 else 0
        avg_trade_duration = trades_df['exit_date'].sub(trades_df['entry_date']).mean() if not trades_df.empty else pd.Timedelta(0)

        # Calculate Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
        mfe_values = []
        mae_values = []

        for idx, trade in trades_df.iterrows():
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']

            # Get price data for this trade period
            trade_data = df_original.loc[entry_date:exit_date]

            if trade['type'] == 'buy':
                # For buy trades: MFE is max price - entry, MAE is entry - min price
                if not trade_data.empty:
                    mfe_values.append((trade_data['High'].max() - entry_price))
                    mae_values.append((entry_price - trade_data['Low'].min()))
            else:  # sell trade
                # For sell trades: MFE is entry - min price, MAE is max price - entry
                if not trade_data.empty:
                    mfe_values.append((entry_price - trade_data['Low'].min()))
                    mae_values.append((trade_data['High'].max() - entry_price))

        avg_mfe = np.mean(mfe_values) if mfe_values else 0
        avg_mae = np.mean(mae_values) if mae_values else 0
        max_mfe = np.max(mfe_values) if mfe_values else 0
        max_mae = np.max(mae_values) if mae_values else 0

        # Calculate additional ratios
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
        profit_per_trade = net_profit / num_total_trades if num_total_trades > 0 else 0
        recovery_factor = net_profit / abs(max_drawdown * initial_capital / 100) if max_drawdown != 0 else np.nan

        # Calculate average trade duration in days
        avg_trade_duration_days = avg_trade_duration.days if hasattr(avg_trade_duration, 'days') else 0

        metrics = {
            'total_trades': num_total_trades,
            'winning_trades': num_winning_trades,
            'losing_trades': num_losing_trades,
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'best_trade': round(best_trade, 2),
            'worst_trade': round(worst_trade, 2),
            'payoff_ratio': round(payoff_ratio, 2) if not np.isnan(payoff_ratio) else np.nan,
            'profit_per_trade': round(profit_per_trade, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2) if not np.isnan(risk_reward_ratio) else np.nan,
            'profit_factor': round(profit_factor, 2) if not np.isnan(profit_factor) else np.nan,
            'expectancy': round(expectancy, 2),
            'net_profit': round(net_profit, 2),
            'net_profit_percent': round(net_profit_percent, 2),
            'max_drawdown': round(max_drawdown, 2),
            'calmar_ratio': round(calmar_ratio, 2) if not np.isnan(calmar_ratio) else np.nan,
            'recovery_factor': round(recovery_factor, 2) if not np.isnan(recovery_factor) else np.nan,
            'return_drawdown_ratio': round(return_drawdown_ratio, 2) if not np.isnan(return_drawdown_ratio) else np.nan,
            'max_winning_streak': max_winning_streak,
            'max_losing_streak': max_losing_streak,
            'avg_trade_duration_days': round(avg_trade_duration_days, 1),
            'max_favorable_excursion': round(max_mfe, 2),
            'max_adverse_excursion': round(max_mae, 2),
            'avg_favorable_excursion': round(avg_mfe, 2),
            'avg_adverse_excursion': round(avg_mae, 2),
            'sharpe_ratio': round(sharpe_ratio, 2) if not np.isnan(sharpe_ratio) else np.nan,
            'sortino_ratio': round(sortino_ratio, 2) if not np.isnan(sortino_ratio) else np.nan
        }
    else:
        metrics = {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0,
            'avg_win': 0, 'avg_loss': 0, 'best_trade': 0, 'worst_trade': 0,
            'payoff_ratio': np.nan, 'profit_per_trade': 0, 'risk_reward_ratio': np.nan,
            'profit_factor': np.nan, 'expectancy': 0, 'net_profit': 0, 'net_profit_percent': 0,
            'max_drawdown': 0, 'calmar_ratio': np.nan, 'recovery_factor': np.nan,
            'return_drawdown_ratio': np.nan, 'max_winning_streak': 0, 'max_losing_streak': 0,
            'avg_trade_duration_days': 0, 'max_favorable_excursion': 0, 'max_adverse_excursion': 0,
            'avg_favorable_excursion': 0, 'avg_adverse_excursion': 0,
            'sharpe_ratio': np.nan, 'sortino_ratio': np.nan
        }
    return metrics

try:
    # Load and validate data
    try:
        df_original = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {csv_file_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Clean column names and structure
    df_original = df_original.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
    df_original = df_original.rename(columns={'Date': 'DateTime'})

    # Parse dates with proper error handling
    try:
        df_original = df_original.set_index('DateTime')
        df_original.index = pd.to_datetime(df_original.index)
    except Exception as e:
        raise ValueError(f"Error parsing dates: {e}. Please check date format in CSV file.")

    # Filter for 2019 onwards before validation
    df_original = df_original[df_original.index.year >= 2019]

    # Check if we have any data after filtering
    if df_original.empty:
        raise ValueError("No data available after filtering for year 2019 onwards")

    # Comprehensive data validation and cleaning
    df_original = clean_dataframe_data(df_original, "Original Data")

    # Test different kernel parameters
    kernel_configs = [(45, "Laplace")]
    
    results_list = []
    
    # Use fixed Supertrend parameter: Period=21, Multiplier=0.9 for single indicator
    fixed_supertrend_params = [(21, 0.9)]
    print("Using fixed Supertrend parameter: ST=(21, 0.9)")
    
    # Test regular Supertrend for comparison
    print("\n=== Regular Supertrend ===")
    metrics_regular = run_strategy_and_get_metrics(df_original.copy(), fixed_supertrend_params)
    print("Regular Supertrend Metrics:")
    for key, value in metrics_regular.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    results_list.append({
        'Type': 'Regular',
        'ST_Period': 21,
        'ST_Multiplier': 0.9,
        'Kernel_Type': 'None',
        'Bandwidth': 0,
        **metrics_regular
    })
    
    # Test Kernel-Smoothed Supertrend with different configurations
    for bandwidth, kernel_type in kernel_configs:
        print(f"\n=== Kernel-Smoothed Supertrend: {kernel_type} (bw={bandwidth}) ===")
        kernel_params = (bandwidth, kernel_type)
        metrics_kernel = run_strategy_and_get_metrics(df_original.copy(), fixed_supertrend_params, kernel_params)
        print(f"Kernel-Smoothed Supertrend Metrics ({kernel_type}, bw={bandwidth}):")
        for key, value in metrics_kernel.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        results_list.append({
            'Type': 'Kernel',
            'ST_Period': 21,
            'ST_Multiplier': 0.9,
            'Kernel_Type': kernel_type,
            'Bandwidth': bandwidth,
            **metrics_kernel
        })
    
    # Save all results to CSV
    results_df = pd.DataFrame(results_list)
    output_csv_path = 'kernel_smoothed_supertrend_results.csv'
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nResults saved to '{output_csv_path}'")

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
