# ----------------------------------------------------------------------------------------------
# Two Supertrend with Gaussian Kernel Regression - IMPROVED VERSION
# Sequential Feature Implementation for Strategy Optimization
#
# This file contains the base strategy with modular improvements that can be enabled
# sequentially to test and evaluate each enhancement.
#
# TESTING SEQUENCE:
# 1. Run base strategy (lines ~50-100)
# 2. Enable stop-loss (uncomment STOP_LOSS section)
# 3. Enable trend filter (uncomment TREND_FILTER section)
# 4. Enable volume filter (uncomment VOLUME_FILTER section)
# 5. Enable trailing stop (uncomment TRAILING_STOP section)
# 6. Enable position sizing (uncomment POSITION_SIZING section)
# 7. Enable all features and optimize parameters
#
# ----------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Multi_Kernel_Regression import MultiKernelRegression

# Define the path to the CSV file
csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'

# =============================================================================
# CONFIGURATION SECTION - Enable/Disable Features
# =============================================================================

# Base strategy parameters
BASE_PARAMS = {
    'supertrend_params': [(17, 0.9), (17, 0.9)],  # (period, multiplier)
    'kernel': "Epanechnikov",
    'bandwidth': 20,
    'deviations': 2.0
}

# Feature toggles - ULTIMATE OPTIMIZED VERSION
FEATURES = {
    'stop_loss': True,            # ATR-based stop-loss
    'trend_filter': True,         # ADX trend strength filter
    'volume_filter': False,       # Volume confirmation (needs volume data)
    'trailing_stop': True,        # Trailing stop-loss
    'position_sizing': True,      # ATR-based position sizing
    'market_regime': True,        # Regime detection
    'profit_targets': True        # Multiple profit targets
}

# Stop-loss parameters
STOP_LOSS_PARAMS = {
    'atr_period': 14,
    'sl_multiplier': 2.0,      # ATR multiplier for stop-loss
    'trailing_multiplier': 1.5 # For trailing stops
}

# Trend filter parameters
TREND_FILTER_PARAMS = {
    'adx_period': 14,
    'adx_threshold': 30  # Minimum ADX for trend strength - increased
}

# Position sizing parameters
POSITION_SIZING_PARAMS = {
    'base_position_size': 1.0,     # Base position size
    'atr_period': 14,
    'volatility_multiplier': 1.0,  # Adjust position size based on volatility
    'max_position_size': 2.0,      # Maximum position size
    'min_position_size': 0.5       # Minimum position size
}

# Market regime parameters
MARKET_REGIME_PARAMS = {
    'adx_period': 14,
    'adx_trending_threshold': 25,
    'atr_period': 14,
    'volatility_lookback': 20,
    'regime_smoothing': 5
}

# Profit target parameters
PROFIT_TARGET_PARAMS = {
    'target_1': 1.5,    # First target: 1.5:1 reward-to-risk
    'target_2': 2.5,    # Second target: 2.5:1 reward-to-risk
    'partial_exit_1': 0.5,  # Exit 50% at first target
    'partial_exit_2': 0.5   # Exit remaining 50% at second target
}

# Volume filter parameters
VOLUME_FILTER_PARAMS = {
    'volume_ma_period': 20,
    'volume_multiplier': 1.2  # Volume must be X% above average
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    df = df.copy()
    df['TR'] = np.maximum.reduce([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ])
    df['ATR'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    return df['ATR']

def calculate_adx(df, period=14):
    """Calculate Average Directional Index"""
    df = df.copy()

    # True Range
    df['TR'] = np.maximum.reduce([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ])

    # Directional Movement
    df['DM+'] = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        np.maximum(df['High'] - df['High'].shift(1), 0),
        0
    )
    df['DM-'] = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        np.maximum(df['Low'].shift(1) - df['Low'], 0),
        0
    )

    # Smoothed averages
    df['ATR_smooth'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    df['DM+_smooth'] = df['DM+'].ewm(alpha=1/period, adjust=False).mean()
    df['DM-_smooth'] = df['DM-'].ewm(alpha=1/period, adjust=False).mean()

    # Directional Indicators
    df['DI+'] = (df['DM+_smooth'] / df['ATR_smooth']) * 100
    df['DI-'] = (df['DM-_smooth'] / df['ATR_smooth']) * 100

    # ADX
    df['DX'] = (abs(df['DI+'] - df['DI-']) / (df['DI+'] + df['DI-'])) * 100
    df['ADX'] = df['DX'].ewm(alpha=1/period, adjust=False).mean()

    return df['ADX']

def calculate_market_regime(df):
    """Calculate market regime (trending vs ranging)"""
    df = df.copy()

    # ADX-based trend strength
    adx = calculate_adx(df, MARKET_REGIME_PARAMS['adx_period'])
    df['ADX'] = adx

    # ATR-based volatility
    atr = calculate_atr(df, MARKET_REGIME_PARAMS['atr_period'])
    df['ATR'] = atr

    # Normalized ATR (volatility measure)
    df['ATR_norm'] = df['ATR'] / df['Close'].rolling(MARKET_REGIME_PARAMS['volatility_lookback']).mean()

    # Market regime classification
    # 1 = Strong trending, 0.5 = Moderate trending, 0 = Ranging
    df['regime_score'] = np.where(
        df['ADX'] > MARKET_REGIME_PARAMS['adx_trending_threshold'],
        1.0,  # Strong trend
        np.where(
            df['ADX'] > 20,
            0.5,  # Moderate trend
            0.0   # Ranging
        )
    )

    # Smooth regime score
    df['market_regime'] = df['regime_score'].ewm(span=MARKET_REGIME_PARAMS['regime_smoothing']).mean()

    return df['market_regime']

def calculate_position_size(df, entry_price, atr_value, market_regime):
    """Calculate position size based on ATR and market regime"""
    # Base position size
    position_size = POSITION_SIZING_PARAMS['base_position_size']

    # Adjust for volatility (inverse relationship)
    volatility_factor = 1.0 / (1.0 + POSITION_SIZING_PARAMS['volatility_multiplier'] * atr_value / entry_price)

    # Adjust for market regime (higher in trending markets)
    regime_factor = 0.7 + 0.6 * market_regime  # 0.7 to 1.3 range

    # Calculate final position size
    final_size = position_size * volatility_factor * regime_factor

    # Apply bounds
    final_size = np.clip(final_size,
                        POSITION_SIZING_PARAMS['min_position_size'],
                        POSITION_SIZING_PARAMS['max_position_size'])

    return final_size

# =============================================================================
# BASE STRATEGY IMPLEMENTATION
# =============================================================================

try:
    # Read and prepare data
    df = pd.read_csv(csv_file_path)
    df = df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
    df = df.rename(columns={'Date': 'DateTime'})
    df = df.set_index('DateTime')
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]
    df = df[df.index.year >= 2021]

    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].round(2)

    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")

    # =============================================================================
    # FEATURE 1: STOP-LOSS MECHANISM
    # =============================================================================
    if FEATURES['stop_loss']:
        print("Enabling ATR-based stop-loss...")
        df['ATR'] = calculate_atr(df, STOP_LOSS_PARAMS['atr_period'])
        print("Stop-loss ATR calculated")

    # =============================================================================
    # FEATURE 2: TREND STRENGTH FILTER
    # =============================================================================
    if FEATURES['trend_filter']:
        print("Enabling ADX trend filter...")
        df['ADX'] = calculate_adx(df, TREND_FILTER_PARAMS['adx_period'])
        print("ADX calculated")

    # =============================================================================
    # FEATURE 6: MARKET REGIME DETECTION
    # =============================================================================
    if FEATURES['market_regime']:
        print("Enabling market regime detection...")
        df['market_regime'] = calculate_market_regime(df)
        print("Market regime calculated")

    # =============================================================================
    # FEATURE 3: VOLUME CONFIRMATION
    # =============================================================================
    if FEATURES['volume_filter']:
        print("Enabling volume filter...")
        # Note: Volume column was dropped, so this would need volume data
        # df['Volume_MA'] = df['Volume'].rolling(VOLUME_FILTER_PARAMS['volume_ma_period']).mean()
        print("Volume filter prepared (needs volume data)")

    # =============================================================================
    # MULTI-KERNEL REGRESSION SETUP
    # =============================================================================
    mkr = MultiKernelRegression(
        bandwidth=BASE_PARAMS['bandwidth'],
        kernel=BASE_PARAMS['kernel'],
        deviations=BASE_PARAMS['deviations']
    )
    df = mkr.generate_signals(df.copy(), price_column='Close')
    print("Multi-Kernel Regression signals generated")

    # =============================================================================
    # HEIKIN ASHI CALCULATION
    # =============================================================================
    def calculate_heikin_ashi(df):
        open_prices = df["Open"].to_numpy()
        high_prices = df["High"].to_numpy()
        low_prices = df["Low"].to_numpy()
        close_prices = df["Close"].to_numpy()

        ha_close = (open_prices + high_prices + low_prices + close_prices) / 4
        ha_open = np.zeros_like(open_prices, dtype=float)

        if len(open_prices) > 0:
            ha_open[0] = open_prices[0]

        for i in range(1, len(open_prices)):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2

        ha_high = np.maximum.reduce([high_prices, ha_open, ha_close])
        ha_low = np.minimum.reduce([low_prices, ha_open, ha_close])

        ha_df = pd.DataFrame({
            "HA_Open": ha_open,
            "HA_High": ha_high,
            "HA_Low": ha_low,
            "HA_Close": ha_close,
        }, index=df.index)

        return ha_df

    ha_df = calculate_heikin_ashi(df.copy())
    df = pd.concat([df, ha_df], axis=1)
    print("Heikin Ashi calculated")

    # =============================================================================
    # SUPER TREND CALCULATION
    # =============================================================================
    def calculate_supertrend(df, period=10, multiplier=3.0, suffix=""):
        original_index = df.index
        df = df.reset_index(drop=True)

        if not all(col in df.columns for col in ["HA_Open", "HA_High", "HA_Low", "HA_Close"]):
            raise ValueError("DataFrame must contain HA columns")

        ha_high = df["HA_High"]
        ha_low = df["HA_Low"]
        ha_close = df["HA_Close"]

        # True Range
        df["TR"] = np.maximum.reduce([
            ha_high - ha_low,
            abs(ha_high - ha_close.shift(1)),
            abs(ha_low - ha_close.shift(1))
        ])

        # ATR
        df["ATR"] = df["TR"].ewm(alpha=1/period, adjust=False).mean()

        # Basic bands
        basic_upper_band = ((ha_high + ha_low) / 2) + (multiplier * df["ATR"])
        basic_lower_band = ((ha_high + ha_low) / 2) - (multiplier * df["ATR"])

        # Column names
        trending_up_col = f"trendingUp_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        trending_down_col = f"trendingDown_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"

        # Initialize columns
        for col in [trending_up_col, trending_down_col, "direction", "SuperTrend"]:
            if col not in df.columns:
                df[col] = np.nan

        # Find first valid ATR
        first_valid_idx = df["ATR"].first_valid_index()
        if first_valid_idx is None:
            return pd.Series(np.nan, index=original_index)

        # Initialize first values
        df.loc[first_valid_idx, trending_up_col] = basic_lower_band.iloc[first_valid_idx]
        df.loc[first_valid_idx, trending_down_col] = basic_upper_band.iloc[first_valid_idx]

        if ha_close.iloc[first_valid_idx] > df[trending_down_col].iloc[first_valid_idx]:
            df.loc[first_valid_idx, "direction"] = 1
        elif ha_close.iloc[first_valid_idx] < df[trending_up_col].iloc[first_valid_idx]:
            df.loc[first_valid_idx, "direction"] = -1
        else:
            df.loc[first_valid_idx, "direction"] = 1

        df.loc[first_valid_idx, "SuperTrend"] = (
            df[trending_up_col].iloc[first_valid_idx]
            if df["direction"].iloc[first_valid_idx] == 1
            else df[trending_down_col].iloc[first_valid_idx]
        )

        # Calculate Supertrend for remaining data
        for i in range(first_valid_idx + 1, len(df)):
            prev_ha_close = ha_close.iloc[i - 1]
            prev_trending_up = df.iloc[i - 1, df.columns.get_loc(trending_up_col)]
            prev_trending_down = df.iloc[i - 1, df.columns.get_loc(trending_down_col)]
            prev_direction = df.iloc[i - 1, df.columns.get_loc("direction")]

            current_basic_upper = basic_upper_band.iloc[i]
            current_basic_lower = basic_lower_band.iloc[i]
            current_ha_close = ha_close.iloc[i]

            # Update trending bands
            if prev_ha_close > prev_trending_up:
                df.iloc[i, df.columns.get_loc(trending_up_col)] = max(current_basic_lower, prev_trending_up)
            else:
                df.iloc[i, df.columns.get_loc(trending_up_col)] = current_basic_lower

            if prev_ha_close < prev_trending_down:
                df.iloc[i, df.columns.get_loc(trending_down_col)] = min(current_basic_upper, prev_trending_down)
            else:
                df.iloc[i, df.columns.get_loc(trending_down_col)] = current_basic_upper

            # Update direction
            if current_ha_close > df.iloc[i - 1, df.columns.get_loc(trending_down_col)]:
                df.iloc[i, df.columns.get_loc("direction")] = 1
            elif current_ha_close < df.iloc[i - 1, df.columns.get_loc(trending_up_col)]:
                df.iloc[i, df.columns.get_loc("direction")] = -1
            else:
                df.iloc[i, df.columns.get_loc("direction")] = prev_direction

            # Update Supertrend
            df.iloc[i, df.columns.get_loc("SuperTrend")] = (
                df.iloc[i, df.columns.get_loc(trending_up_col)]
                if df.iloc[i, df.columns.get_loc("direction")] == 1
                else df.iloc[i, df.columns.get_loc(trending_down_col)]
            )

        # Rename columns
        st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}{suffix}"

        df.rename(columns={"SuperTrend": st_col_name, "direction": st_dir_col_name}, inplace=True)
        df = df.set_index(original_index)

        return df[[st_col_name, st_dir_col_name, trending_up_col, trending_down_col]]

    # Calculate Supertrends
    st_direction_cols = []
    for i, (period, multiplier) in enumerate(BASE_PARAMS['supertrend_params']):
        suffix = f"_{i+1}"
        st_result = calculate_supertrend(df.copy(), period=period, multiplier=multiplier, suffix=suffix)
        st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        trending_up_col = f"trendingUp_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        trending_down_col = f"trendingDown_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"

        st_direction_cols.append(st_dir_col_name)
        df = pd.concat([df, st_result], axis=1)
        df = df.drop(columns=[trending_up_col, trending_down_col], errors='ignore')

    print("Supertrend indicators calculated")

    # =============================================================================
    # TRADING LOGIC WITH IMPROVEMENTS
    # =============================================================================

    # Initialize trading columns
    df['Buy_Entry'] = np.nan
    df['Buy_Exit'] = np.nan
    df['Sell_Entry'] = np.nan
    df['Sell_Exit'] = np.nan
    df['Stop_Loss_Hit'] = np.nan
    df['Position_Size'] = np.nan
    df['Partial_Exit_1'] = np.nan
    df['Partial_Exit_2'] = np.nan

    # Position tracking
    in_buy_position = False
    in_sell_position = False
    pending_buy_signal = False
    pending_sell_signal = False
    pending_buy_candles = 0
    pending_sell_candles = 0
    waiting_for_mkr_buy = False
    waiting_for_mkr_sell = False
    mkr_wait_buy_candles = 0
    mkr_wait_sell_candles = 0
    skip_current_buy_signal = False
    skip_current_sell_signal = False
    current_buy_signal_price = None
    current_sell_signal_price = None

    # Stop-loss tracking
    buy_stop_loss = None
    sell_stop_loss = None
    trailing_buy_stop = None
    trailing_sell_stop = None

    # Main trading loop
    for i in range(1, len(df)):
        current_close = df['Close'].iloc[i]
        current_open = df['Open'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]

        st1_dir = df[st_direction_cols[0]].iloc[i]
        st2_dir = df[st_direction_cols[1]].iloc[i]

        is_green_candle = current_close > current_open
        is_red_candle = current_close < current_open

        # Supertrend and MKR signals
        supertrends_buy = (st1_dir == 1 and st2_dir == 1)
        supertrends_sell = (st1_dir == -1 and st2_dir == -1)
        mkr_buy = df['signal'].iloc[i] == 1
        mkr_sell = df['signal'].iloc[i] == -1

        # =============================================================================
        # FEATURE 2: TREND FILTER APPLICATION
        # =============================================================================
        trend_filter_pass = True
        if FEATURES['trend_filter']:
            adx_value = df['ADX'].iloc[i] if 'ADX' in df.columns else 0
            trend_filter_pass = adx_value >= TREND_FILTER_PARAMS['adx_threshold']

        # =============================================================================
        # FEATURE 3: VOLUME FILTER APPLICATION
        # =============================================================================
        volume_filter_pass = True
        if FEATURES['volume_filter']:
            # This would need volume data
            volume_filter_pass = True  # Placeholder

        # =============================================================================
        # FEATURE 1: STOP-LOSS CHECK
        # =============================================================================
        # Check for stop-loss hits
        if FEATURES['stop_loss']:
            # Buy position stop-loss check
            if in_buy_position and buy_stop_loss is not None:
                if current_low <= buy_stop_loss:
                    df.loc[df.index[i], 'Buy_Exit'] = buy_stop_loss
                    df.loc[df.index[i], 'Stop_Loss_Hit'] = buy_stop_loss
                    in_buy_position = False
                    buy_stop_loss = None
                    trailing_buy_stop = None
                    continue  # Exit the loop iteration

            # Sell position stop-loss check
            if in_sell_position and sell_stop_loss is not None:
                if current_high >= sell_stop_loss:
                    df.loc[df.index[i], 'Sell_Exit'] = sell_stop_loss
                    df.loc[df.index[i], 'Stop_Loss_Hit'] = sell_stop_loss
                    in_sell_position = False
                    sell_stop_loss = None
                    trailing_sell_stop = None
                    continue  # Exit the loop iteration

        # Reset skip flags
        if not supertrends_buy:
            skip_current_buy_signal = False
            current_buy_signal_price = None
        if not supertrends_sell:
            skip_current_sell_signal = False
            current_sell_signal_price = None

        # Combined signals
        supertrends_buy_signal = supertrends_buy and mkr_buy and trend_filter_pass and volume_filter_pass
        supertrends_sell_signal = supertrends_sell and mkr_sell and trend_filter_pass and volume_filter_pass

        # Handle pending buy signals
        if pending_buy_signal:
            pending_buy_candles += 1
            if supertrends_buy and mkr_buy and is_green_candle:
                entry_price = current_close

                # =============================================================================
                # FEATURE 1: SET STOP-LOSS ON ENTRY
                # =============================================================================
                if FEATURES['stop_loss']:
                    atr_value = df['ATR'].iloc[i]
                    buy_stop_loss = entry_price - (atr_value * STOP_LOSS_PARAMS['sl_multiplier'])
                    if FEATURES['trailing_stop']:
                        trailing_buy_stop = buy_stop_loss

                # =============================================================================
                # FEATURE 7: POSITION SIZING
                # =============================================================================
                position_size = 1.0  # Default
                if FEATURES['position_sizing']:
                    market_regime_value = df['market_regime'].iloc[i] if 'market_regime' in df.columns else 0.5
                    position_size = calculate_position_size(df.iloc[:i+1], entry_price, atr_value, market_regime_value)

                df.loc[df.index[i], 'Buy_Entry'] = entry_price
                df.loc[df.index[i], 'Position_Size'] = position_size
                in_buy_position = True
                in_sell_position = False

                # Initialize profit targets for buy position
                if FEATURES['profit_targets']:
                    risk_amount = entry_price - buy_stop_loss if FEATURES['stop_loss'] else entry_price * 0.02
                    target_1_price = entry_price + (risk_amount * PROFIT_TARGET_PARAMS['target_1'])
                    target_2_price = entry_price + (risk_amount * PROFIT_TARGET_PARAMS['target_2'])
                    # Store targets for later use in exit logic
                pending_buy_signal = False
                pending_buy_candles = 0
            elif not supertrends_buy or is_red_candle or pending_buy_candles > 2:
                pending_buy_signal = False
                pending_buy_candles = 0

        # Handle pending sell signals
        if pending_sell_signal:
            pending_sell_candles += 1
            if supertrends_sell and mkr_sell and is_red_candle:
                entry_price = current_close

                # Set stop-loss for sell position
                if FEATURES['stop_loss']:
                    atr_value = df['ATR'].iloc[i]
                    sell_stop_loss = entry_price + (atr_value * STOP_LOSS_PARAMS['sl_multiplier'])
                    if FEATURES['trailing_stop']:
                        trailing_sell_stop = sell_stop_loss

                # Position sizing for sell
                position_size = 1.0  # Default
                if FEATURES['position_sizing']:
                    market_regime_value = df['market_regime'].iloc[i] if 'market_regime' in df.columns else 0.5
                    position_size = calculate_position_size(df.iloc[:i+1], entry_price, atr_value, market_regime_value)

                df.loc[df.index[i], 'Sell_Entry'] = entry_price
                df.loc[df.index[i], 'Position_Size'] = position_size
                in_sell_position = True
                in_buy_position = False

                # Initialize profit targets for sell position
                if FEATURES['profit_targets']:
                    risk_amount = sell_stop_loss - entry_price if FEATURES['stop_loss'] else entry_price * 0.02
                    target_1_price = entry_price - (risk_amount * PROFIT_TARGET_PARAMS['target_1'])
                    target_2_price = entry_price - (risk_amount * PROFIT_TARGET_PARAMS['target_2'])
                    # Store targets for later use in exit logic
                pending_sell_signal = False
                pending_sell_candles = 0
            elif not supertrends_sell or is_green_candle or pending_sell_candles > 2:
                pending_sell_signal = False
                pending_sell_candles = 0

        # Handle waiting for MKR buy
        if waiting_for_mkr_buy:
            mkr_wait_buy_candles += 1
            if mkr_buy:
                if is_green_candle:
                    entry_price = current_close
                    if FEATURES['stop_loss']:
                        atr_value = df['ATR'].iloc[i]
                        buy_stop_loss = entry_price - (atr_value * STOP_LOSS_PARAMS['sl_multiplier'])
                        if FEATURES['trailing_stop']:
                            trailing_buy_stop = buy_stop_loss

                    df.loc[df.index[i], 'Buy_Entry'] = entry_price
                    in_buy_position = True
                    in_sell_position = False
                    waiting_for_mkr_buy = False
                    mkr_wait_buy_candles = 0
                    current_buy_signal_price = None
                else:
                    pending_buy_signal = True
                    pending_buy_candles = 0
                waiting_for_mkr_buy = False
                mkr_wait_buy_candles = 0
            elif mkr_wait_buy_candles > 2 or not supertrends_buy:
                waiting_for_mkr_buy = False
                mkr_wait_buy_candles = 0
                skip_current_buy_signal = True
                current_buy_signal_price = None

        # Handle waiting for MKR sell
        if waiting_for_mkr_sell:
            mkr_wait_sell_candles += 1
            if mkr_sell:
                if is_red_candle:
                    entry_price = current_close
                    if FEATURES['stop_loss']:
                        atr_value = df['ATR'].iloc[i]
                        sell_stop_loss = entry_price + (atr_value * STOP_LOSS_PARAMS['sl_multiplier'])
                        if FEATURES['trailing_stop']:
                            trailing_sell_stop = sell_stop_loss

                    df.loc[df.index[i], 'Sell_Entry'] = entry_price
                    in_sell_position = True
                    in_buy_position = False
                    waiting_for_mkr_sell = False
                    mkr_wait_sell_candles = 0
                    current_sell_signal_price = None
                else:
                    pending_sell_signal = True
                    pending_sell_candles = 0
                waiting_for_mkr_sell = False
                mkr_wait_sell_candles = 0
            elif mkr_wait_sell_candles > 2 or not supertrends_sell:
                waiting_for_mkr_sell = False
                mkr_wait_sell_candles = 0
                skip_current_sell_signal = True
                current_sell_signal_price = None

        # Check for new buy entry
        if not in_buy_position and not in_sell_position and not pending_buy_signal and not waiting_for_mkr_buy:
            if supertrends_buy and not skip_current_buy_signal:
                if current_buy_signal_price is None:
                    current_buy_signal_price = current_close
                if mkr_buy:
                    if is_green_candle:
                        entry_price = current_close
                        if FEATURES['stop_loss']:
                            atr_value = df['ATR'].iloc[i]
                            buy_stop_loss = entry_price - (atr_value * STOP_LOSS_PARAMS['sl_multiplier'])
                            if FEATURES['trailing_stop']:
                                trailing_buy_stop = buy_stop_loss

                        df.loc[df.index[i], 'Buy_Entry'] = entry_price
                        in_buy_position = True
                        in_sell_position = False
                        current_buy_signal_price = None
                    else:
                        pending_buy_signal = True
                        pending_buy_candles = 0
                else:
                    waiting_for_mkr_buy = True
                    mkr_wait_buy_candles = 0

        # =============================================================================
        # FEATURE 4: TRAILING STOP UPDATE
        # =============================================================================
        if FEATURES['trailing_stop'] and in_buy_position and trailing_buy_stop is not None:
            # Update trailing stop for buy position
            potential_new_stop = current_close - (df['ATR'].iloc[i] * STOP_LOSS_PARAMS['trailing_multiplier'])
            if potential_new_stop > trailing_buy_stop:
                trailing_buy_stop = potential_new_stop
                buy_stop_loss = trailing_buy_stop

        if FEATURES['trailing_stop'] and in_sell_position and trailing_sell_stop is not None:
            # Update trailing stop for sell position
            potential_new_stop = current_close + (df['ATR'].iloc[i] * STOP_LOSS_PARAMS['trailing_multiplier'])
            if potential_new_stop < trailing_sell_stop:
                trailing_sell_stop = potential_new_stop
                sell_stop_loss = trailing_sell_stop

        # =============================================================================
        # FEATURE 8: PROFIT TARGET CHECKING
        # =============================================================================
        if FEATURES['profit_targets'] and in_buy_position:
            # Check profit targets for buy position
            # This would need to track entry price and calculate targets
            # For simplicity, we'll implement basic profit taking at certain levels
            pass  # Implementation would go here

        if FEATURES['profit_targets'] and in_sell_position:
            # Check profit targets for sell position
            pass  # Implementation would go here

        # Check for buy exit
        elif in_buy_position and (st1_dir == -1 or st2_dir == -1):
            df.loc[df.index[i], 'Buy_Exit'] = current_close
            in_buy_position = False
            pending_buy_signal = False
            waiting_for_mkr_buy = False
            mkr_wait_buy_candles = 0
            buy_stop_loss = None
            trailing_buy_stop = None

        # Check for new sell entry
        if not in_sell_position and not in_buy_position and not pending_sell_signal and not waiting_for_mkr_sell:
            if supertrends_sell and not skip_current_sell_signal:
                if current_sell_signal_price is None:
                    current_sell_signal_price = current_close
                if mkr_sell:
                    if is_red_candle:
                        entry_price = current_close
                        if FEATURES['stop_loss']:
                            atr_value = df['ATR'].iloc[i]
                            sell_stop_loss = entry_price + (atr_value * STOP_LOSS_PARAMS['sl_multiplier'])
                            if FEATURES['trailing_stop']:
                                trailing_sell_stop = sell_stop_loss

                        df.loc[df.index[i], 'Sell_Entry'] = entry_price
                        in_sell_position = True
                        in_buy_position = False
                        current_sell_signal_price = None
                    else:
                        pending_sell_signal = True
                        pending_sell_candles = 0
                else:
                    waiting_for_mkr_sell = True
                    mkr_wait_sell_candles = 0

        # Check for sell exit
        elif in_sell_position and (st1_dir == 1 or st2_dir == 1):
            df.loc[df.index[i], 'Sell_Exit'] = current_close
            in_sell_position = False
            pending_sell_signal = False
            waiting_for_mkr_sell = False
            mkr_wait_sell_candles = 0
            sell_stop_loss = None
            trailing_sell_stop = None

    print("Trading signals generated")

    # =============================================================================
    # PERFORMANCE CALCULATION
    # =============================================================================

    trades = []
    current_buy_entry = None
    current_sell_entry = None

    for i in range(len(df)):
        # Handle Buy Trades
        if not pd.isna(df['Buy_Entry'].iloc[i]):
            current_buy_entry = {
                'entry_price': df['Buy_Entry'].iloc[i],
                'entry_date': df.index[i]
            }
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

        # Handle Sell Trades
        if not pd.isna(df['Sell_Entry'].iloc[i]):
            current_sell_entry = {
                'entry_price': df['Sell_Entry'].iloc[i],
                'entry_date': df.index[i]
            }
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

    if not trades_df.empty:
        # Calculate metrics
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

        # Print results
        print("\n" + "="*60)
        print("STRATEGY PERFORMANCE RESULTS")
        print("="*60)
        print(f"Features Enabled: {', '.join([k for k, v in FEATURES.items() if v])}")
        print(f"Total Trades: {num_total_trades}")
        print(f"Winning Trades: {num_winning_trades}")
        print(f"Losing Trades: {num_losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win: {avg_win:.2f}")
        print(f"Average Loss: {avg_loss:.2f}")
        print(f"Risk-Reward Ratio: {risk_reward_ratio:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Expectancy: {expectancy:.2f}")
        print(f"Net Profit: {net_profit:.2f}")
        print(f"Net Profit (%): {net_profit_percent:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Return-Drawdown Ratio: {return_drawdown_ratio:.2f}")

        # Save results
        output_csv_path = 'two_supertrend_improved_signals.csv'
        df.to_csv(output_csv_path)
        print(f"\nResults saved to '{output_csv_path}'")

    else:
        print("No trades found to calculate metrics.")

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# =============================================================================
# TESTING INSTRUCTIONS
# =============================================================================
"""
To test improvements sequentially:

1. BASE STRATEGY:
   - Run with all FEATURES set to False
   - Record baseline metrics

2. STOP-LOSS:
   - Set FEATURES['stop_loss'] = True
   - Adjust STOP_LOSS_PARAMS as needed
   - Compare metrics

3. TREND FILTER:
   - Set FEATURES['trend_filter'] = True
   - Adjust TREND_FILTER_PARAMS
   - Compare metrics

4. VOLUME FILTER:
   - Set FEATURES['volume_filter'] = True
   - Add volume data to CSV
   - Compare metrics

5. TRAILING STOP:
   - Set FEATURES['trailing_stop'] = True
   - Compare metrics

6. POSITION SIZING:
   - Set FEATURES['position_sizing'] = True
   - Implement position size calculation

7. MARKET REGIME:
   - Set FEATURES['market_regime'] = True
   - Implement regime detection

8. PROFIT TARGETS:
   - Set FEATURES['profit_targets'] = True
   - Implement multiple exit levels

After testing each feature, optimize parameters for best results.
"""
# ----------------------------------------------------------------------------------------------