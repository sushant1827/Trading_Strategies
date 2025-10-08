
# Single Supertrend Gaussian Strategy with Profit Booking
# Strategy Total Profit (Points): Single supertrend version with profit booking mechanism

# ----------------------------------------------------------------------------------------------

# Entry and exit conditions for buy and sell signals in the Single Supertrend Strategy:

# Buy Entry Conditions:
# - A potential buy signal is generated when the Supertrend indicator shows upward trend (direction = 1)
#   AND the Multi-Kernel Regression signal indicates a buy (signal = 1)
# - Trade is executed at the OPEN price of the candle immediately following the signal candle
# - Both buy and sell signals can be generated on the same candle when applicable

# Buy Exit/Profit Booking:
# - Stop Loss: 0.2% of entry price
# - Profit Booking: 40% position booked at 1.5x target (0.3%), 40% at 2.5x target (0.5%), 20% held until next signal

# Sell Entry Conditions:
# - A potential sell signal is generated when the Supertrend indicator shows downward trend (direction = -1)
#   AND the Multi-Kernel Regression signal indicates a sell (signal = -1)
# - Trade is executed at the OPEN price of the candle immediately following the signal candle
# - Both buy and sell signals can be generated on the same candle when applicable

# Sell Exit/Profit Booking:
# - Stop Loss: 0.2% of entry price
# - Profit Booking: 40% position booked at 1.5x target (0.3%), 40% at 2.5x target (0.5%), 20% held until next signal

# ----------------------------------------------------------------------------------------------

# Configuration Parameters - All parameters are set at the top of the code
CSV_FILE_PATH = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'
KERNEL_TYPE = "Epanechnikov"
KERNEL_BANDWIDTH = 9
KERNEL_DEVIATIONS = 2.0

# Supertrend Parameters
SUPERTRND_PERIOD = 26
SUPERTRND_MULTIPLIER = 0.9

# Trading Parameters
INITIAL_CAPITAL = 100000  # Starting capital
RISK_PER_TRADE_PCT = 0.02  # Risk per trade as % of capital (2%)
TARGET_PCT = 0.0025  # Target percentage (0.25%)
SL_PCT = 0.0015  # Stop Loss percentage (0.15%)

# Profit Booking Parameters
PROFIT_BOOKING_1_PCT = 0.4  # 40% position booked at 1.5x target
PROFIT_BOOKING_2_PCT = 0.4  # 40% position booked at 2.5x target
REMAINING_POSITION_PCT = 0.2  # 20% remaining position

# ----------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Multi_Kernel_Regression import MultiKernelRegression

try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(CSV_FILE_PATH)

    # Remove 'Unnamed: 0' and 'Volume' columns
    df = df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')

    # Rename 'Date' column to 'DateTime'
    df = df.rename(columns={'Date': 'DateTime'})

    # Set 'DateTime' column as index
    df = df.set_index('DateTime')

    # Convert index to datetime objects
    df.index = pd.to_datetime(df.index)

    # Drop duplicate index entries, keeping the first one
    df = df[~df.index.duplicated(keep='first')]
    print(f"After dropping duplicates, df index is unique: {df.index.is_unique}")

    # Filter data from 2021 onwards
    df = df[df.index.year >= 2021]
    print(f"After filtering for 2021 onwards, df shape: {df.shape}")

    # Round all numerical columns to 2 decimal places
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].round(2)

    # Initialize MultiKernelRegression
    mkr = MultiKernelRegression(bandwidth=KERNEL_BANDWIDTH, kernel=KERNEL_TYPE, deviations=KERNEL_DEVIATIONS)
    df = mkr.generate_signals(df.copy(), price_column='Close')
    print(f"After adding Kernel Regression signals, df columns: {df.columns.tolist()}")

    def calculate_supertrend(df, period=SUPERTRND_PERIOD, multiplier=SUPERTRND_MULTIPLIER, suffix=""):
        print(f"Inside calculate_supertrend, input df index is unique: {df.index.is_unique}")
        original_index = df.index
        print(f"Inside calculate_supertrend, original_index is unique: {original_index.is_unique}")
        df = df.reset_index(drop=True)

        if not all(col in df.columns for col in ["HA_Open", "HA_High", "HA_Low", "HA_Close"]):
            raise ValueError("DataFrame must contain 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close' columns for Supertrend calculation.")

        ha_high = df["HA_High"]
        ha_low = df["HA_Low"]
        ha_close = df["HA_Close"]

        df["TR"] = np.maximum.reduce([ha_high - ha_low, abs(ha_high - ha_close.shift(1)), abs(ha_low - ha_close.shift(1))])
        df["ATR"] = df["TR"].ewm(alpha=1 / period, adjust=False).mean()

        basic_upper_band = ((ha_high + ha_low) / 2) + (multiplier * df["ATR"])
        basic_lower_band = ((ha_high + ha_low) / 2) - (multiplier * df["ATR"])

        trending_up_col_name = f"trendingUp_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        trending_down_col_name = f"trendingDown_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"

        for col in [trending_up_col_name, trending_down_col_name, "direction", "SuperTrend"]:
            if col not in df.columns:
                df[col] = np.nan

        first_valid_atr_idx = df["ATR"].first_valid_index()
        if first_valid_atr_idx is None:
            return pd.Series(np.nan, index=original_index)

        df.loc[first_valid_atr_idx, trending_up_col_name] = basic_lower_band.iloc[first_valid_atr_idx]
        df.loc[first_valid_atr_idx, trending_down_col_name] = basic_upper_band.iloc[first_valid_atr_idx]

        if ha_close.iloc[first_valid_atr_idx] > df[trending_down_col_name].iloc[first_valid_atr_idx]:
            df.loc[first_valid_atr_idx, "direction"] = 1
        elif ha_close.iloc[first_valid_atr_idx] < df[trending_up_col_name].iloc[first_valid_atr_idx]:
            df.loc[first_valid_atr_idx, "direction"] = -1
        else:
            df.loc[first_valid_atr_idx, "direction"] = 1

        df.loc[first_valid_atr_idx, "SuperTrend"] = (
            df[trending_up_col_name].iloc[first_valid_atr_idx]
            if df["direction"].iloc[first_valid_atr_idx] == 1
            else df[trending_down_col_name].iloc[first_valid_atr_idx]
        )

        trending_up_loc = df.columns.get_loc(trending_up_col_name)
        trending_down_loc = df.columns.get_loc(trending_down_col_name)
        direction_loc = df.columns.get_loc("direction")
        supertrend_loc = df.columns.get_loc("SuperTrend")

        for i in range(first_valid_atr_idx + 1, len(df)):
            prev_ha_close = ha_close.iloc[i - 1]
            prev_trending_up = df.iloc[i - 1, trending_up_loc]
            prev_trending_down = df.iloc[i - 1, trending_down_loc]
            prev_direction = df.iloc[i - 1, direction_loc]

            current_basic_upper_band = basic_upper_band.iloc[i]
            current_basic_lower_band = basic_lower_band.iloc[i]
            current_ha_close = ha_close.iloc[i]

            if prev_ha_close > prev_trending_up:
                df.iloc[i, trending_up_loc] = max(current_basic_lower_band, prev_trending_up)
            else:
                df.iloc[i, trending_up_loc] = current_basic_lower_band

            if prev_ha_close < prev_trending_down:
                df.iloc[i, trending_down_loc] = min(current_basic_upper_band, prev_trending_down)
            else:
                df.iloc[i, trending_down_loc] = current_basic_upper_band

            if current_ha_close > df.iloc[i - 1, trending_down_loc]:
                df.iloc[i, direction_loc] = 1
            elif current_ha_close < df.iloc[i - 1, trending_up_loc]:
                df.iloc[i, direction_loc] = -1
            else:
                df.iloc[i, direction_loc] = prev_direction

            df.iloc[i, supertrend_loc] = (
                df.iloc[i, trending_up_loc]
                if df.iloc[i, direction_loc] == 1
                else df.iloc[i, trending_down_loc]
            )

        st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}{suffix}"

        df.rename(columns={"SuperTrend": st_col_name, "direction": st_dir_col_name}, inplace=True)
        df = df.set_index(original_index)
        return df[[st_col_name, st_dir_col_name, trending_up_col_name, trending_down_col_name]]

    def calculate_heikin_ashi(df):
        print(f"Inside calculate_heikin_ashi, input df index is unique: {df.index.is_unique}")
        if not all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
            raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns for Heikin Ashi calculation.")

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

        print(f"Inside calculate_heikin_ashi, output ha_df index is unique: {ha_df.index.is_unique}")
        return ha_df

    # Calculate Heikin Ashi
    ha_df = calculate_heikin_ashi(df.copy())
    df = pd.concat([df, ha_df], axis=1)
    print(f"After concat with HA, df index is unique: {df.index.is_unique}")
    print(f"df columns after HA concat: {df.columns.tolist()}")

    # Calculate Single Supertrend
    print(f"\nCalculating Single Supertrend with Period={SUPERTRND_PERIOD}, Multiplier={SUPERTRND_MULTIPLIER}")
    st_result_ha = calculate_supertrend(df.copy(), period=SUPERTRND_PERIOD, multiplier=SUPERTRND_MULTIPLIER, suffix="")
    st_col_name = f"ST_{SUPERTRND_PERIOD}_{str(SUPERTRND_MULTIPLIER).replace('.', '_')}"
    st_dir_col_name = f"ST_Dir_{SUPERTRND_PERIOD}_{str(SUPERTRND_MULTIPLIER).replace('.', '_')}"
    trending_up_col_name = f"trendingUp_HA_{SUPERTRND_PERIOD}_{str(SUPERTRND_MULTIPLIER).replace('.', '_')}"
    trending_down_col_name = f"trendingDown_HA_{SUPERTRND_PERIOD}_{str(SUPERTRND_MULTIPLIER).replace('.', '_')}"

    print(f"st_result_ha columns before concat: {st_result_ha.columns.tolist()}")
    df = pd.concat([df, st_result_ha], axis=1)
    print(f"df columns after Single ST concat: {df.columns.tolist()}")

    df = df.drop(columns=[trending_up_col_name, trending_down_col_name], errors='ignore')
    print(f"df columns after Single ST drop: {df.columns.tolist()}")

    # Initialize columns for enhanced strategy
    df['Buy_Signal'] = np.nan  # Signal candle for buy
    df['Sell_Signal'] = np.nan  # Signal candle for sell
    df['Buy_Entry'] = np.nan   # Next candle open for buy
    df['Sell_Entry'] = np.nan  # Next candle open for sell
    df['Buy_Exit'] = np.nan
    df['Sell_Exit'] = np.nan
    df['Quantity'] = np.nan
    df['Position_Status'] = 'none'  # none, buy_position, sell_position
    df['Entry_Price'] = np.nan
    df['Stop_Loss'] = np.nan
    df['Target_1'] = np.nan  # 1.5x target for 40% position
    df['Target_2'] = np.nan  # 2.5x target for 40% position
    df['Profit_Booked_1'] = np.nan  # 40% profit booking at target 1
    df['Profit_Booked_2'] = np.nan  # 40% profit booking at target 2
    df['Remaining_Position'] = np.nan  # 20% remaining position

    # Track current capital for position sizing
    current_capital = INITIAL_CAPITAL

    # Track position state
    position_data = {
        'status': 'none',  # none, buy_position, sell_position
        'entry_price': 0,
        'quantity': 0,
        'stop_loss': 0,
        'target_1': 0,
        'target_2': 0,
        'profit_booked_1': False,
        'profit_booked_2': False,
        'remaining_quantity': 0
    }

    # Generate signals and entries
    for i in range(1, len(df)):
        current_close = df['Close'].iloc[i]
        current_open = df['Open'].iloc[i]
        next_open = df['Open'].iloc[i] if i == len(df) - 1 else df['Open'].iloc[i + 1]

        st_dir = df[st_dir_col_name].iloc[i]

        # Check for buy/sell signals
        supertrend_buy = (st_dir == 1)
        supertrend_sell = (st_dir == -1)
        mkr_buy = df['signal'].iloc[i] == 1
        mkr_sell = df['signal'].iloc[i] == -1

        # Generate buy signal (can be on same candle as sell signal)
        if supertrend_buy and mkr_buy and position_data['status'] != 'buy_position':
            df.loc[df.index[i], 'Buy_Signal'] = current_close

        # Generate sell signal (can be on same candle as buy signal)
        if supertrend_sell and mkr_sell and position_data['status'] != 'sell_position':
            df.loc[df.index[i], 'Sell_Signal'] = current_close

        # Execute buy entry at next candle's open
        if (df['Buy_Signal'].iloc[i-1] and not pd.isna(df['Buy_Signal'].iloc[i-1]) and
            position_data['status'] != 'buy_position' and i < len(df)):

            entry_price = next_open

            # Calculate position size based on 2% risk per trade
            risk_amount = current_capital * RISK_PER_TRADE_PCT
            sl_distance = entry_price * SL_PCT
            quantity = risk_amount / sl_distance

            # Round quantity to nearest integer (minimum 1)
            quantity = max(1, round(quantity))

            df.loc[df.index[i], 'Buy_Entry'] = entry_price
            df.loc[df.index[i], 'Quantity'] = quantity
            df.loc[df.index[i], 'Position_Status'] = 'buy_position'
            df.loc[df.index[i], 'Entry_Price'] = entry_price

            # Calculate targets and stop loss (0.2% each)
            target_profit = entry_price * TARGET_PCT
            sl_profit = entry_price * SL_PCT

            position_data['status'] = 'buy_position'
            position_data['entry_price'] = entry_price
            position_data['quantity'] = quantity
            position_data['stop_loss'] = entry_price - sl_profit  # For buy position
            position_data['target_1'] = entry_price + (target_profit * 1.5)  # 1.5x target
            position_data['target_2'] = entry_price + (target_profit * 2.5)  # 2.5x target
            position_data['profit_booked_1'] = False
            position_data['profit_booked_2'] = False
            position_data['remaining_quantity'] = quantity

            # Update DataFrame
            df.loc[df.index[i], 'Stop_Loss'] = position_data['stop_loss']
            df.loc[df.index[i], 'Target_1'] = position_data['target_1']
            df.loc[df.index[i], 'Target_2'] = position_data['target_2']
            df.loc[df.index[i], 'Remaining_Position'] = position_data['remaining_quantity']

        # Execute sell entry at next candle's open
        elif (df['Sell_Signal'].iloc[i-1] and not pd.isna(df['Sell_Signal'].iloc[i-1]) and
              position_data['status'] != 'sell_position' and i < len(df)):

            entry_price = next_open

            # Calculate position size based on 2% risk per trade
            risk_amount = current_capital * RISK_PER_TRADE_PCT
            sl_distance = entry_price * SL_PCT
            quantity = risk_amount / sl_distance

            # Round quantity to nearest integer (minimum 1)
            quantity = max(1, round(quantity))

            df.loc[df.index[i], 'Sell_Entry'] = entry_price
            df.loc[df.index[i], 'Quantity'] = quantity
            df.loc[df.index[i], 'Position_Status'] = 'sell_position'
            df.loc[df.index[i], 'Entry_Price'] = entry_price

            # Calculate targets and stop loss (0.2% each)
            target_profit = entry_price * TARGET_PCT
            sl_profit = entry_price * SL_PCT

            position_data['status'] = 'sell_position'
            position_data['entry_price'] = entry_price
            position_data['quantity'] = quantity
            position_data['stop_loss'] = entry_price + sl_profit  # For sell position
            position_data['target_1'] = entry_price - (target_profit * 1.5)  # 1.5x target
            position_data['target_2'] = entry_price - (target_profit * 2.5)  # 2.5x target
            position_data['profit_booked_1'] = False
            position_data['profit_booked_2'] = False
            position_data['remaining_quantity'] = quantity

            # Update DataFrame
            df.loc[df.index[i], 'Stop_Loss'] = position_data['stop_loss']
            df.loc[df.index[i], 'Target_1'] = position_data['target_1']
            df.loc[df.index[i], 'Target_2'] = position_data['target_2']
            df.loc[df.index[i], 'Remaining_Position'] = position_data['remaining_quantity']

        # Manage existing positions
        elif position_data['status'] == 'buy_position' and i < len(df):
            current_high = df['High'].iloc[i]
            current_low = df['Low'].iloc[i]

            # Check for stop loss hit
            if current_low <= position_data['stop_loss']:
                df.loc[df.index[i], 'Buy_Exit'] = position_data['stop_loss']
                df.loc[df.index[i], 'Position_Status'] = 'exit'
                position_data['status'] = 'none'
                position_data['remaining_quantity'] = 0

            # Check for profit booking at target 1 (40% position)
            elif (not position_data['profit_booked_1'] and
                  current_high >= position_data['target_1'] and
                  position_data['remaining_quantity'] > 0):

                profit_book_quantity = int(position_data['quantity'] * PROFIT_BOOKING_1_PCT)
                df.loc[df.index[i], 'Profit_Booked_1'] = profit_book_quantity
                position_data['profit_booked_1'] = True
                position_data['remaining_quantity'] -= profit_book_quantity
                df.loc[df.index[i], 'Remaining_Position'] = position_data['remaining_quantity']

            # Check for profit booking at target 2 (40% position)
            elif (not position_data['profit_booked_2'] and
                  current_high >= position_data['target_2'] and
                  position_data['remaining_quantity'] > 0):

                profit_book_quantity = int(position_data['quantity'] * PROFIT_BOOKING_2_PCT)
                df.loc[df.index[i], 'Profit_Booked_2'] = profit_book_quantity
                position_data['profit_booked_2'] = True
                position_data['remaining_quantity'] -= profit_book_quantity
                df.loc[df.index[i], 'Remaining_Position'] = position_data['remaining_quantity']

            # Exit remaining position if new signal appears
            elif (position_data['remaining_quantity'] > 0 and
                  (not pd.isna(df['Buy_Signal'].iloc[i]) or not pd.isna(df['Sell_Signal'].iloc[i]))):

                df.loc[df.index[i], 'Buy_Exit'] = current_close
                df.loc[df.index[i], 'Position_Status'] = 'exit'
                position_data['status'] = 'none'
                position_data['remaining_quantity'] = 0

        # Manage existing sell positions
        elif position_data['status'] == 'sell_position' and i < len(df):
            current_high = df['High'].iloc[i]
            current_low = df['Low'].iloc[i]

            # Check for stop loss hit
            if current_high >= position_data['stop_loss']:
                df.loc[df.index[i], 'Sell_Exit'] = position_data['stop_loss']
                df.loc[df.index[i], 'Position_Status'] = 'exit'
                position_data['status'] = 'none'
                position_data['remaining_quantity'] = 0

            # Check for profit booking at target 1 (40% position)
            elif (not position_data['profit_booked_1'] and
                  current_low <= position_data['target_1'] and
                  position_data['remaining_quantity'] > 0):

                profit_book_quantity = int(position_data['quantity'] * PROFIT_BOOKING_1_PCT)
                df.loc[df.index[i], 'Profit_Booked_1'] = profit_book_quantity
                position_data['profit_booked_1'] = True
                position_data['remaining_quantity'] -= profit_book_quantity
                df.loc[df.index[i], 'Remaining_Position'] = position_data['remaining_quantity']

            # Check for profit booking at target 2 (40% position)
            elif (not position_data['profit_booked_2'] and
                  current_low <= position_data['target_2'] and
                  position_data['remaining_quantity'] > 0):

                profit_book_quantity = int(position_data['quantity'] * PROFIT_BOOKING_2_PCT)
                df.loc[df.index[i], 'Profit_Booked_2'] = profit_book_quantity
                position_data['profit_booked_2'] = True
                position_data['remaining_quantity'] -= profit_book_quantity
                df.loc[df.index[i], 'Remaining_Position'] = position_data['remaining_quantity']

            # Exit remaining position if new signal appears
            elif (position_data['remaining_quantity'] > 0 and
                  (not pd.isna(df['Buy_Signal'].iloc[i]) or not pd.isna(df['Sell_Signal'].iloc[i]))):

                df.loc[df.index[i], 'Sell_Exit'] = current_close
                df.loc[df.index[i], 'Position_Status'] = 'exit'
                position_data['status'] = 'none'
                position_data['remaining_quantity'] = 0

    # Calculate profits with profit booking
    trades = []
    current_buy_entry = None
    current_sell_entry = None
    profit_booked_trades = []

    for i in range(len(df)):
        # Handle Buy Trades with profit booking
        if not pd.isna(df['Buy_Entry'].iloc[i]):
            current_buy_entry = {
                'entry_price': df['Buy_Entry'].iloc[i],
                'entry_date': df.index[i],
                'quantity': df['Quantity'].iloc[i],
                'target_1': df['Target_1'].iloc[i],
                'target_2': df['Target_2'].iloc[i]
            }

        if not pd.isna(df['Buy_Exit'].iloc[i]) and current_buy_entry is not None:
            exit_price = df['Buy_Exit'].iloc[i]
            profit = (exit_price - current_buy_entry['entry_price']) * current_buy_entry['quantity']
            trades.append({
                'type': 'buy',
                'entry_date': current_buy_entry['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_buy_entry['entry_price'],
                'exit_price': exit_price,
                'quantity': current_buy_entry['quantity'],
                'profit': profit,
                'exit_type': 'full_exit'
            })
            # Update current capital with profit/loss
            current_capital += profit
            current_buy_entry = None

        # Handle partial exits for buy positions
        if not pd.isna(df['Profit_Booked_1'].iloc[i]) and current_buy_entry is not None:
            exit_price = current_buy_entry['target_1']
            quantity_booked = df['Profit_Booked_1'].iloc[i]
            profit = (exit_price - current_buy_entry['entry_price']) * quantity_booked
            profit_booked_trades.append({
                'type': 'buy',
                'entry_date': current_buy_entry['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_buy_entry['entry_price'],
                'exit_price': exit_price,
                'quantity': quantity_booked,
                'profit': profit,
                'exit_type': 'profit_booking_1'
            })
            # Update current capital with partial profit
            current_capital += profit

        if not pd.isna(df['Profit_Booked_2'].iloc[i]) and current_buy_entry is not None:
            exit_price = current_buy_entry['target_2']
            quantity_booked = df['Profit_Booked_2'].iloc[i]
            profit = (exit_price - current_buy_entry['entry_price']) * quantity_booked
            profit_booked_trades.append({
                'type': 'buy',
                'entry_date': current_buy_entry['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_buy_entry['entry_price'],
                'exit_price': exit_price,
                'quantity': quantity_booked,
                'profit': profit,
                'exit_type': 'profit_booking_2'
            })
            # Update current capital with partial profit
            current_capital += profit

        # Handle Sell Trades with profit booking
        if not pd.isna(df['Sell_Entry'].iloc[i]):
            current_sell_entry = {
                'entry_price': df['Sell_Entry'].iloc[i],
                'entry_date': df.index[i],
                'quantity': df['Quantity'].iloc[i],
                'target_1': df['Target_1'].iloc[i],
                'target_2': df['Target_2'].iloc[i]
            }

        if not pd.isna(df['Sell_Exit'].iloc[i]) and current_sell_entry is not None:
            exit_price = df['Sell_Exit'].iloc[i]
            profit = (current_sell_entry['entry_price'] - exit_price) * current_sell_entry['quantity']
            trades.append({
                'type': 'sell',
                'entry_date': current_sell_entry['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_sell_entry['entry_price'],
                'exit_price': exit_price,
                'quantity': current_sell_entry['quantity'],
                'profit': profit,
                'exit_type': 'full_exit'
            })
            # Update current capital with profit/loss
            current_capital += profit
            current_sell_entry = None

        # Handle partial exits for sell positions
        if not pd.isna(df['Profit_Booked_1'].iloc[i]) and current_sell_entry is not None:
            exit_price = current_sell_entry['target_1']
            quantity_booked = df['Profit_Booked_1'].iloc[i]
            profit = (current_sell_entry['entry_price'] - exit_price) * quantity_booked
            profit_booked_trades.append({
                'type': 'sell',
                'entry_date': current_sell_entry['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_sell_entry['entry_price'],
                'exit_price': exit_price,
                'quantity': quantity_booked,
                'profit': profit,
                'exit_type': 'profit_booking_1'
            })
            # Update current capital with partial profit
            current_capital += profit

        if not pd.isna(df['Profit_Booked_2'].iloc[i]) and current_sell_entry is not None:
            exit_price = current_sell_entry['target_2']
            quantity_booked = df['Profit_Booked_2'].iloc[i]
            profit = (current_sell_entry['entry_price'] - exit_price) * quantity_booked
            profit_booked_trades.append({
                'type': 'sell',
                'entry_date': current_sell_entry['entry_date'],
                'exit_date': df.index[i],
                'entry_price': current_sell_entry['entry_price'],
                'exit_price': exit_price,
                'quantity': quantity_booked,
                'profit': profit,
                'exit_type': 'profit_booking_2'
            })
            # Update current capital with partial profit
            current_capital += profit

    # Combine all trades
    all_trades = trades + profit_booked_trades
    trades_df = pd.DataFrame(all_trades)

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

        # Calculate Returns (Cumulative Profit) for the strategy
        returns_series = trades_df.set_index('exit_date')['profit'].cumsum()
        net_profit = returns_series.iloc[-1] if not returns_series.empty else 0

        # Calculate Maximum Drawdown
        if not returns_series.empty:
            equity_curve_for_metrics = INITIAL_CAPITAL + returns_series.fillna(0)
            peak = equity_curve_for_metrics.expanding(min_periods=1).max()
            drawdown = (equity_curve_for_metrics - peak) / peak
            max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0
        else:
            equity_curve_for_metrics = pd.Series(INITIAL_CAPITAL, index=df.index)
            max_drawdown = 0

        # Calculate Net Profit Percentage for Return-Drawdown Ratio
        net_profit_percent = (net_profit / INITIAL_CAPITAL) * 100 if INITIAL_CAPITAL != 0 else 0
        return_drawdown_ratio = (net_profit_percent / abs(max_drawdown)) if max_drawdown != 0 else np.nan

        # Calculate Winning and Losing Streaks
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

        # Sharpe Ratio and Sortino Ratio
        daily_returns = trades_df.set_index('exit_date')['profit'].resample('D').sum().fillna(0)
        annualized_returns = daily_returns.mean() * 252
        annualized_std_dev = daily_returns.std() * np.sqrt(252)

        sharpe_ratio = annualized_returns / annualized_std_dev if annualized_std_dev != 0 else np.nan

        downside_returns = daily_returns[daily_returns < 0]
        downside_std_dev = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
        sortino_ratio = annualized_returns / downside_std_dev if downside_std_dev != 0 else np.nan

        # Profit by year and type
        trades_df['exit_year'] = trades_df['exit_date'].dt.year
        profit_by_year = trades_df.groupby('exit_year')['profit'].sum()
        buy_profit_by_year = trades_df[trades_df['type'] == 'buy'].groupby('exit_year')['profit'].sum()
        sell_profit_by_year = trades_df[trades_df['type'] == 'sell'].groupby('exit_year')['profit'].sum()

        print("\n--- Single Supertrend Trading Strategy Metrics ---")
        print(f"Kernel: {KERNEL_TYPE}")
        print(f"Supertrend Parameters: Period={SUPERTRND_PERIOD}, Multiplier={SUPERTRND_MULTIPLIER}")
        print(f"Target %: {TARGET_PCT*100}%")
        print(f"Stop Loss %: {SL_PCT*100}%")
        print(f"Risk Per Trade %: {RISK_PER_TRADE_PCT*100}%")
        print(f"Total Trades: {num_total_trades}")
        print(f"Winning Trades: {num_winning_trades}")
        print(f"Losing Trades: {num_losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win: {avg_win:.2f}")
        print(f"Average Loss: {avg_loss:.2f}")
        print(f"Risk-Reward Ratio: {risk_reward_ratio:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Expectancy: {expectancy:.2f}")
        print(f"Initial Capital: {INITIAL_CAPITAL:.2f}")
        print(f"Net Profit: {net_profit:.2f}")
        print(f"Net Profit (% of Initial Capital): {net_profit_percent:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Return-Drawdown Ratio: {return_drawdown_ratio:.2f}")
        print(f"Max Winning Streak: {max_winning_streak}")
        print(f"Max Losing Streak: {max_losing_streak}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")

        print("\nProfit by Year:")
        for year, profit in profit_by_year.items():
            buy_p = buy_profit_by_year.get(year, 0)
            sell_p = sell_profit_by_year.get(year, 0)
            print(f"  {year}: Total Profit: {profit:.2f}, Buy Profit: {buy_p:.2f}, Sell Profit: {sell_p:.2f}")

    else:
        print("\nNo trades found to calculate metrics.")

    # Save the DataFrame to a CSV file
    output_csv_path = 'single_supertrend_gaussian_signals.csv'
    df.to_csv(output_csv_path)
    print(f"\nDataFrame saved to '{output_csv_path}'")

except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
