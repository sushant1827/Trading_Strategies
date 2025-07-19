import pandas as pd
import numpy as np
import ta

# Define the path to the CSV file
csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_60_Min.csv'

try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

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

    # Round all numerical columns to 2 decimal places
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].round(2)

    def calculate_supertrend(
        df,
        period=10,
        multiplier=3.0,
        trending_up_col_name="trendingUp",
        trending_down_col_name="trendingDown",
    ):
        print(f"Inside calculate_supertrend, input df index is unique: {df.index.is_unique}")
        # Store original index to restore later
        original_index = df.index
        print(f"Inside calculate_supertrend, original_index is unique: {original_index.is_unique}")
        # Reset index to ensure a default integer index for iloc operations
        df = df.reset_index(drop=True)

        # Ensure the DataFrame has the necessary Heikin Ashi columns
        if not all(
            col in df.columns for col in ["HA_Open", "HA_High", "HA_Low", "HA_Close"]
        ):
            raise ValueError(
                "DataFrame must contain 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close' columns for Supertrend calculation."
            )

        ha_high = df["HA_High"]
        ha_low = df["HA_Low"]
        ha_close = df["HA_Close"]

        # Calculate True Range based on HA candles
        df["TR"] = np.maximum.reduce(
            [
                ha_high - ha_low,
                abs(ha_high - ha_close.shift(1)),
                abs(ha_low - ha_close.shift(1)),
            ]
        )

        # ATR using Wilder's smoothing (alpha = 1/period)
        df["ATR"] = df["TR"].ewm(alpha=1 / period, adjust=False).mean()

        # Calculate base bands
        basic_upper_band = ((ha_high + ha_low) / 2) + (multiplier * df["ATR"])
        basic_lower_band = ((ha_high + ha_low) / 2) - (multiplier * df["ATR"])

        # Initialize Supertrend components
        for col in [
            trending_up_col_name,
            trending_down_col_name,
            "direction",
            "SuperTrend",
        ]:
            if col not in df.columns:
                df[col] = np.nan

        # Find the first valid ATR index (integer position)
        first_valid_atr_idx = df["ATR"].first_valid_index()
        if first_valid_atr_idx is None:
            # If no valid ATR, return NaNs with the original index
            return pd.Series(np.nan, index=original_index)

        # Initialize for the first valid point
        df.loc[first_valid_atr_idx, trending_up_col_name] = basic_lower_band.iloc[
            first_valid_atr_idx
        ]
        df.loc[first_valid_atr_idx, trending_down_col_name] = basic_upper_band.iloc[
            first_valid_atr_idx
        ]

        if (
            ha_close.iloc[first_valid_atr_idx]
            > df[trending_down_col_name].iloc[first_valid_atr_idx]
        ):
            df.loc[first_valid_atr_idx, "direction"] = 1
        elif (
            ha_close.iloc[first_valid_atr_idx]
            < df[trending_up_col_name].iloc[first_valid_atr_idx]
        ):
            df.loc[first_valid_atr_idx, "direction"] = -1
        else:
            df.loc[first_valid_atr_idx, "direction"] = 1

        df.loc[first_valid_atr_idx, "SuperTrend"] = (
            df[trending_up_col_name].iloc[first_valid_atr_idx]
            if df["direction"].iloc[first_valid_atr_idx] == 1
            else df[trending_down_col_name].iloc[first_valid_atr_idx]
        )

        # Get column integer locations once outside the loop
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

            # Calculate trendingUp
            if prev_ha_close > prev_trending_up:
                df.iloc[i, trending_up_loc] = max(
                    current_basic_lower_band, prev_trending_up
                )
            else:
                df.iloc[i, trending_up_loc] = current_basic_lower_band

            # Calculate trendingDown
            if prev_ha_close < prev_trending_down:
                df.iloc[i, trending_down_loc] = min(
                    current_basic_upper_band, prev_trending_down
                )
            else:
                df.iloc[i, trending_down_loc] = current_basic_upper_band

            # Calculate direction
            if current_ha_close > df.iloc[i - 1, trending_down_loc]:
                df.iloc[i, direction_loc] = 1
            elif current_ha_close < df.iloc[i - 1, trending_up_loc]:
                df.iloc[i, direction_loc] = -1
            else:
                df.iloc[i, direction_loc] = prev_direction

            # Calculate SuperTrend value
            df.iloc[i, supertrend_loc] = (
                df.iloc[i, trending_up_loc]
                if df.iloc[i, direction_loc] == 1
                else df.iloc[i, trending_down_loc]
            )

        st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}"
        st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}"

        # Rename the columns to be unique for this SuperTrend instance
        df.rename(
            columns={"SuperTrend": st_col_name, "direction": st_dir_col_name}, inplace=True
        )

        # Restore original index and return the relevant columns including trendingUp and trendingDown
        df = df.set_index(original_index)
        return df[
            [st_col_name, st_dir_col_name, trending_up_col_name, trending_down_col_name]
        ]


    def calculate_heikin_ashi(df):
        print(f"Inside calculate_heikin_ashi, input df index is unique: {df.index.is_unique}")
        # Ensure the DataFrame has the necessary OHLC columns
        if not all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
            raise ValueError(
                "DataFrame must contain 'Open', 'High', 'Low', 'Close' columns for Heikin Ashi calculation."
            )

        open_prices = df["Open"].to_numpy()
        high_prices = df["High"].to_numpy()
        low_prices = df["Low"].to_numpy()
        close_prices = df["Close"].to_numpy()

        ha_close = (open_prices + high_prices + low_prices + close_prices) / 4

        ha_open = np.zeros_like(open_prices, dtype=float)
        if len(open_prices) > 0:
            ha_open[0] = open_prices[0]  # First HA_Open is current Open

        for i in range(1, len(open_prices)):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2

        ha_high = np.maximum.reduce([high_prices, ha_open, ha_close])
        ha_low = np.minimum.reduce([low_prices, ha_open, ha_close])

        ha_df = pd.DataFrame(
            {
                "HA_Open": ha_open,
                "HA_High": ha_high,
                "HA_Low": ha_low,
                "HA_Close": ha_close,
            },
            index=df.index,
        )  # Preserve original index

        print(f"Inside calculate_heikin_ashi, output ha_df index is unique: {ha_df.index.is_unique}")
        return ha_df

    # Calculate Heikin Ashi
    ha_df = calculate_heikin_ashi(df.copy())
    df = pd.concat([df, ha_df], axis=1)
    print(f"After concat with HA, df index is unique: {df.index.is_unique}")
    print(f"df columns after HA concat: {df.columns.tolist()}")

    # Parameters for Supertrend 1 (10, 1.0)
    supertrend_period_1 = 10
    supertrend_multiplier_1 = 1.0

    # Calculate Supertrend 1 using Heikin Ashi values
    st_result_ha_1 = calculate_supertrend(
        df.copy(),
        period=supertrend_period_1,
        multiplier=supertrend_multiplier_1,
        trending_up_col_name="trendingUp_HA_10_1",
        trending_down_col_name="trendingDown_HA_10_1",
    )
    print(f"st_result_ha_1 columns before concat: {st_result_ha_1.columns.tolist()}")
    df = pd.concat([df, st_result_ha_1], axis=1)
    print(f"df columns after ST1 concat: {df.columns.tolist()}")

    # Remove temporary trendingUp and trendingDown columns for ST1
    df = df.drop(columns=["trendingUp_HA_10_1", "trendingDown_HA_10_1"], errors='ignore')
    print(f"df columns after ST1 drop: {df.columns.tolist()}")

    # Parameters for Supertrend 2 (11, 2.0)
    supertrend_period_2 = 11
    supertrend_multiplier_2 = 2.0

    # Calculate Supertrend 2 using Heikin Ashi values
    st_result_ha_2 = calculate_supertrend(
        df.copy(),
        period=supertrend_period_2,
        multiplier=supertrend_multiplier_2,
        trending_up_col_name="trendingUp_HA_11_2",
        trending_down_col_name="trendingDown_HA_11_2",
    )
    print(f"st_result_ha_2 columns before concat: {st_result_ha_2.columns.tolist()}")
    df = pd.concat([df, st_result_ha_2], axis=1)
    print(f"df columns after ST2 concat: {df.columns.tolist()}")

    # Remove temporary trendingUp and trendingDown columns for ST2
    df = df.drop(columns=["trendingUp_HA_11_2", "trendingDown_HA_11_2"], errors='ignore')
    print(f"df columns after ST2 drop: {df.columns.tolist()}")


    # Parameters for Supertrend 3 (12, 3.0)
    supertrend_period_3 = 12
    supertrend_multiplier_3 = 3.0

    # Calculate Supertrend 3 using Heikin Ashi values
    st_result_ha_3 = calculate_supertrend(
        df.copy(),
        period=supertrend_period_3,
        multiplier=supertrend_multiplier_3,
        trending_up_col_name="trendingUp_HA_12_3",
        trending_down_col_name="trendingDown_HA_12_3",
    )
    print(f"st_result_ha_3 columns before concat: {st_result_ha_3.columns.tolist()}")
    df = pd.concat([df, st_result_ha_3], axis=1)
    print(f"df columns after ST3 concat: {df.columns.tolist()}")

    # Remove temporary trendingUp and trendingDown columns for ST3
    df = df.drop(columns=["trendingUp_HA_12_3", "trendingDown_HA_12_3"], errors='ignore')
    print(f"df columns after ST3 drop: {df.columns.tolist()}")

    # Initialize Buy_Entry and Buy_Exit columns
    df['Buy_Entry'] = np.nan
    df['Buy_Exit'] = np.nan
    df['Sell_Entry'] = np.nan
    df['Sell_Exit'] = np.nan

    in_buy_position = False
    in_sell_position = False

    # Iterate through the DataFrame to generate Buy_Entry, Buy_Exit, Sell_Entry, and Sell_Exit signals
    for i in range(1, len(df)):
        # Check for Buy_Entry condition
        if not in_buy_position and not in_sell_position and \
           df['ST_Dir_10_1_0'].iloc[i] == 1 and \
           df['ST_Dir_11_2_0'].iloc[i] == 1 and \
           df['ST_Dir_12_3_0'].iloc[i] == 1:
            df.loc[df.index[i], 'Buy_Entry'] = df['Close'].iloc[i]
            in_buy_position = True
            in_sell_position = False # Ensure no sell position if a buy is entered
        # Check for Buy_Exit condition
        elif in_buy_position and \
             (df['ST_Dir_10_1_0'].iloc[i] == -1 or \
              df['ST_Dir_11_2_0'].iloc[i] == -1 or \
              df['ST_Dir_12_3_0'].iloc[i] == -1):
            df.loc[df.index[i], 'Buy_Exit'] = df['Close'].iloc[i]
            in_buy_position = False

        # Check for Sell_Entry condition
        if not in_sell_position and not in_buy_position and \
           df['ST_Dir_10_1_0'].iloc[i] == -1 and \
           df['ST_Dir_11_2_0'].iloc[i] == -1 and \
           df['ST_Dir_12_3_0'].iloc[i] == -1:
            df.loc[df.index[i], 'Sell_Entry'] = df['Close'].iloc[i]
            in_sell_position = True
            in_buy_position = False # Ensure no buy position if a sell is entered
        # Check for Sell_Exit condition
        elif in_sell_position and \
             (df['ST_Dir_10_1_0'].iloc[i] == 1 or \
              df['ST_Dir_11_2_0'].iloc[i] == 1 or \
              df['ST_Dir_12_3_0'].iloc[i] == 1):
            df.loc[df.index[i], 'Sell_Exit'] = df['Close'].iloc[i]
            in_sell_position = False

    # Calculate profits
    total_buy_profit = 0
    total_sell_profit = 0
    current_buy_entry_price = np.nan
    current_sell_entry_price = np.nan

    for i in range(len(df)):
        if not pd.isna(df['Buy_Entry'].iloc[i]):
            current_buy_entry_price = df['Buy_Entry'].iloc[i]
        if not pd.isna(df['Buy_Exit'].iloc[i]) and not pd.isna(current_buy_entry_price):
            total_buy_profit += (df['Buy_Exit'].iloc[i] - current_buy_entry_price)
            current_buy_entry_price = np.nan # Reset after exit

        if not pd.isna(df['Sell_Entry'].iloc[i]):
            current_sell_entry_price = df['Sell_Entry'].iloc[i]
        if not pd.isna(df['Sell_Exit'].iloc[i]) and not pd.isna(current_sell_entry_price):
            total_sell_profit += (current_sell_entry_price - df['Sell_Exit'].iloc[i])
            current_sell_entry_price = np.nan # Reset after exit

    total_profit = total_buy_profit + total_sell_profit

    print("\n--- Profit Analysis ---")
    print(f"Overall BUY Position Profit: {total_buy_profit:.2f}")
    print(f"Overall SELL Position Profit: {total_sell_profit:.2f}")
    print(f"Total Profit: {total_profit:.2f}")
    print("-----------------------")

    # Print the first 5 rows with new Supertrend and Buy/Sell columns
    print("First 5 rows of the DataFrame with Supertrend and Buy/Sell columns:")
    print(df.head().to_markdown(numalign="left", stralign="left"))

    # Print the last 5 rows
    print("\nLast 5 rows of the DataFrame:")
    print(df.tail().to_markdown(numalign="left", stralign="left"))

    # Save the DataFrame to a CSV file
    output_csv_path = 'three_supertrend_signals.csv'
    df.to_csv(output_csv_path)
    print(f"\nDataFrame saved to '{output_csv_path}'")

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
