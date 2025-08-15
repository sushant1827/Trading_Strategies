import pandas as pd
import numpy as np
import ta

# Define the path to the CSV file
csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'

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

    # Filter data from 2021 onwards
    df = df[df.index.year >= 2021]
    print(f"After filtering for 2021 onwards, df shape: {df.shape}")

    # Round all numerical columns to 2 decimal places
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].round(2)

    def calculate_supertrend(df, period=10, multiplier=3.0, suffix=""):
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

        # Dynamic column names for trendingUp and trendingDown, with suffix
        trending_up_col_name = f"trendingUp_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        trending_down_col_name = f"trendingDown_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"

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

        st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}{suffix}"

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

    # Define Supertrend parameters as a list of tuples (period, multiplier)
    supertrend_params = [(9, 1.0), (21, 1.0)]  # Add more parameters as needed
    st_direction_cols = [] # To store the names of the Supertrend direction columns

    for i, (period, multiplier) in enumerate(supertrend_params):
        suffix = f"_{i+1}" # Unique suffix for each supertrend
        print(f"\nCalculating Supertrend {i+1} with Period={period}, Multiplier={multiplier}")
        st_result_ha = calculate_supertrend(
            df.copy(),
            period=period,
            multiplier=multiplier,
            suffix=suffix # Pass the unique suffix
        )
        st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        trending_up_col_name = f"trendingUp_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"
        trending_down_col_name = f"trendingDown_HA_{period}_{str(multiplier).replace('.', '_')}{suffix}"

        st_direction_cols.append(st_dir_col_name)

        print(f"st_result_ha columns before concat: {st_result_ha.columns.tolist()}")
        df = pd.concat([df, st_result_ha], axis=1)
        print(f"df columns after ST{i+1} concat: {df.columns.tolist()}")

        # Remove temporary trendingUp and trendingDown columns
        df = df.drop(columns=[trending_up_col_name, trending_down_col_name], errors='ignore')
        print(f"df columns after ST{i+1} drop: {df.columns.tolist()}")


    # Initialize Buy_Entry and Buy_Exit columns
    df['Buy_Entry'] = np.nan
    df['Buy_Exit'] = np.nan
    df['Sell_Entry'] = np.nan
    df['Sell_Exit'] = np.nan

    in_buy_position = False
    in_sell_position = False
    pending_buy_signal = False
    pending_sell_signal = False

    # Iterate through the DataFrame to generate Buy_Entry, Buy_Exit, Sell_Entry, and Sell_Exit signals
    for i in range(1, len(df)):
        current_close = df['Close'].iloc[i]
        current_open = df['Open'].iloc[i]
        st1_dir = df[st_direction_cols[0]].iloc[i]
        st2_dir = df[st_direction_cols[1]].iloc[i]

        is_green_candle = current_close > current_open
        is_red_candle = current_close < current_open
        
        # Check if Supertrends are giving a buy signal
        supertrends_buy_signal = (st1_dir == 1 and st2_dir == 1)
        # Check if Supertrends are giving a sell signal
        supertrends_sell_signal = (st1_dir == -1 and st2_dir == -1)

        # Handle pending buy signals
        if pending_buy_signal:
            if supertrends_buy_signal and is_green_candle:
                df.loc[df.index[i], 'Buy_Entry'] = current_close
                in_buy_position = True
                in_sell_position = False
                pending_buy_signal = False
            elif not supertrends_buy_signal or is_red_candle: # If signal invalidates or candle turns red
                pending_buy_signal = False # Invalidate pending signal

        # Handle pending sell signals
        if pending_sell_signal:
            if supertrends_sell_signal and is_red_candle:
                df.loc[df.index[i], 'Sell_Entry'] = current_close
                in_sell_position = True
                in_buy_position = False
                pending_sell_signal = False
            elif not supertrends_sell_signal or is_green_candle: # If signal invalidates or candle turns green
                pending_sell_signal = False # Invalidate pending signal

        # Check for new Buy_Entry condition (if not already in position or pending)
        if not in_buy_position and not in_sell_position and not pending_buy_signal and supertrends_buy_signal:
            if is_green_candle:
                df.loc[df.index[i], 'Buy_Entry'] = current_close
                in_buy_position = True
                in_sell_position = False
            else: # Supertrend buy signal but red/doji candle
                pending_buy_signal = True

        # Check for Buy_Exit condition
        elif in_buy_position and (st1_dir == -1 or st2_dir == -1):
            df.loc[df.index[i], 'Buy_Exit'] = current_close
            in_buy_position = False
            pending_buy_signal = False # Clear any pending buy if exiting

        # Check for new Sell_Entry condition (if not already in position or pending)
        if not in_sell_position and not in_buy_position and not pending_sell_signal and supertrends_sell_signal:
            if is_red_candle:
                df.loc[df.index[i], 'Sell_Entry'] = current_close
                in_sell_position = True
                in_buy_position = False
            else: # Supertrend sell signal but green/doji candle
                pending_sell_signal = True

        # Check for Sell_Exit condition
        elif in_sell_position and (st1_dir == 1 or st2_dir == 1):
            df.loc[df.index[i], 'Sell_Exit'] = current_close
            in_sell_position = False
            pending_sell_signal = False # Clear any pending sell if exiting

    # Calculate profits
    # Calculate profits and store individual trades
    trades = []
    current_buy_entry = None
    current_sell_entry = None

    for i in range(len(df)):
        # Handle Buy Trades
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
            current_buy_entry = None # Reset after exit

        # Handle Sell Trades
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
            current_sell_entry = None # Reset after exit

    trades_df = pd.DataFrame(trades)

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

        # Calculate Returns (Cumulative Profit)
        returns_series = trades_df.set_index('exit_date')['profit'].cumsum()
        net_profit = returns_series.iloc[-1] if not returns_series.empty else 0

        # Define initial capital for drawdown calculation
        initial_capital = 100000  # Assuming a starting capital of 100,000

        # Calculate Maximum Drawdown based on equity curve
        if not returns_series.empty:
            # Create an equity curve by adding initial capital to cumulative profits
            equity_curve = initial_capital + returns_series.fillna(0)
            
            # Calculate the peak of the equity curve
            peak = equity_curve.expanding(min_periods=1).max()
            
            # Calculate drawdown as a percentage of the peak equity
            drawdown = (equity_curve - peak) / peak
            max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0
        else:
            max_drawdown = 0

        # Calculate Net Profit Percentage for Return-Drawdown Ratio
        net_profit_percent = (net_profit / initial_capital) * 100 if initial_capital != 0 else 0
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
            else: # Neutral trade (profit == 0)
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
        # For these, we need daily returns. Let's resample the trade profits.
        # Assuming a risk-free rate of 0 for simplicity, and 252 trading days in a year.
        daily_returns = trades_df.set_index('exit_date')['profit'].resample('D').sum().fillna(0)
        annualized_returns = daily_returns.mean() * 252
        annualized_std_dev = daily_returns.std() * np.sqrt(252)

        sharpe_ratio = annualized_returns / annualized_std_dev if annualized_std_dev != 0 else np.nan

        # Sortino Ratio: Only considers downside deviation
        downside_returns = daily_returns[daily_returns < 0]
        downside_std_dev = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
        sortino_ratio = annualized_returns / downside_std_dev if downside_std_dev != 0 else np.nan

        # Profit each year
        trades_df['exit_year'] = trades_df['exit_date'].dt.year
        profit_by_year = trades_df.groupby('exit_year')['profit'].sum()
        buy_profit_by_year = trades_df[trades_df['type'] == 'buy'].groupby('exit_year')['profit'].sum()
        sell_profit_by_year = trades_df[trades_df['type'] == 'sell'].groupby('exit_year')['profit'].sum()

        print("\n--- Trading Metrics ---")
        for i, (period, multiplier) in enumerate(supertrend_params):
            print(f"Supertrend {i+1} Parameters: Period={period}, Multiplier={multiplier}")
        print(f"Total Trades: {num_total_trades}")
        print(f"Winning Trades: {num_winning_trades}")
        print(f"Losing Trades: {num_losing_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win: {avg_win:.2f}")
        print(f"Average Loss: {avg_loss:.2f}")
        print(f"Risk-Reward Ratio: {risk_reward_ratio:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Expectancy: {expectancy:.2f}")
        print(f"Initial Capital for Drawdown Calculation: {initial_capital:.2f}")
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

    # Print the last 5 rows
    # print("\nLast 5 rows of the DataFrame:")
    # print(df.tail(3).to_markdown(numalign="left", stralign="left"))

    # Save the DataFrame to a CSV file
    output_csv_path = 'two_supertrend_signals.csv'
    df.to_csv(output_csv_path)
    print(f"\nDataFrame saved to '{output_csv_path}'")

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
