import pandas as pd
import numpy as np
import random


# Define initial_capital globally for use across functions.
initial_capital = 100000

def calculate_supertrend(
    df,
    period=10,
    multiplier=3.0,
    trending_up_col_name="trendingUp",
    trending_down_col_name="trendingDown",
    use_heikin_ashi=True, # Added parameter to control Heikin Ashi usage
):
    """
    Calculates the Supertrend indicator for a given DataFrame.

    The Supertrend is a trend-following indicator that uses Average True Range (ATR)
    to set its bands. It dynamically adjusts its position based on price action,
    providing clear buy/sell signals.

    Args:
        df (pd.DataFrame): Input DataFrame with OHLC (Open, High, Low, Close) data.
                           If `use_heikin_ashi` is True, it must also contain
                           'HA_Open', 'HA_High', 'HA_Low', 'HA_Close' columns.
        period (int): The period for ATR calculation (default: 10).
        multiplier (float): The multiplier for ATR to determine band distance (default: 3.0).
        trending_up_col_name (str): Name for the trending up column (default: "trendingUp").
        trending_down_col_name (str): Name for the trending down column (default: "trendingDown").
        use_heikin_ashi (bool): If True, uses Heikin Ashi prices for calculation;
                                otherwise, uses standard OHLC prices (default: True).

    Returns:
        pd.DataFrame: A DataFrame containing the Supertrend, direction, trendingUp,
                      and trendingDown columns.
    Raises:
        ValueError: If required Heikin Ashi columns are missing when `use_heikin_ashi` is True,
                    or if standard OHLC columns are missing when `use_heikin_ashi` is False.
    """
    # Store original index to restore later
    original_index = df.index
    # Reset index to ensure a default integer index for iloc operations,
    # which is more robust for loop-based calculations.
    df = df.reset_index(drop=True)

    # Determine which price columns to use based on the `use_heikin_ashi` flag
    if use_heikin_ashi:
        required_cols = ["HA_Open", "HA_High", "HA_Low", "HA_Close"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"DataFrame must contain {required_cols} columns for Supertrend calculation "
                "when use_heikin_ashi is True."
            )
        high_price = df["HA_High"]
        low_price = df["HA_Low"]
        close_price = df["HA_Close"]
    else:
        required_cols = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"DataFrame must contain {required_cols} columns for Supertrend calculation "
                "when use_heikin_ashi is False."
            )
        high_price = df["High"]
        low_price = df["Low"]
        close_price = df["Close"]
        
    # Calculate True Range (TR)
    # TR is the greatest of:
    # 1. Current High - Current Low
    # 2. Absolute value of Current High - Previous Close
    # 3. Absolute value of Current Low - Previous Close
    df["TR"] = np.maximum.reduce(
        [
            high_price - low_price,
            abs(high_price - close_price.shift(1)),
            abs(low_price - close_price.shift(1)),
        ]
    )

    # Calculate Average True Range (ATR) using Wilder's smoothing method (EMA-like)
    # Alpha for EWM is 1/period for Wilder's smoothing.
    df["ATR"] = df["TR"].ewm(alpha=1 / period, adjust=False).mean()

    # Calculate basic upper and lower bands
    # These are the initial bands before applying the trend logic.
    basic_upper_band = ((high_price + low_price) / 2) + (multiplier * df["ATR"])
    basic_lower_band = ((high_price + low_price) / 2) - (multiplier * df["ATR"])

    # Initialize Supertrend related columns with NaN
    # These columns will be populated in the loop.
    for col in [
        trending_up_col_name,
        trending_down_col_name,
        "direction",
        "SuperTrend",
    ]:
        if col not in df.columns: # Avoid re-creating if already exists (e.g., from previous runs)
            df[col] = np.nan

    # Find the first valid index where ATR is not NaN.
    # Supertrend calculation starts from this point.
    first_valid_atr_idx = df["ATR"].first_valid_index()
    if first_valid_atr_idx is None:
        # If no valid ATR can be calculated, return a Series of NaNs with the original index.
        return pd.Series(np.nan, index=original_index)

    # Initialize Supertrend values for the first valid data point.
    # The initial trendingUp is the basic_lower_band, and trendingDown is basic_upper_band.
    df.loc[first_valid_atr_idx, trending_up_col_name] = basic_lower_band.iloc[
        first_valid_atr_idx
    ]
    df.loc[first_valid_atr_idx, trending_down_col_name] = basic_upper_band.iloc[
        first_valid_atr_idx
    ]

    # Determine initial direction and SuperTrend value based on the first valid close price.
    if (
        close_price.iloc[first_valid_atr_idx]
        > df[trending_down_col_name].iloc[first_valid_atr_idx]
    ):
        df.loc[first_valid_atr_idx, "direction"] = 1  # Upward trend
    elif (
        close_price.iloc[first_valid_atr_idx]
        < df[trending_up_col_name].iloc[first_valid_atr_idx]
    ):
        df.loc[first_valid_atr_idx, "direction"] = -1  # Downward trend
    else:
        # If price is within bands, default to upward trend (common practice or previous direction)
        df.loc[first_valid_atr_idx, "direction"] = 1

    df.loc[first_valid_atr_idx, "SuperTrend"] = (
        df[trending_up_col_name].iloc[first_valid_atr_idx]
        if df["direction"].iloc[first_valid_atr_idx] == 1
        else df[trending_down_col_name].iloc[first_valid_atr_idx]
    )

    # Get integer locations of columns for faster access within the loop.
    # This avoids repeated string lookups which can be slow.
    trending_up_loc = df.columns.get_loc(trending_up_col_name)
    trending_down_loc = df.columns.get_loc(trending_down_col_name)
    direction_loc = df.columns.get_loc("direction")
    supertrend_loc = df.columns.get_loc("SuperTrend")

    # Iterate through the DataFrame from the second valid point to calculate Supertrend.
    for i in range(first_valid_atr_idx + 1, len(df)):
        # Get previous candle's values
        prev_close = close_price.iloc[i - 1]
        prev_trending_up = df.iloc[i - 1, trending_up_loc]
        prev_trending_down = df.iloc[i - 1, trending_down_loc]
        prev_direction = df.iloc[i - 1, direction_loc]

        # Get current candle's values
        current_basic_upper_band = basic_upper_band.iloc[i]
        current_basic_lower_band = basic_lower_band.iloc[i]
        current_close = close_price.iloc[i]

        # Calculate current trendingUp band
        # If previous close was above the previous trendingUp band,
        # the current trendingUp band is the maximum of current basic_lower_band and previous trendingUp.
        # Otherwise, it's just the current basic_lower_band.
        if prev_close > prev_trending_up:
            df.iloc[i, trending_up_loc] = max(
                current_basic_lower_band, prev_trending_up
            )
        else:
            df.iloc[i, trending_up_loc] = current_basic_lower_band

        # Calculate current trendingDown band
        # If previous close was below the previous trendingDown band,
        # the current trendingDown band is the minimum of current basic_upper_band and previous trendingDown.
        # Otherwise, it's just the current basic_upper_band.
        if prev_close < prev_trending_down:
            df.iloc[i, trending_down_loc] = min(
                current_basic_upper_band, prev_trending_down
            )
        else:
            df.iloc[i, trending_down_loc] = current_basic_upper_band

        # Determine current direction
        # If current close crosses above trendingDown, direction is 1 (up).
        # If current close crosses below trendingUp, direction is -1 (down).
        # Otherwise, direction remains the same as the previous candle.
        if current_close > df.iloc[i - 1, trending_down_loc]:
            df.iloc[i, direction_loc] = 1
        elif current_close < df.iloc[i - 1, trending_up_loc]:
            df.iloc[i, direction_loc] = -1
        else:
            df.iloc[i, direction_loc] = prev_direction

        # Calculate current SuperTrend value
        # If direction is 1 (up), SuperTrend is trendingUp band.
        # If direction is -1 (down), SuperTrend is trendingDown band.
        df.iloc[i, supertrend_loc] = (
            df.iloc[i, trending_up_loc]
            if df.iloc[i, direction_loc] == 1
            else df.iloc[i, trending_down_loc]
        )

    # Generate unique column names for the SuperTrend instance based on period and multiplier.
    st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}"
    st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}"

    # Rename the generic SuperTrend and direction columns to their unique names.
    df.rename(
        columns={"SuperTrend": st_col_name, "direction": st_dir_col_name}, inplace=True
    )

    # Restore original index and return the relevant columns.
    # This includes the calculated Supertrend, its direction, and the trending bands.
    df = df.set_index(original_index)
    return df[
        [st_col_name, st_dir_col_name, trending_up_col_name, trending_down_col_name]
    ]


def calculate_heikin_ashi(df):
    """
    Calculates Heikin Ashi candles from standard OHLC (Open, High, Low, Close) data.

    Heikin Ashi candles smooth out price action, making trends easier to spot.

    Args:
        df (pd.DataFrame): Input DataFrame with 'Open', 'High', 'Low', 'Close' columns.

    Returns:
        pd.DataFrame: A new DataFrame containing 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close' columns,
                      with the original index preserved.
    Raises:
        ValueError: If required OHLC columns are missing.
    """
    # Ensure the DataFrame has the necessary OHLC columns
    if not all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        raise ValueError(
            "DataFrame must contain 'Open', 'High', 'Low', 'Close' columns for Heikin Ashi calculation."
        )

    # Convert price columns to NumPy arrays for faster calculations.
    open_prices = df["Open"].to_numpy()
    high_prices = df["High"].to_numpy()
    low_prices = df["Low"].to_numpy()
    close_prices = df["Close"].to_numpy()

    # Calculate HA_Close: Average of current Open, High, Low, Close.
    ha_close = (open_prices + high_prices + low_prices + close_prices) / 4

    # Initialize HA_Open array.
    ha_open = np.zeros_like(open_prices, dtype=float)
    if len(open_prices) > 0:
        # The first HA_Open is the same as the first regular Open.
        ha_open[0] = open_prices[0]

    # Calculate subsequent HA_Open values: Average of previous HA_Open and previous HA_Close.
    for i in range(1, len(open_prices)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2

    # Calculate HA_High: Maximum of current High, current HA_Open, current HA_Close.
    ha_high = np.maximum.reduce([high_prices, ha_open, ha_close])
    # Calculate HA_Low: Minimum of current Low, current HA_Open, current HA_Close.
    ha_low = np.minimum.reduce([low_prices, ha_open, ha_close])

    # Create a new DataFrame for Heikin Ashi values, preserving the original index.
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


def calculate_ema(df, period=50, price_col="Close"):
    """
    Calculates the Exponential Moving Average (EMA) for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        period (int): The period for the EMA calculation.
        price_col (str): The name of the column to use for EMA calculation (default: 'Close').

    Returns:
        pd.Series: A Series containing the EMA values.
    Raises:
        ValueError: If the specified price column is not found in the DataFrame.
    """
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column for EMA calculation.")
    return df[price_col].ewm(span=period, adjust=False).mean()


def calculate_trade_profit(
    entry_price, exit_price, trade_type, trade_size, brokerage_pct=0.0003, slippage_points=2
):
    """
    Calculates the net profit for a trade, accounting for brokerage fees and slippage,
    and considering the trade size (number of units/lots).

    Args:
        entry_price (float): The price at which the trade was entered.
        exit_price (float): The price at which the trade was exited.
        trade_type (str): Type of trade, 'long' or 'short'.
        trade_size (int): The number of units/lots traded.
        brokerage_pct (float): Brokerage fee as a percentage (e.g., 0.0003 for 0.03%).
        slippage_points (float): Slippage in points for each buy and sell transaction.

    Returns:
        float: The net profit after deducting transaction costs.
    """
    if trade_size <= 0:
        return 0.0

    # Adjust entry and exit prices for slippage
    # Slippage simulates real-world execution where prices might differ slightly
    # from the theoretical entry/exit points.
    adjusted_entry_price = entry_price
    adjusted_exit_price = exit_price

    if trade_type == "long":
        # For long entry (buying), we assume buying at a slightly higher price due to slippage.
        adjusted_entry_price = entry_price + slippage_points
        # For long exit (selling), we assume selling at a slightly lower price due to slippage.
        adjusted_exit_price = exit_price - slippage_points
    elif trade_type == "short":
        # For short entry (selling), we assume selling at a slightly lower price due to slippage.
        adjusted_entry_price = entry_price - slippage_points
        # For short exit (buying back), we assume buying at a slightly higher price due to slippage.
        adjusted_exit_price = exit_price + slippage_points

    # Calculate gross profit per unit based on trade type.
    gross_profit_per_unit = (
        adjusted_exit_price - adjusted_entry_price
        if trade_type == "long"
        else adjusted_entry_price - adjusted_exit_price
    )
    
    # Total gross profit for the trade size.
    gross_profit = gross_profit_per_unit * trade_size

    # Calculate brokerage cost for both entry and exit transactions.
    brokerage_cost_entry = adjusted_entry_price * trade_size * brokerage_pct
    brokerage_cost_exit = adjusted_exit_price * trade_size * brokerage_pct
    brokerage_cost = brokerage_cost_entry + brokerage_cost_exit

    # Total transaction cost includes only brokerage for now.
    transaction_cost = brokerage_cost

    # Net profit is gross profit minus total transaction costs.
    return gross_profit - transaction_cost


def generate_trades_df(df, initial_capital, risk_per_trade_pct, brokerage_pct=0.0003, slippage_points=2):
    """
    Simulates trades based on entry/exit signals and stop-loss, generating a DataFrame of trades
    and an equity curve.

    Args:
        df (pd.DataFrame): DataFrame with OHLC, Supertrend, EMA, and signal columns
                           ('Open', 'High', 'Low', 'BUY_Entry', 'SELL_Entry',
                           'BUY_Exit', 'SELL_Exit', 'StopLoss').
        initial_capital (float): Starting capital for the simulation.
        risk_per_trade_pct (float): Percentage of capital to risk per trade (not directly used
                                    for trade sizing in this version, but kept for consistency).
        brokerage_pct (float): Brokerage fee as a percentage.
        slippage_points (float): Slippage in points for each transaction.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame of executed trades.
            - pd.DataFrame: DataFrame representing the equity curve over time.
    """
    trades = []
    position_type = None  # Tracks current position: None, 'long', or 'short'.
    entry_date = None
    entry_price = None
    stop_loss_price = None
    current_capital = initial_capital
    trade_size = 0 # Number of units/lots for the current trade.
    
    # Store capital at each step for equity curve visualization.
    equity_curve_data = []
    # Initialize equity curve with initial capital at the first date of the DataFrame.
    if not df.empty:
        equity_curve_data.append({"Date": df.index[0], "Capital": initial_capital})

    # Loop through the DataFrame starting from the second row (index 1)
    # to allow checking previous candle's signals for current candle's entry.
    for i in range(1, len(df)):
        current_date = df.index[i]
        current_open = df["Open"].iloc[i]
        current_low = df["Low"].iloc[i]
        current_high = df["High"].iloc[i]

        # Entry signals are based on the PREVIOUS candle's close and indicator values,
        # but the trade is executed at the CURRENT candle's open.
        prev_buy_entry = df["BUY_Entry"].iloc[i-1]
        prev_sell_entry = df["SELL_Entry"].iloc[i-1]

        # Exit signals are based on the CURRENT candle's conditions.
        buy_exit = df["BUY_Exit"].iloc[i]
        sell_exit = df["SELL_Exit"].iloc[i]

        # Stop loss for the current candle, used for open positions or new entries.
        current_stoploss = df["StopLoss"].iloc[i]

        # Update trailing stop loss for open positions.
        # For long positions, stop loss trails upwards (max of current and previous stop loss).
        if position_type == "long" and stop_loss_price is not None:
            if current_stoploss is not None and not np.isnan(current_stoploss):
                stop_loss_price = max(stop_loss_price, current_stoploss)
        # For short positions, stop loss trails downwards (min of current and previous stop loss).
        elif position_type == "short" and stop_loss_price is not None:
            if current_stoploss is not None and not np.isnan(current_stoploss):
                stop_loss_price = min(stop_loss_price, current_stoploss)

        # Check for stop-loss hit for an open long position.
        # If current low goes below or equals the stop loss, exit the trade.
        if position_type == "long" and stop_loss_price is not None and current_low <= stop_loss_price:
            exit_date = current_date
            exit_price = stop_loss_price # Exit at the stop-loss price.
            
            profit_loss = calculate_trade_profit(
                entry_price, exit_price, "long", trade_size, brokerage_pct, slippage_points
            )
            current_capital += profit_loss
            trades.append(
                {
                    "trade_type": "long",
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "profit_loss": profit_loss,
                    "trade_size": trade_size,
                    "exit_reason": "StopLoss",
                    "capital_after_trade": current_capital,
                }
            )
            # Reset position variables after closing the trade.
            position_type = None
            entry_date = None
            entry_price = None
            stop_loss_price = None
            trade_size = 0
            equity_curve_data.append({"Date": current_date, "Capital": current_capital})
            # Continue to the next iteration as a position was just closed.
            continue
        # Check for stop-loss hit for an open short position.
        # If current high goes above or equals the stop loss, exit the trade.
        elif position_type == "short" and stop_loss_price is not None and current_high >= stop_loss_price:
            exit_date = current_date
            exit_price = stop_loss_price # Exit at the stop-loss price.
            
            profit_loss = calculate_trade_profit(
                entry_price, exit_price, "short", trade_size, brokerage_pct, slippage_points
            )
            current_capital += profit_loss
            trades.append(
                {
                    "trade_type": "short",
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "profit_loss": profit_loss,
                    "trade_size": trade_size,
                    "exit_reason": "StopLoss",
                    "capital_after_trade": current_capital,
                }
            )
            # Reset position variables after closing the trade.
            position_type = None
            entry_date = None
            entry_price = None
            stop_loss_price = None
            trade_size = 0
            equity_curve_data.append({"Date": current_date, "Capital": current_capital})
            # Continue to the next iteration as a position was just closed.
            continue

        # If no position is open, check for new entry signals from the PREVIOUS candle.
        if position_type is None:
            trade_size = 1 # Fixed trade size as requested (e.g., 1 unit/lot).

            if prev_buy_entry: # If a buy signal was generated on the previous candle.
                position_type = "long"
                entry_date = current_date
                entry_price = current_open # Enter at the current candle's open.
                stop_loss_price = current_stoploss # Set initial stop loss.
            elif prev_sell_entry: # If a sell signal was generated on the previous candle.
                position_type = "short"
                entry_date = current_date
                entry_price = current_open # Enter at the current candle's open.
                stop_loss_price = current_stoploss # Set initial stop loss.
        # If a long position is open, check for exit signals.
        elif position_type == "long":
            if buy_exit: # If a buy exit signal is generated on the current candle.
                exit_date = current_date
                exit_price = current_open # Exit at the current candle's open.
                profit_loss = calculate_trade_profit(
                    entry_price, exit_price, "long", trade_size, brokerage_pct, slippage_points
                )
                current_capital += profit_loss
                trades.append(
                    {
                        "trade_type": "long",
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": exit_date,
                        "exit_price": exit_price,
                        "profit_loss": profit_loss,
                        "trade_size": trade_size,
                        "exit_reason": "Signal",
                        "capital_after_trade": current_capital,
                    }
                )
                equity_curve_data.append({"Date": current_date, "Capital": current_capital})

                # Check for immediate reversal: if a sell signal was present on the previous candle.
                if prev_sell_entry:
                    trade_size = 3 # Fixed trade size for reversal entry (e.g., 3 units/lots).
                    position_type = "short"
                    entry_date = current_date
                    entry_price = current_open
                    stop_loss_price = current_stoploss
                else:
                    # If no reversal, reset position variables.
                    position_type = None
                    stop_loss_price = None
                    trade_size = 0
        # If a short position is open, check for exit signals.
        elif position_type == "short":
            if sell_exit: # If a sell exit signal is generated on the current candle.
                exit_date = current_date
                exit_price = current_open # Exit at the current candle's open.
                profit_loss = calculate_trade_profit(
                    entry_price, exit_price, "short", trade_size, brokerage_pct, slippage_points
                )
                current_capital += profit_loss
                trades.append(
                    {
                        "trade_type": "short",
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": exit_date,
                        "exit_price": exit_price,
                        "profit_loss": profit_loss,
                        "trade_size": trade_size,
                        "exit_reason": "Signal",
                        "capital_after_trade": current_capital,
                    }
                )
                equity_curve_data.append({"Date": current_date, "Capital": current_capital})

                # Check for immediate reversal: if a buy signal was present on the previous candle.
                if prev_buy_entry:
                    trade_size = 3 # Fixed trade size for reversal entry.
                    position_type = "long"
                    entry_date = current_date
                    entry_price = current_open
                    stop_loss_price = current_stoploss
                else:
                    # If no reversal, reset position variables.
                    position_type = None
                    stop_loss_price = None
                    trade_size = 0
        
        # Append current capital to equity curve data for each date, even if no trade occurred.
        # This ensures a continuous equity curve.
        if not equity_curve_data or equity_curve_data[-1]["Date"] != current_date:
            equity_curve_data.append({"Date": current_date, "Capital": current_capital})


    # If still in a position at the end of the data, close the trade at the last available open price.
    if position_type is not None:
        exit_date = df.index[-1]
        exit_price = df["Open"].iloc[-1]
        profit_loss = calculate_trade_profit(
            entry_price, exit_price, position_type, trade_size, brokerage_pct, slippage_points
        )
        current_capital += profit_loss
        trades.append(
            {
                "trade_type": position_type,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "profit_loss": profit_loss,
                "trade_size": trade_size,
                "exit_reason": "EndOfData",
                "capital_after_trade": current_capital,
            }
        )
        equity_curve_data.append({"Date": exit_date, "Capital": current_capital})

    # Convert the list of trades and equity curve data into DataFrames.
    trades_df = pd.DataFrame(trades)
    equity_curve_df = pd.DataFrame(equity_curve_data).set_index("Date")
    return trades_df, equity_curve_df


def calculate_performance_metrics(trades_df, equity_curve_df, risk_free_rate=0.0):
    """
    Calculates various performance metrics for a trading strategy.

    Args:
        trades_df (pd.DataFrame): DataFrame containing details of executed trades.
        equity_curve_df (pd.DataFrame): DataFrame representing the equity curve over time.
        risk_free_rate (float): Annual risk-free rate for Sharpe Ratio calculation (default: 0.0).

    Returns:
        pd.DataFrame: A DataFrame with a single row containing all calculated metrics.
    """
    if trades_df.empty:
        # Return a DataFrame with zero values if no trades were executed.
        return pd.DataFrame(
            [
                {
                    "total_profit": 0,
                    "num_trades": 0,
                    "num_winning_trades": 0,
                    "num_losing_trades": 0,
                    "win_rate": 0,
                    "avg_profit_per_winning_trade": 0,
                    "avg_loss_per_losing_trade": 0,
                    "max_drawdown": 0,
                    "profit_factor": 0,
                    "sharpe_ratio": 0,
                    "final_capital": 0,
                }
            ]
        )

    total_profit = trades_df["profit_loss"].sum()
    num_trades = len(trades_df)

    winning_trades = trades_df[trades_df["profit_loss"] > 0]
    losing_trades = trades_df[trades_df["profit_loss"] < 0]

    num_winning_trades = len(winning_trades)
    num_losing_trades = len(losing_trades)

    win_rate = (num_winning_trades / num_trades) * 100 if num_trades > 0 else 0

    avg_profit_per_winning_trade = (
        winning_trades["profit_loss"].mean() if num_winning_trades > 0 else 0
    )
    avg_loss_per_losing_trade = (
        losing_trades["profit_loss"].mean() if num_losing_trades > 0 else 0
    )

    # Calculate Max Drawdown using the equity curve.
    # Max Drawdown is the largest percentage drop from a peak in equity to a subsequent trough.
    if not equity_curve_df.empty:
        equity_curve = equity_curve_df["Capital"]
        peak = equity_curve.cummax() # Cumulative maximum capital up to each point.
        drawdown_values = (peak - equity_curve) / peak # Percentage drop from peak.
        max_drawdown = drawdown_values.max() * 100 if not drawdown_values.empty else 0
    else:
        max_drawdown = 0

    # Profit Factor: Ratio of total gross profit to total gross loss.
    # A value greater than 1 indicates a profitable strategy.
    total_gross_profit = winning_trades["profit_loss"].sum()
    total_gross_loss = abs(losing_trades["profit_loss"].sum())
    profit_factor = (
        total_gross_profit / total_gross_loss if total_gross_loss > 0 else np.inf
    )

    # Sharpe Ratio: Measures risk-adjusted return.
    # It indicates the average return earned in excess of the risk-free rate per unit of volatility.
    if not equity_curve_df.empty and len(equity_curve_df) > 1:
        # Calculate daily returns from the equity curve.
        returns = equity_curve_df["Capital"].pct_change().dropna()
        if returns.std() > 0:
            # Annualize Sharpe Ratio (assuming 252 trading days in a year).
            sharpe_ratio = (
                (returns.mean() - risk_free_rate / 252)
                / returns.std()
                * np.sqrt(252)
            )
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0

    # Final capital at the end of the simulation.
    final_capital = equity_curve_df["Capital"].iloc[-1] if not equity_curve_df.empty else 0

    metrics = {
        "total_profit": total_profit,
        "num_trades": num_trades,
        "num_winning_trades": num_winning_trades,
        "num_losing_trades": num_losing_trades,
        "win_rate": win_rate,
        "avg_profit_per_winning_trade": avg_profit_per_winning_trade,
        "avg_loss_per_losing_trade": avg_loss_per_losing_trade,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "final_capital": final_capital,
    }
    return pd.DataFrame([metrics])


def calculate_composite_score(metrics_df, initial_capital=100000):
    """
    Calculates a composite score for a trading strategy based on various performance metrics.
    This score is used to rank strategies during optimization.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing performance metrics (single row).
        initial_capital (float): The initial capital used for the simulation.

    Returns:
        dict: A dictionary containing normalized metrics and the preliminary composite score.
    """
    if metrics_df.empty:
        return {
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "inverse_max_drawdown": 0.0,
            "win_rate": 0.0,
            "normalized_total_profit": 0.0,
            "normalized_final_capital": 0.0,
            "preliminary_composite_score": 0.0,
        }

    metrics = metrics_df.iloc[0]

    # Calculate inverse of Max Drawdown.
    # This metric is designed so that higher values are better (lower drawdown).
    # Special handling for zero drawdown: if profitable, assign a high value; otherwise, a penalty.
    inverse_max_drawdown = 0
    if metrics["max_drawdown"] > 0:
        inverse_max_drawdown = 1 / metrics["max_drawdown"]
    elif metrics["max_drawdown"] == 0 and metrics["total_profit"] > 0:
        inverse_max_drawdown = 1  # High value if no drawdown and profitable.
    else:
        inverse_max_drawdown = -1 # Penalty if losing and no drawdown (implies no trades or constant loss).

    # Normalize total profit by initial capital to get a relative profit.
    normalized_total_profit = metrics["total_profit"] / initial_capital
    
    # Normalize final capital by initial capital to get a relative final capital.
    normalized_final_capital = metrics["final_capital"] / initial_capital

    # Define weights for each metric in the composite score.
    # These weights determine the importance of each metric in the overall score.
    weights = {
        "sharpe_ratio": 0.20,
        "profit_factor": 0.20,
        "inverse_max_drawdown": 0.20,
        "win_rate": 0.20,
        "normalized_total_profit": 0.10,
        "normalized_final_capital": 0.10,
    }

    # Calculate preliminary normalized scores for each component.
    # These are heuristic normalizations for a single strategy run,
    # mapping values to a 0-1 range for consistent scoring.

    # Sharpe Ratio normalization: Shift negative values to positive range and clamp.
    prelim_sharpe_norm = (
        (metrics["sharpe_ratio"] + 10) / 20 if metrics["sharpe_ratio"] > -10 else 0
    )
    prelim_sharpe_norm = max(0, min(1, prelim_sharpe_norm))

    # Profit Factor normalization: Cap at 10 for scaling and clamp.
    prelim_profit_factor_norm = (
        min(10, metrics["profit_factor"]) / 10
        if metrics["profit_factor"] != np.inf
        else 1
    )
    prelim_profit_factor_norm = max(0, min(1, prelim_profit_factor_norm))

    # Inverse Max Drawdown normalization: Map -1 to 0, scale others.
    prelim_inverse_max_drawdown_norm = 0
    if inverse_max_drawdown == 1:
        prelim_inverse_max_drawdown_norm = 1
    elif inverse_max_drawdown > 0:
        prelim_inverse_max_drawdown_norm = min(1, inverse_max_drawdown / 10)

    # Win Rate normalization: Already 0-100, normalize to 0-1.
    prelim_win_rate_norm = metrics["win_rate"] / 100

    # Normalized Total Profit normalization: Shift negative values to positive range and clamp.
    prelim_normalized_total_profit_norm = (
        (normalized_total_profit + 1) / 2 if normalized_total_profit > -1 else 0
    )
    prelim_normalized_total_profit_norm = max(
        0, min(1, prelim_normalized_total_profit_norm)
    )

    # Normalized Final Capital normalization: Shift negative values to positive range and clamp.
    prelim_normalized_final_capital_norm = (
        (normalized_final_capital + 1) / 2 if normalized_final_capital > -1 else 0
    )
    prelim_normalized_final_capital_norm = max(
        0, min(1, prelim_normalized_final_capital_norm)
    )

    # Calculate the preliminary composite score using weighted sum of normalized components.
    preliminary_composite_score = (
        prelim_sharpe_norm * weights["sharpe_ratio"]
        + prelim_profit_factor_norm * weights["profit_factor"]
        + prelim_inverse_max_drawdown_norm * weights["inverse_max_drawdown"]
        + prelim_win_rate_norm * weights["win_rate"]
        + prelim_normalized_total_profit_norm * weights["normalized_total_profit"]
        + prelim_normalized_final_capital_norm * weights["normalized_final_capital"]
    )

    return {
        "sharpe_ratio": metrics["sharpe_ratio"],
        "profit_factor": metrics["profit_factor"],
        "inverse_max_drawdown": inverse_max_drawdown,
        "win_rate": metrics["win_rate"],
        "normalized_total_profit": normalized_total_profit,
        "normalized_final_capital": normalized_final_capital,
        "preliminary_composite_score": preliminary_composite_score,
    }


def run_optimization(
    base_df,
    period_ranges,
    multiplier_ranges,
    ema_fast_period_range,
    ema_slow_period_range,
    stoploss_atr_multiplier_range,
    risk_per_trade_pct_range,
    initial_capital=100000,
    brokerage_pct=0.0003,
    slippage_points=2,
    num_iterations=2, # Added num_iterations as a parameter
):
    """
    Runs an optimization process to find the best Supertrend and EMA strategy parameters.
    It iterates through random combinations of parameters, calculates performance metrics,
    and identifies the best performing strategy based on a composite score.

    Args:
        base_df (pd.DataFrame): The base DataFrame with OHLC and Heikin Ashi data.
        period_ranges (list): List of tuples, each defining (start, end, step) for Supertrend periods.
                              Expected format: [(p1_start, p1_end, p1_step), (p2_start, p2_end, p2_step)].
        multiplier_ranges (list): List of tuples, each defining (start, end, step) for Supertrend multipliers.
                                  Expected format: [(m1_start, m1_end, m1_step), (m2_start, m2_end, m2_step)].
        ema_fast_period_range (tuple): (start, end, step) for fast EMA periods.
        ema_slow_period_range (tuple): (start, end, step) for slow EMA periods.
        stoploss_atr_multiplier_range (tuple): (start, end, step) for stop-loss ATR multiplier.
        risk_per_trade_pct_range (tuple): (start, end, step) for risk per trade percentage.
        initial_capital (float): Starting capital for each simulation run.
        brokerage_pct (float): Brokerage fee percentage.
        slippage_points (float): Slippage in points.
        num_iterations (int): Number of random parameter combinations to test (default: 2).

    Returns:
        pd.DataFrame: A DataFrame containing results for all tested parameter combinations,
                      including performance metrics and composite scores.
    """
    all_results = []

    # Unpack parameter ranges for easier access.
    p1_start, p1_end, p1_step = period_ranges[0]
    m1_start, m1_end, m1_step = multiplier_ranges[0]
    p2_start, p2_end, p2_step = period_ranges[1]
    m2_start, m2_end, m2_step = multiplier_ranges[1]
    ema_fast_p_start, ema_fast_p_end, ema_fast_p_step = ema_fast_period_range
    ema_slow_p_start, ema_slow_p_end, ema_slow_p_step = ema_slow_period_range
    sl_atr_mult_start, sl_atr_mult_end, sl_atr_mult_step = stoploss_atr_multiplier_range
    risk_pct_start, risk_pct_end, risk_pct_step = risk_per_trade_pct_range

    # Generate lists of possible values for each parameter.
    periods1 = list(range(p1_start, p1_end + 1, p1_step))
    multipliers1 = list(np.round(np.arange(m1_start, m1_end + m1_step, m1_step), 1))
    periods2 = list(range(p2_start, p2_end + 1, p2_step))
    multipliers2 = list(np.round(np.arange(m2_start, m2_end + m2_step, m2_step), 1))
    ema_fast_periods = list(range(ema_fast_p_start, ema_fast_p_end + 1, ema_fast_p_step))
    ema_slow_periods = list(range(ema_slow_p_start, ema_slow_p_end + 1, ema_slow_p_step))
    stoploss_atr_multipliers = list(
        np.round(
            np.arange(sl_atr_mult_start, sl_atr_mult_end + sl_atr_mult_step, sl_atr_mult_step),
            2,
        )
    )
    risk_per_trade_pcts = list(
        np.round(
            np.arange(risk_pct_start, risk_pct_end + risk_pct_step, risk_pct_step),
            3,
        )
    )

    print(f"Starting optimization with {num_iterations} random combinations...")

    # Loop for the specified number of iterations to test random parameter combinations.
    for _ in range(num_iterations):
        # Randomly select one value for each parameter from its respective range.
        p1 = random.choice(periods1)
        m1 = random.choice(multipliers1)
        p2 = random.choice(periods2)
        m2 = random.choice(multipliers2)
        ema_fast_p = random.choice(ema_fast_periods)
        ema_slow_p = random.choice(ema_slow_periods)
        sl_atr_mult = random.choice(stoploss_atr_multipliers)
        risk_pct = random.choice(risk_per_trade_pcts)

        # Create a copy of the base DataFrame for each iteration to avoid modifying the original.
        df_copy = base_df.copy()

        # --- Calculate ATR for StopLoss (using standard OHLC for this specific ATR) ---
        # This ATR is used for the stop-loss calculation, not directly for Supertrend bands.
        # The `use_heikin_ashi` parameter in `calculate_supertrend` will control that.
        high_for_atr = df_copy["High"]
        low_for_atr = df_copy["Low"]
        close_for_atr = df_copy["Close"]

        df_copy["TR"] = np.maximum.reduce(
            [
                high_for_atr - low_for_atr,
                abs(high_for_atr - close_for_atr.shift(1)),
                abs(low_for_atr - close_for_atr.shift(1)),
            ]
        )
        # Use p1 as the period for this general ATR calculation.
        df_copy["ATR"] = df_copy["TR"].ewm(alpha=1/p1, adjust=False).mean()

        # --- Calculate SuperTrend columns for two different sets of parameters ---
        # Supertrend 1
        st_1_name = f"ST_{p1}_{str(m1).replace('.', '_')}"
        st_1_dir_name = f"ST_Dir_{p1}_{str(m1).replace('.', '_')}"
        st_1_trending_up_name = f"trendingUp_{p1}_{str(m1).replace('.', '_')}"
        st_1_trending_down_name = f"trendingDown_{p1}_{str(m1).replace('.', '_')}"
        st_1_df_result = calculate_supertrend(
            base_df.copy(), # Pass a fresh copy to avoid side effects
            period=p1,
            multiplier=m1,
            trending_up_col_name=st_1_trending_up_name,
            trending_down_col_name=st_1_trending_down_name,
            use_heikin_ashi=True, # Explicitly use Heikin Ashi for Supertrend calculation
        )

        # Supertrend 2
        st_2_name = f"ST_{p2}_{str(m2).replace('.', '_')}"
        st_2_dir_name = f"ST_Dir_{p2}_{str(m2).replace('.', '_')}"
        st_2_trending_up_name = f"trendingUp_{p2}_{str(m2).replace('.', '_')}"
        st_2_trending_down_name = f"trendingDown_{p2}_{str(m2).replace('.', '_')}"
        st_2_df_result = calculate_supertrend(
            base_df.copy(), # Pass a fresh copy
            period=p2,
            multiplier=m2,
            trending_up_col_name=st_2_trending_up_name,
            trending_down_col_name=st_2_trending_down_name,
            use_heikin_ashi=True, # Explicitly use Heikin Ashi for Supertrend calculation
        )

        # --- Calculate EMAs ---
        ema_fast_name = f"EMA_Fast_{ema_fast_p}"
        ema_slow_name = f"EMA_Slow_{ema_slow_p}"
        df_copy[ema_fast_name] = calculate_ema(
            base_df.copy(), period=ema_fast_p, price_col="Close" # Use original Close for EMA
        )
        df_copy[ema_slow_name] = calculate_ema(
            base_df.copy(), period=ema_slow_p, price_col="Close" # Use original Close for EMA
        )

        # Join the calculated Supertrend results back to the main DataFrame copy.
        df_copy = df_copy.join(st_1_df_result)
        # Only join the second Supertrend if its parameters are different from the first.
        if st_1_name != st_2_name:
            df_copy = df_copy.join(st_2_df_result)

        # --- Generate BUY/SELL Entry and Exit Signals ---
        # BUY Entry Signal:
        # - Previous HA_Close was below both Supertrend 1 and Supertrend 2.
        # - Current HA_Close is above both Supertrend 1 and Supertrend 2.
        # - Both Supertrend directions are upward (1).
        # - Fast EMA is above Slow EMA (EMA crossover for confirmation).
        df_copy["BUY_Entry"] = (
            (df_copy["HA_Close"].shift(1) < df_copy[st_1_name].shift(1))
            & (df_copy["HA_Close"].shift(1) < df_copy[st_2_name].shift(1))
            & (df_copy["HA_Close"] > df_copy[st_1_name])
            & (df_copy["HA_Close"] > df_copy[st_2_name])
            & (df_copy[st_1_dir_name] == 1)
            & (df_copy[st_2_dir_name] == 1)
            & (df_copy[ema_fast_name] > df_copy[ema_slow_name])
        )

        # Calculate StopLoss based on the lower period supertrend (assuming p1 is lower period).
        # For an upward trend (direction 1), stop loss is trendingUp band minus ATR multiplier * ATR.
        # For a downward trend (direction -1), stop loss is trendingDown band plus ATR multiplier * ATR.
        df_copy["StopLoss"] = np.where(
            df_copy[st_1_dir_name] == 1,
            df_copy[st_1_trending_up_name] - sl_atr_mult * df_copy["ATR"],
            df_copy[st_1_trending_down_name] + sl_atr_mult * df_copy["ATR"],
        )

        # BUY Exit Signal:
        # - Previous HA_Close was above Supertrend 1 AND current HA_Close is below Supertrend 1 AND Supertrend 1 direction is downward (-1)
        # OR
        # - Previous HA_Close was above Supertrend 2 AND current HA_Close is below Supertrend 2 AND Supertrend 2 direction is downward (-1)
        df_copy["BUY_Exit"] = (
            (
                (df_copy["HA_Close"].shift(1) > df_copy[st_1_name].shift(1))
                & (df_copy["HA_Close"] < df_copy[st_1_name])
                & (df_copy[st_1_dir_name] == -1)
            )
            | (
                (df_copy["HA_Close"].shift(1) > df_copy[st_2_name].shift(1))
                & (df_copy["HA_Close"] < df_copy[st_2_name])
                & (df_copy[st_2_dir_name] == -1)
            )
        )

        # SELL Entry Signal:
        # - Previous HA_Close was above both Supertrend 1 and Supertrend 2.
        # - Current HA_Close is below both Supertrend 1 and Supertrend 2.
        # - Both Supertrend directions are downward (-1).
        # - Fast EMA is below Slow EMA (EMA crossover for confirmation).
        df_copy["SELL_Entry"] = (
            (df_copy["HA_Close"].shift(1) > df_copy[st_1_name].shift(1))
            & (df_copy["HA_Close"].shift(1) > df_copy[st_2_name].shift(1))
            & (df_copy["HA_Close"] < df_copy[st_1_name])
            & (df_copy["HA_Close"] < df_copy[st_2_name])
            & (df_copy[st_1_dir_name] == -1)
            & (df_copy[st_2_dir_name] == -1)
            & (df_copy[ema_fast_name] < df_copy[ema_slow_name])
        )

        # SELL Exit Signal:
        # - Previous HA_Close was below Supertrend 1 AND current HA_Close is above Supertrend 1 AND Supertrend 1 direction is upward (1)
        # OR
        # - Previous HA_Close was below Supertrend 2 AND current HA_Close is above Supertrend 2 AND Supertrend 2 direction is upward (1)
        df_copy["SELL_Exit"] = (
            (
                (df_copy["HA_Close"].shift(1) < df_copy[st_1_name].shift(1))
                & (df_copy["HA_Close"] > df_copy[st_1_name])
                & (df_copy[st_1_dir_name] == 1)
            )
            | (
                (df_copy["HA_Close"].shift(1) < df_copy[st_2_name].shift(1))
                & (df_copy["HA_Close"] > df_copy[st_2_name])
                & (df_copy[st_2_dir_name] == 1)
            )
        )

        # Round all numeric columns to 2 decimal places for cleaner data.
        df_copy = df_copy.round(2)

        # Remove rows with NaN values, especially at the beginning due to indicator calculations.
        df_copy.dropna(inplace=True)

        # Save the DataFrame to a temporary CSV file for debugging purposes.
        df_copy.to_csv("debug_signals.csv", index=True)
        
        # If the DataFrame is not empty after dropping NaNs, proceed with trade generation and metric calculation.
        if not df_copy.empty:
            trades_df, equity_curve_df = generate_trades_df(
                df_copy,
                initial_capital=initial_capital,
                risk_per_trade_pct=risk_pct,
                brokerage_pct=brokerage_pct,
                slippage_points=slippage_points,
            )
            performance_metrics = calculate_performance_metrics(trades_df, equity_curve_df)
            composite_metrics = calculate_composite_score(
                performance_metrics, initial_capital
            )

            # Prepare the result row for this parameter combination.
            result_row = {
                "period1": p1,
                "multiplier1": m1,
                "period2": p2,
                "multiplier2": m2,
                "ema_fast_period": ema_fast_p,
                "ema_slow_period": ema_slow_p,
                "stoploss_atr_multiplier": sl_atr_mult,
                "risk_per_trade_pct": risk_pct,
                **performance_metrics.iloc[0].to_dict(), # Unpack all performance metrics.
                **{
                    k: v
                    for k, v in composite_metrics.items()
                    if k != "preliminary_composite_score"
                }, # Unpack composite score components except the final score itself.
            }
            # Add the preliminary composite score to the result row.
            result_row["preliminary_composite_score"] = composite_metrics[
                "preliminary_composite_score"
            ]
            all_results.append(result_row)

            # Print summary for strategies that meet certain criteria (e.g., high profit factor, win rate, or composite score).
            if (
                performance_metrics["profit_factor"].iloc[0] > 2
                and performance_metrics["win_rate"].iloc[0] >= 70
            ) or (composite_metrics["preliminary_composite_score"] > 0.7):
                print(
                    f"  Found strategy: P1={p1}, M1={m1}, P2={p2}, M2={m2}, EMA_Fast_P={ema_fast_p}, EMA_Slow_P={ema_slow_p}, SL_ATR_Mult={sl_atr_mult}, Risk_Pct={risk_pct}"
                )
                print(
                    f"    Total Profit: {performance_metrics['total_profit'].iloc[0]:.2f}"
                )
                print(
                    f"    Profit Factor: {performance_metrics['profit_factor'].iloc[0]:.2f}"
                )
                print(f"    Win Rate: {performance_metrics['win_rate'].iloc[0]:.2f}%")
                print(
                    f"    Preliminary Composite Score: {composite_metrics['preliminary_composite_score']:.4f}"
                )
                if not (performance_metrics["profit_factor"].iloc[0] > 2 and performance_metrics["win_rate"].iloc[0] > 50):
                    print("    (Borderline-good strategy based on composite score)")

    # Convert all collected results into a single DataFrame.
    results_df = pd.DataFrame(all_results)

    # Normalize metrics across all runs for a final, consistent composite score calculation.
    if not results_df.empty:
        # Replace infinite profit factor values with NaN for proper normalization.
        results_df["profit_factor"] = results_df["profit_factor"].replace(
            [np.inf, -np.inf], np.nan
        )

        # Min-Max Normalization for each metric to scale them between 0 and 1.
        # This is crucial for combining different metrics into a single score.
        
        # Sharpe Ratio normalization.
        min_sharpe = results_df["sharpe_ratio"].min()
        max_sharpe = results_df["sharpe_ratio"].max()
        if max_sharpe == min_sharpe:
            results_df["sharpe_ratio_norm"] = 0.0
        else:
            results_df["sharpe_ratio_norm"] = (
                results_df["sharpe_ratio"] - min_sharpe
            ) / (max_sharpe - min_sharpe)

        # Profit Factor normalization.
        min_pf = results_df["profit_factor"].min()
        max_pf = results_df["profit_factor"].max()
        if max_pf == min_pf:
            results_df["profit_factor_norm"] = 0.0
        else:
            results_df["profit_factor_norm"] = (
                results_df["profit_factor"] - min_pf
            ) / (max_pf - min_pf)

        # Inverse Max Drawdown normalization.
        min_inv_dd = results_df["inverse_max_drawdown"].min()
        max_inv_dd = results_df["inverse_max_drawdown"].max()
        if max_inv_dd == min_inv_dd:
            results_df["inverse_max_drawdown_norm"] = 0.0
        else:
            results_df["inverse_max_drawdown_norm"] = (
                results_df["inverse_max_drawdown"] - min_inv_dd
            ) / (max_inv_dd - min_inv_dd)

        # Win Rate normalization (already 0-100, scale to 0-1).
        results_df["win_rate_norm"] = results_df["win_rate"] / 100

        # Normalized Total Profit normalization.
        min_norm_profit = results_df["normalized_total_profit"].min()
        max_norm_profit = results_df["normalized_total_profit"].max()
        if max_norm_profit == min_norm_profit:
            results_df["normalized_total_profit_norm"] = 0.0
        else:
            results_df["normalized_total_profit_norm"] = (
                results_df["normalized_total_profit"] - min_norm_profit
            ) / (max_norm_profit - min_norm_profit)

        # Normalized Final Capital normalization.
        min_norm_final_capital = results_df["normalized_final_capital"].min()
        max_norm_final_capital = results_df["normalized_final_capital"].max()
        if max_norm_final_capital == min_norm_final_capital:
            results_df["normalized_final_capital_norm"] = 0.0
        else:
            results_df["normalized_final_capital_norm"] = (
                results_df["normalized_final_capital"] - min_norm_final_capital
            ) / (max_norm_final_capital - min_norm_final_capital)

        # Fill any remaining NaNs after normalization with 0 (e.g., if a metric was constant across all runs).
        results_df = results_df.fillna(0)

        # Define weights for the final composite score calculation.
        weights = {
                "sharpe_ratio": 0.20,
                "profit_factor": 0.20,
                "inverse_max_drawdown": 0.20,
                "win_rate": 0.20,
                "normalized_total_profit": 0.10,
                "normalized_final_capital": 0.10,
            }

        # Calculate the final composite score using the normalized metrics and defined weights.
        results_df["composite_score"] = (
            results_df["sharpe_ratio_norm"] * weights["sharpe_ratio"]
            + results_df["profit_factor_norm"] * weights["profit_factor"]
            + results_df["inverse_max_drawdown_norm"]
            * weights["inverse_max_drawdown"]
            + results_df["win_rate_norm"] * weights["win_rate"]
            + results_df["normalized_total_profit_norm"]
            * weights["normalized_total_profit"]
            + results_df["normalized_final_capital_norm"]
            * weights["normalized_final_capital"]
        )

    return results_df


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the path to the historical data CSV file.
    file_path = (
        r"D:\Sushant\Fyers_AlgoTrade\Fyers_Data\Nifty50_Index\NIFTY50_INDEX_D_Min.csv"
    )

    try:
        # Load the data.
        df = pd.read_csv(file_path)
        
        # Drop unnecessary columns, ignoring errors if columns don't exist.
        df = df.drop(columns=["Unnamed: 0", "Volume"], errors="ignore")
        
        # Convert 'Date' column to datetime objects and normalize to remove time component.
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        
        # Remove duplicate date entries, keeping the first occurrence.
        initial_rows_before_dedup = len(df)
        df.drop_duplicates(subset=["Date"], keep="first", inplace=True)
        rows_removed_dedup = initial_rows_before_dedup - len(df)
        if rows_removed_dedup > 0:
            print(f"Removed {rows_removed_dedup} duplicate date rows.")

        # Set 'Date' column as the DataFrame index.
        df = df.set_index("Date")

        # Filter data to include only entries from '2021-01-01' onwards.
        df = df[df.index >= "2021-01-01"]
        # Example: df = df[df.index < "2022-01-01"] # Uncomment to filter for a specific year

        # Store original index before resetting for Heikin Ashi calculation.
        original_index = df.index
        df = df.reset_index(drop=True)

        # Calculate Heikin Ashi candles and join them back to the main DataFrame.
        ha_df = calculate_heikin_ashi(
            df.copy() # Pass a copy to avoid modifying the original DataFrame during HA calculation.
        )
        df = df.join(ha_df) # Join HA columns back to the original DataFrame.

        # Restore the original index after Heikin Ashi calculation.
        df.index = original_index

        # Define optimization ranges for Supertrend periods, multipliers, EMA periods,
        # stop-loss ATR multiplier, and risk per trade percentage.
        # These ranges define the search space for the optimization algorithm.
        period_ranges = [(8, 8, 1), (8, 8, 1)] # Example: (start, end, step) for period 1 and period 2
        multiplier_ranges = [(1.0, 1.0, 1), (1.0, 1.0, 1)] # Example: (start, end, step) for multiplier 1 and multiplier 2
        ema_fast_period_range = (20, 20, 1) # Example: (start, end, step) for fast EMA
        ema_slow_period_range = (50, 50, 1) # Example: (start, end, step) for slow EMA
        stoploss_atr_multiplier_range = (2.0, 2.0, 1) # Example: (start, end, step) for stop-loss ATR multiplier
        risk_per_trade_pct_range = (0.02, 0.02, 0.01) # Example: (start, end, step) for risk per trade percentage

        # Run the optimization process.
        optimization_results = run_optimization(
            df.copy(), # Pass a copy of the DataFrame to the optimization function.
            period_ranges,
            multiplier_ranges,
            ema_fast_period_range,
            ema_slow_period_range,
            stoploss_atr_multiplier_range,
            risk_per_trade_pct_range,
            initial_capital=initial_capital,
            brokerage_pct=0.0003,
            slippage_points=2,
            num_iterations=2, # Number of random combinations to test.
        )

        print("\nOptimization Results:")
        if not optimization_results.empty:
            # Sort the optimization results by the composite score in descending order
            # and print the top 20 strategies.
            print(
                optimization_results.sort_values(
                    by="composite_score", ascending=False
                ).head(20)
            )
            # Save all optimization results to a CSV file.
            optimization_results.to_csv("supertrend_optimization_results.csv", index=False)
            print("\nOptimization results saved to 'supertrend_optimization_results.csv'")

            # Identify the best strategy based on the highest composite score.
            best_params = optimization_results.loc[
                optimization_results["composite_score"].idxmax()
            ]

            # Extract best EMA periods (converted to int as they are periods).
            best_ema_fast_p = int(best_params["ema_fast_period"])
            best_ema_slow_p = int(best_params["ema_slow_period"])

            # Create a fresh copy of the DataFrame for the best strategy run.
            best_df_copy = df.copy()

            # --- Recalculate ATR for StopLoss for the best strategy ---
            # This ATR is used for the stop-loss calculation, not directly for Supertrend bands.
            high_for_atr_best = best_df_copy["High"]
            low_for_atr_best = best_df_copy["Low"]
            close_for_atr_best = best_df_copy["Close"]

            best_df_copy["TR"] = np.maximum.reduce(
                [
                    high_for_atr_best - low_for_atr_best,
                    abs(high_for_atr_best - close_for_atr_best.shift(1)),
                    abs(low_for_atr_best - close_for_atr_best.shift(1)),
                ]
            )

            # Use the best period1 for this general ATR calculation.
            period_best_atr = int(best_params['period1'])
            best_df_copy["ATR"] = best_df_copy["TR"].ewm(alpha=1/period_best_atr, adjust=False).mean()

            # --- Recalculate Supertrend columns for the best strategy parameters ---
            # Supertrend 1 for best parameters
            st_1_best_name = f"ST_{int(best_params['period1'])}_{str(best_params['multiplier1']).replace('.', '_')}"
            st_1_best_dir_name = f"ST_Dir_{int(best_params['period1'])}_{str(best_params['multiplier1']).replace('.', '_')}"
            st_1_best_trending_up_name = f"trendingUp_{int(best_params['period1'])}_{str(best_params['multiplier1']).replace('.', '_')}"
            st_1_best_trending_down_name = f"trendingDown_{int(best_params['period1'])}_{str(best_params['multiplier1']).replace('.', '_')}"
            st_1_best_df = calculate_supertrend(
                df.copy(), # Pass a fresh copy
                period=int(best_params["period1"]),
                multiplier=best_params["multiplier1"],
                trending_up_col_name=st_1_best_trending_up_name,
                trending_down_col_name=st_1_best_trending_down_name,
                use_heikin_ashi=True, # Explicitly use Heikin Ashi
            )
            best_df_copy = best_df_copy.join(st_1_best_df)

            # Supertrend 2 for best parameters
            st_2_best_name = f"ST_{int(best_params['period2'])}_{str(best_params['multiplier2']).replace('.', '_')}"
            st_2_best_dir_name = f"ST_Dir_{int(best_params['period2'])}_{str(best_params['multiplier2']).replace('.', '_')}"
            st_2_best_trending_up_name = f"trendingUp_{int(best_params['period2'])}_{str(best_params['multiplier2']).replace('.', '_')}"
            st_2_best_trending_down_name = f"trendingDown_{int(best_params['period2'])}_{str(best_params['multiplier2']).replace('.', '_')}"
            st_2_best_df = calculate_supertrend(
                df.copy(), # Pass a fresh copy
                period=int(best_params["period2"]),
                multiplier=best_params["multiplier2"],
                trending_up_col_name=st_2_best_trending_up_name,
                trending_down_col_name=st_2_best_trending_down_name,
                use_heikin_ashi=True, # Explicitly use Heikin Ashi
            )
            if st_1_best_name != st_2_best_name:
                best_df_copy = best_df_copy.join(st_2_best_df)

            # --- Recalculate EMAs for the best parameters ---
            ema_fast_best_name = f"EMA_Fast_{best_ema_fast_p}"
            ema_slow_best_name = f"EMA_Slow_{best_ema_slow_p}"
            best_df_copy[ema_fast_best_name] = calculate_ema(
                best_df_copy, period=best_ema_fast_p, price_col="Close"
            )
            best_df_copy[ema_slow_best_name] = calculate_ema(
                best_df_copy, period=best_ema_slow_p, price_col="Close"
            )

            # --- Regenerate BUY/SELL Entry and Exit Signals for the best strategy ---
            best_df_copy["BUY_Entry"] = (
                (best_df_copy["HA_Close"].shift(1) < best_df_copy[st_1_best_name].shift(1))
                & (
                    best_df_copy["HA_Close"].shift(1)
                    < best_df_copy[st_2_best_name].shift(1)
                )
                & (best_df_copy["HA_Close"] > best_df_copy[st_1_best_name])
                & (best_df_copy["HA_Close"] > best_df_copy[st_2_best_name])
                & (best_df_copy[st_1_best_dir_name] == 1)
                & (best_df_copy[st_2_best_dir_name] == 1)
                & (best_df_copy[ema_fast_best_name] > best_df_copy[ema_slow_best_name])
            )

            # Calculate StopLoss for the best strategy.
            best_df_copy["StopLoss"] = np.where(
                best_df_copy[st_1_best_dir_name] == 1,
                best_df_copy[st_1_best_trending_up_name]
                - best_params["stoploss_atr_multiplier"] * best_df_copy["ATR"],
                best_df_copy[st_1_best_trending_down_name]
                + best_params["stoploss_atr_multiplier"] * best_df_copy["ATR"],
            )

            best_df_copy["BUY_Exit"] = (
                (best_df_copy["HA_Close"].shift(1) > best_df_copy[st_1_best_name].shift(1))
                & (best_df_copy["HA_Close"] < best_df_copy[st_1_best_name])
                & (best_df_copy[st_1_best_dir_name] == -1)
            ) | (
                (
                    best_df_copy["HA_Close"].shift(1)
                    > best_df_copy[st_2_best_name].shift(1)
                )
                & (best_df_copy["HA_Close"] < best_df_copy[st_2_best_name])
                & (best_df_copy[st_2_best_dir_name] == -1)
            )

            best_df_copy["SELL_Entry"] = (
                (best_df_copy["HA_Close"].shift(1) > best_df_copy[st_1_best_name].shift(1))
                & (
                    best_df_copy["HA_Close"].shift(1)
                    > best_df_copy[st_2_best_name].shift(1)
                )
                & (best_df_copy["HA_Close"] < best_df_copy[st_1_best_name])
                & (best_df_copy["HA_Close"] < best_df_copy[st_2_best_name])
                & (best_df_copy[st_1_best_dir_name] == -1)
                & (best_df_copy[st_2_best_dir_name] == -1)
                & (best_df_copy[ema_fast_best_name] < best_df_copy[ema_slow_best_name])
            )

            best_df_copy["SELL_Exit"] = (
                (best_df_copy["HA_Close"].shift(1) < best_df_copy[st_1_best_name].shift(1))
                & (best_df_copy["HA_Close"] > best_df_copy[st_1_best_name])
                & (best_df_copy[st_1_best_dir_name] == 1)
            ) | (
                (
                    best_df_copy["HA_Close"].shift(1)
                    < best_df_copy[st_2_best_name].shift(1)
                )
                & (best_df_copy["HA_Close"] > best_df_copy[st_2_best_name])
                & (best_df_copy[st_2_best_dir_name] == 1)
            )

            # Generate trades and performance metrics for the best strategy.
            best_trades_df, best_equity_curve_df = generate_trades_df(
                best_df_copy,
                initial_capital=initial_capital,
                risk_per_trade_pct=best_params["risk_per_trade_pct"],
                brokerage_pct=0.0003,
                slippage_points=2,
            )
            best_performance_metrics = calculate_performance_metrics(best_trades_df, best_equity_curve_df)

            # Save the trades and performance metrics of the best strategy to CSV files.
            best_trades_df.to_csv("supertrend_best_trades.csv", index=False)
            print("\nBest strategy trades saved to 'supertrend_best_trades.csv'")

            best_performance_metrics.to_csv(
                "supertrend_best_performance_metrics.csv", index=False
            )
            print(
                "Best strategy performance metrics saved to 'supertrend_best_performance_metrics.csv'"
            )

        else:
            print("No optimization results generated.")

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
