import pandas as pd
import numpy as np
import random


initial_capital = 100000 # Define initial_capital globally

def calculate_supertrend(
    df,
    period=10,
    multiplier=3.0,
    trending_up_col_name="trendingUp",
    trending_down_col_name="trendingDown",
):
    # Store original index to restore later
    original_index = df.index
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

    use_heikin_ashi = True  # Use Heikin Ashi for Supertrend calculation

    if use_heikin_ashi:
        high = df["HA_High"]
        low = df["HA_Low"]
        close = df["HA_Close"]
    else:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        
    df["TR"] = np.maximum.reduce(
        [
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1)),
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

    return ha_df


def calculate_ema(df, period=50, price_col="Close"):
    """
    Calculates the Exponential Moving Average (EMA) for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'Close' column.
        period (int): The period for the EMA calculation.
        price_col (str): The name of the column to use for EMA calculation (default: 'Close').

    Returns:
        pd.Series: A Series containing the EMA values.
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
    adjusted_entry_price = entry_price
    adjusted_exit_price = exit_price

    if trade_type == "long":
        # For long entry, we buy at a higher price due to slippage
        adjusted_entry_price = entry_price + slippage_points
        # For long exit (selling), we sell at a lower price due to slippage
        adjusted_exit_price = exit_price - slippage_points
    elif trade_type == "short":
        # For short entry (selling), we sell at a lower price due to slippage
        adjusted_entry_price = entry_price - slippage_points
        # For short exit (buying back), we buy at a higher price due to slippage
        adjusted_exit_price = exit_price + slippage_points

    gross_profit_per_unit = (
        adjusted_exit_price - adjusted_entry_price
        if trade_type == "long"
        else adjusted_entry_price - adjusted_exit_price
    )
    
    gross_profit = gross_profit_per_unit * trade_size

    # Calculate brokerage cost separately for entry and exit transactions
    brokerage_cost_entry = adjusted_entry_price * trade_size * brokerage_pct
    brokerage_cost_exit = adjusted_exit_price * trade_size * brokerage_pct
    brokerage_cost = brokerage_cost_entry + brokerage_cost_exit

    transaction_cost = brokerage_cost

    return gross_profit - transaction_cost


def generate_trades_df(df, initial_capital, risk_per_trade_pct, brokerage_pct=0.0003, slippage_points=2):
    trades = []
    position_type = None  # None, 'long', 'short'
    entry_date = None
    entry_price = None
    stop_loss_price = None
    current_capital = initial_capital
    trade_size = 0 # Number of units/lots for the current trade
    
    # Store capital at each step for equity curve
    equity_curve_data = []
    # Initialize equity curve with initial capital at the first date
    if not df.empty:
        equity_curve_data.append({"Date": df.index[0], "Capital": initial_capital})

    # Start loop from 1 to allow checking previous candle's signals
    for i in range(1, len(df)): # Changed range start from 0 to 1
        current_date = df.index[i]
        current_open = df["Open"].iloc[i]
        current_low = df["Low"].iloc[i]
        current_high = df["High"].iloc[i]

        # Signals from the PREVIOUS candle
        prev_buy_entry = df["BUY_Entry"].iloc[i-1] # Check signal from previous candle
        prev_sell_entry = df["SELL_Entry"].iloc[i-1] # Check signal from previous candle

        # Exits are still based on current candle's conditions
        buy_exit = df["BUY_Exit"].iloc[i]
        sell_exit = df["SELL_Exit"].iloc[i]

        current_stoploss = df["StopLoss"].iloc[i] # Stop loss for the current candle, used for open positions or new entries

        # Update trailing stop loss for open positions
        if position_type == "long" and stop_loss_price is not None:
            # Ensure current_stoploss is not None before using it for max
            if current_stoploss is not None:
                stop_loss_price = max(stop_loss_price, current_stoploss)
        elif position_type == "short" and stop_loss_price is not None:
            # Ensure current_stoploss is not None before using it for min
            if current_stoploss is not None:
                stop_loss_price = min(stop_loss_price, current_stoploss)

        # Check for stoploss hit first if a position is open
        if position_type == "long" and stop_loss_price is not None and current_low <= stop_loss_price:
            exit_date = current_date
            exit_price = stop_loss_price # Exit at stoploss price
            
            # Calculate profit/loss for the exited trade
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
            position_type = None
            entry_date = None
            entry_price = None
            stop_loss_price = None
            trade_size = 0
            equity_curve_data.append({"Date": current_date, "Capital": current_capital})
            # Continue to next iteration after closing a position due to stop loss
            continue
        elif position_type == "short" and stop_loss_price is not None and current_high >= stop_loss_price:
            exit_date = current_date
            exit_price = stop_loss_price # Exit at stoploss price
            
            # Calculate profit/loss for the exited trade
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
            position_type = None
            entry_date = None
            entry_price = None
            stop_loss_price = None
            trade_size = 0
            equity_curve_data.append({"Date": current_date, "Capital": current_capital})
            # Continue to next iteration after closing a position due to stop loss
            continue

        # If no position is open, check for new entry signals from the PREVIOUS candle
        if position_type is None:
            trade_size = 1 # Fixed trade size as requested

            if prev_buy_entry: # Entry on current candle's open if signal on previous
                position_type = "long"
                entry_date = current_date
                entry_price = current_open
                stop_loss_price = current_stoploss
            elif prev_sell_entry: # Entry on current candle's open if signal on previous
                position_type = "short"
                entry_date = current_date
                entry_price = current_open
                stop_loss_price = current_stoploss
        elif position_type == "long":
            if buy_exit: # Exit on current candle's open if signal on current
                exit_date = current_date
                exit_price = current_open
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

                if prev_sell_entry: # Reversal: immediate short entry on current candle's open if signal on previous
                    trade_size = 3 # Fixed trade size for reversal entry
                    position_type = "short"
                    entry_date = current_date
                    entry_price = current_open
                    stop_loss_price = current_stoploss
                else:
                    position_type = None
                    stop_loss_price = None
                    trade_size = 0
        elif position_type == "short":
            if sell_exit: # Exit on current candle's open if signal on current
                exit_date = current_date
                exit_price = current_open
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

                if prev_buy_entry: # Reversal: immediate long entry on current candle's open if signal on previous
                    trade_size = 3 # Fixed trade size for reversal entry
                    position_type = "long"
                    entry_date = current_date
                    entry_price = current_open
                    stop_loss_price = current_stoploss
                else:
                    position_type = None
                    stop_loss_price = None
                    trade_size = 0
        
        # Append current capital at each step, even if no trade occurred
        # This needs to be adjusted if the loop starts from 1.
        # The first equity point should be initial capital at the first date.
        # Then, append for each subsequent date.
        if not equity_curve_data or equity_curve_data[-1]["Date"] != current_date:
            equity_curve_data.append({"Date": current_date, "Capital": current_capital})


    # If still in position at the end of the data, close the trade
    # This logic remains the same, as it closes on the last available open price.
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

    trades_df = pd.DataFrame(trades)
    equity_curve_df = pd.DataFrame(equity_curve_data).set_index("Date")
    return trades_df, equity_curve_df


def calculate_performance_metrics(trades_df, equity_curve_df, risk_free_rate=0.0):
    if trades_df.empty:
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

    # Calculate Max Drawdown using the equity curve
    if not equity_curve_df.empty:
        equity_curve = equity_curve_df["Capital"]
        peak = equity_curve.cummax()
        drawdown_values = (peak - equity_curve) / peak
        max_drawdown = drawdown_values.max() * 100 if not drawdown_values.empty else 0
    else:
        max_drawdown = 0

    # Profit Factor
    total_gross_profit = winning_trades["profit_loss"].sum()
    total_gross_loss = abs(losing_trades["profit_loss"].sum())
    profit_factor = (
        total_gross_profit / total_gross_loss if total_gross_loss > 0 else np.inf
    )

    # Sharpe Ratio (using daily returns from equity curve)
    if not equity_curve_df.empty and len(equity_curve_df) > 1:
        returns = equity_curve_df["Capital"].pct_change().dropna()
        if returns.std() > 0:
            sharpe_ratio = (
                (returns.mean() - risk_free_rate / 252)
                / returns.std()
                * np.sqrt(252)
            )  # Annualized
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0

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
    if metrics_df.empty:
        return 0.0

    metrics = metrics_df.iloc[0]

    # Normalize metrics
    # Sharpe Ratio: Can be negative, so shift and scale. Or just use as is if higher is better.
    # For normalization, we need a range. Let's assume a min/max or use rank-based normalization if we have many results.
    # For a single run, we'll just use the value directly, assuming higher is better.
    # When used in optimization, we'll normalize across all results.

    # For now, just return the raw metrics. Normalization will happen in run_optimization.
    # This function will be updated to take a series of metrics and return a single score.

    # Placeholder for now, actual normalization will be done in run_optimization
    # This function will be called for each row in the optimization results DataFrame.

    # To avoid division by zero or negative values for inverse drawdown, handle cases where max_drawdown is 0 or very small.
    inverse_max_drawdown = 0
    if metrics["max_drawdown"] > 0:
        inverse_max_drawdown = 1 / metrics["max_drawdown"]
    elif metrics["max_drawdown"] == 0 and metrics["total_profit"] > 0:
        inverse_max_drawdown = 1  # Assign a high value if no drawdown and profitable
    else:
        inverse_max_drawdown = (
            -1
        )  # Assign a penalty if losing and no drawdown (implies no trades or constant loss)

    # Normalize total profit by initial capital
    normalized_total_profit = metrics["total_profit"] / initial_capital
    
    # Normalize final capital by initial capital
    normalized_final_capital = metrics["final_capital"] / initial_capital

    # Define weights for preliminary composite score calculation (same as final for consistency)
    weights = {
        "sharpe_ratio": 0.20,
        "profit_factor": 0.20,
        "inverse_max_drawdown": 0.20,
        "win_rate": 0.20,
        "normalized_total_profit": 0.10,
        "normalized_final_capital": 0.10, # New weight for final capital
    }

    # Calculate a preliminary composite score using raw values (not globally normalized)
    # This score is for immediate feedback within the loop.
    # Handle potential inf/NaN for profit_factor and negative for others for a more robust preliminary score.

    # Simple normalization for preliminary score (0-1 range for each component)
    # This is a heuristic and not a true normalization across the entire dataset of results.
    # It will give a relative score for the current strategy.

    # For sharpe_ratio: can be negative. Shift to positive range for scoring.
    prelim_sharpe_norm = (
        (metrics["sharpe_ratio"] + 10) / 20 if metrics["sharpe_ratio"] > -10 else 0
    )  # Example shift
    prelim_sharpe_norm = max(0, min(1, prelim_sharpe_norm))  # Clamp to 0-1

    # For profit_factor: can be inf. Cap it.
    prelim_profit_factor_norm = (
        min(10, metrics["profit_factor"]) / 10
        if metrics["profit_factor"] != np.inf
        else 1
    )  # Cap at 10 for normalization
    prelim_profit_factor_norm = max(
        0, min(1, prelim_profit_factor_norm)
    )  # Clamp to 0-1

    # For inverse_max_drawdown: can be -1. Map -1 to 0, others scale.
    prelim_inverse_max_drawdown_norm = 0
    if inverse_max_drawdown == 1:  # No drawdown, profitable
        prelim_inverse_max_drawdown_norm = 1
    elif inverse_max_drawdown > 0:  # Some drawdown, but positive inverse
        prelim_inverse_max_drawdown_norm = min(
            1, inverse_max_drawdown / 10
        )  # Scale, cap at 1
    # If inverse_max_drawdown is -1 or 0, it remains 0 or very low.

    # For win_rate: already 0-100. Normalize to 0-1.
    prelim_win_rate_norm = metrics["win_rate"] / 100

    # For normalized_total_profit: can be negative. Shift to positive range.
    prelim_normalized_total_profit_norm = (
        (normalized_total_profit + 1) / 2 if normalized_total_profit > -1 else 0
    )  # Example shift
    prelim_normalized_total_profit_norm = max(
        0, min(1, prelim_normalized_total_profit_norm)
    )  # Clamp to 0-1

    # For normalized_final_capital: can be negative. Shift to positive range.
    prelim_normalized_final_capital_norm = (
        (normalized_final_capital + 1) / 2 if normalized_final_capital > -1 else 0
    ) # Example shift
    prelim_normalized_final_capital_norm = max(
        0, min(1, prelim_normalized_final_capital_norm)
    ) # Clamp to 0-1


    preliminary_composite_score = (
        prelim_sharpe_norm * weights["sharpe_ratio"]
        + prelim_profit_factor_norm * weights["profit_factor"]
        + prelim_inverse_max_drawdown_norm * weights["inverse_max_drawdown"]
        + prelim_win_rate_norm * weights["win_rate"]
        + prelim_normalized_total_profit_norm * weights["normalized_total_profit"]
        + prelim_normalized_final_capital_norm * weights["normalized_final_capital"] # Add to score
    )

    return {
        "sharpe_ratio": metrics["sharpe_ratio"],
        "profit_factor": metrics["profit_factor"],
        "inverse_max_drawdown": inverse_max_drawdown,
        "win_rate": metrics["win_rate"],
        "normalized_total_profit": normalized_total_profit,
        "normalized_final_capital": normalized_final_capital, # Include in return
        "preliminary_composite_score": preliminary_composite_score,
    }


def run_optimization(
    base_df,
    period_ranges,
    multiplier_ranges,
    ema_fast_period_range, # Changed to fast EMA range
    ema_slow_period_range, # Added slow EMA range
    stoploss_atr_multiplier_range,
    risk_per_trade_pct_range,
    initial_capital=100000,
    brokerage_pct=0.0003,
    slippage_points=2,
):
    all_results = []

    # Unpack ranges
    p1_start, p1_end, p1_step = period_ranges[0]
    m1_start, m1_end, m1_step = multiplier_ranges[0]
    p2_start, p2_end, p2_step = period_ranges[1]
    m2_start, m2_end, m2_step = multiplier_ranges[1]
    ema_fast_p_start, ema_fast_p_end, ema_fast_p_step = ema_fast_period_range # Unpack fast EMA range
    ema_slow_p_start, ema_slow_p_end, ema_slow_p_step = ema_slow_period_range # Unpack slow EMA range
    sl_atr_mult_start, sl_atr_mult_end, sl_atr_mult_step = stoploss_atr_multiplier_range
    risk_pct_start, risk_pct_end, risk_pct_step = risk_per_trade_pct_range

    # Generate actual ranges
    periods1 = list(range(p1_start, p1_end + 1, p1_step))
    multipliers1 = list(np.round(np.arange(m1_start, m1_end + m1_step, m1_step), 1))
    periods2 = list(range(p2_start, p2_end + 1, p2_step))
    multipliers2 = list(np.round(np.arange(m2_start, m2_end + m2_step, m2_step), 1))
    ema_fast_periods = list(range(ema_fast_p_start, ema_fast_p_end + 1, ema_fast_p_step)) # Fast EMA periods
    ema_slow_periods = list(range(ema_slow_p_start, ema_slow_p_end + 1, ema_slow_p_step)) # Slow EMA periods
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

    num_iterations = 2

    print(f"Starting optimization with {num_iterations} random combinations...")

    for _ in range(num_iterations):
        # Randomly select parameters for each SuperTrend, EMAs, ATR multiplier, and Risk per Trade
        p1 = random.choice(periods1)
        m1 = random.choice(multipliers1)
        p2 = random.choice(periods2)
        m2 = random.choice(multipliers2)
        ema_fast_p = random.choice(ema_fast_periods) # Random fast EMA period
        ema_slow_p = random.choice(ema_slow_periods) # Random slow EMA period
        sl_atr_mult = random.choice(stoploss_atr_multipliers)
        risk_pct = random.choice(risk_per_trade_pcts)

        df_copy = base_df.copy()

        ha_high_main = df_copy["HA_High"]
        ha_low_main = df_copy["HA_Low"]
        ha_close_main = df_copy["HA_Close"]

        # Calculate ATR once for the main DataFrame
        use_heikin_ashi = False  # Set to True to use Heikin Ashi

        if use_heikin_ashi:
            high = df_copy["HA_High"]
            low = df_copy["HA_Low"]
            close = df_copy["HA_Close"]

        else:
            high = df_copy["High"]
            low = df_copy["Low"]
            close = df_copy["Close"]

        df_copy["TR"] = np.maximum.reduce(
            [
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1)),
            ]
        )

        # Use a default period for this main ATR calculation, e.g., p1
        df_copy["ATR"] = df_copy["TR"].ewm(alpha=1/p1, adjust=False).mean()

        # Calculate SuperTrend columns
        st_1_name = f"ST_{p1}_{str(m1).replace('.', '_')}"
        st_1_dir_name = f"ST_Dir_{p1}_{str(m1).replace('.', '_')}"
        st_1_trending_up_name = f"trendingUp_{p1}_{str(m1).replace('.', '_')}"
        st_1_trending_down_name = f"trendingDown_{p1}_{str(m1).replace('.', '_')}"
        st_1_df_result = calculate_supertrend(
            base_df.copy(),
            period=p1,
            multiplier=m1,
            trending_up_col_name=st_1_trending_up_name,
            trending_down_col_name=st_1_trending_down_name,
        )

        st_2_name = f"ST_{p2}_{str(m2).replace('.', '_')}"
        st_2_dir_name = f"ST_Dir_{p2}_{str(m2).replace('.', '_')}"
        st_2_trending_up_name = f"trendingUp_{p2}_{str(m2).replace('.', '_')}"
        st_2_trending_down_name = f"trendingDown_{p2}_{str(m2).replace('.', '_')}"
        st_2_df_result = calculate_supertrend(
            base_df.copy(),
            period=p2,
            multiplier=m2,
            trending_up_col_name=st_2_trending_up_name,
            trending_down_col_name=st_2_trending_down_name,
        )

        # Calculate EMAs
        ema_fast_name = f"EMA_Fast_{ema_fast_p}"
        ema_slow_name = f"EMA_Slow_{ema_slow_p}"
        df_copy[ema_fast_name] = calculate_ema(
            base_df.copy(), period=ema_fast_p, price_col="Close"
        )
        df_copy[ema_slow_name] = calculate_ema(
            base_df.copy(), period=ema_slow_p, price_col="Close"
        )

        # Join the results to df_copy.
        df_copy = df_copy.join(st_1_df_result)
        if st_1_name != st_2_name:
            df_copy = df_copy.join(st_2_df_result)

        # Generate BUY_Entry, BUY_Exit, SELL_Entry, SELL_Exit signals
        df_copy["BUY_Entry"] = (
            (df_copy["HA_Close"].shift(1) < df_copy[st_1_name].shift(1))
            & (df_copy["HA_Close"].shift(1) < df_copy[st_2_name].shift(1))
            & (df_copy["HA_Close"] > df_copy[st_1_name])
            & (df_copy["HA_Close"] > df_copy[st_2_name])
            & (df_copy[st_1_dir_name] == 1)
            & (df_copy[st_2_dir_name] == 1)
            & (df_copy[ema_fast_name] > df_copy[ema_slow_name]) # EMA crossover for BUY
        )

        # Calculate StopLoss based on the lower period supertrend (assuming p1 is lower period)
        df_copy["StopLoss"] = np.where(
            df_copy[st_1_dir_name] == 1,
            df_copy[st_1_trending_up_name] - sl_atr_mult * df_copy["ATR"],
            df_copy[st_1_trending_down_name] + sl_atr_mult * df_copy["ATR"],
        )

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

        df_copy["SELL_Entry"] = (
            (df_copy["HA_Close"].shift(1) > df_copy[st_1_name].shift(1))
            & (df_copy["HA_Close"].shift(1) > df_copy[st_2_name].shift(1))
            & (df_copy["HA_Close"] < df_copy[st_1_name])
            & (df_copy["HA_Close"] < df_copy[st_2_name])
            & (df_copy[st_1_dir_name] == -1)
            & (df_copy[st_2_dir_name] == -1)
            & (df_copy[ema_fast_name] < df_copy[ema_slow_name]) # EMA crossover for SELL
        )

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

        # Round all numeric columns to 2 decimal places
        df_copy = df_copy.round(2)

        # Remove rows with NaN values (especially at the beginning due to ATR and EMA calculation)
        df_copy.dropna(inplace=True)

        # Save the DataFrame to a temporary CSV file for debugging
        # This is useful for debugging and checking the generated signals
        df_copy.to_csv("debug_signals.csv", index=True)
        

        if not df_copy.empty:
            trades_df, equity_curve_df = generate_trades_df(
                df_copy, initial_capital=initial_capital, risk_per_trade_pct=risk_pct, brokerage_pct=brokerage_pct, slippage_points=slippage_points
            )
            performance_metrics = calculate_performance_metrics(trades_df, equity_curve_df)
            composite_metrics = calculate_composite_score(
                performance_metrics, initial_capital
            )

            # Append results, including all performance metrics and composite score components
            result_row = {
                "period1": p1,
                "multiplier1": m1,
                "period2": p2,
                "multiplier2": m2,
                "ema_fast_period": ema_fast_p, # Changed to fast EMA period
                "ema_slow_period": ema_slow_p, # Added slow EMA period
                "stoploss_atr_multiplier": sl_atr_mult,
                "risk_per_trade_pct": risk_pct,
                **performance_metrics.iloc[0].to_dict(),
                **{
                    k: v
                    for k, v in composite_metrics.items()
                    if k != "preliminary_composite_score"
                },
            }
            # Add the preliminary composite score to the result row
            result_row["preliminary_composite_score"] = composite_metrics[
                "preliminary_composite_score"
            ]
            all_results.append(result_row)

            # Print performance metrics if profit_factor > 1 and win_rate > 50%
            # Print performance metrics if profit_factor > 2 and win_rate > 50% OR if preliminary_composite_score is high
            if (
                performance_metrics["profit_factor"].iloc[0] > 2
                and performance_metrics["win_rate"].iloc[0] >= 70
            ) or (composite_metrics["preliminary_composite_score"] > 0.7): # Log if composite_score is borderline good
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
                # Indicate if it's a borderline good strategy
                if not (performance_metrics["profit_factor"].iloc[0] > 2 and performance_metrics["win_rate"].iloc[0] > 50):
                    print("    (Borderline-good strategy based on composite score)")

    results_df = pd.DataFrame(all_results)

    # Normalize metrics for composite score calculation across all runs
    if not results_df.empty:
        # Handle potential infinite values in profit_factor before normalization
        results_df["profit_factor"] = results_df["profit_factor"].replace(
            [np.inf, -np.inf], np.nan
        )

        # Fill NaN values for normalization. Using 0 or a small number for profit_factor, and 0 for others.
        # For inverse_max_drawdown, if it's -1 (penalty), it should remain low.
        results_df["sharpe_ratio_norm"] = (
            results_df["sharpe_ratio"] - results_df["sharpe_ratio"].min()
        ) / (results_df["sharpe_ratio"].max() - results_df["sharpe_ratio"].min())
        results_df["profit_factor_norm"] = (
            results_df["profit_factor"] - results_df["profit_factor"].min()
        ) / (results_df["profit_factor"].max() - results_df["profit_factor"].min())

        # For inverse_max_drawdown, ensure positive values for normalization.
        # If max_drawdown was 0 and profit > 0, inverse_max_drawdown was 1. If max_drawdown was 0 and profit <= 0, inverse_max_drawdown was -1.
        # We need to handle the -1 case. Let's map -1 to 0 for normalization, and scale others.
        min_inv_dd = results_df["inverse_max_drawdown"].min()
        max_inv_dd = results_df["inverse_max_drawdown"].max()

        # If all inverse_max_drawdown are -1 or 0, avoid division by zero in normalization
        if max_inv_dd == min_inv_dd:
            results_df["inverse_max_drawdown_norm"] = 0.0
        else:
            results_df["inverse_max_drawdown_norm"] = (
                results_df["inverse_max_drawdown"] - min_inv_dd
            ) / (max_inv_dd - min_inv_dd)

        # Ensure win_rate is already 0-100, normalize to 0-1
        results_df["win_rate_norm"] = results_df["win_rate"] / 100

        # Normalized total profit is already 0-1 (or can be negative if loss)
        # For normalization, shift negative values to positive range if present
        min_norm_profit = results_df["normalized_total_profit"].min()
        if min_norm_profit < 0:
            results_df["normalized_total_profit_norm"] = (
                results_df["normalized_total_profit"] - min_norm_profit
            ) / (results_df["normalized_total_profit"].max() - min_norm_profit)
        else:
            results_df["normalized_total_profit_norm"] = (
                results_df["normalized_total_profit"]
                / results_df["normalized_total_profit"].max()
            )

        # Normalized final capital
        min_norm_final_capital = results_df["normalized_final_capital"].min()
        if min_norm_final_capital < 0:
            results_df["normalized_final_capital_norm"] = (
                results_df["normalized_final_capital"] - min_norm_final_capital
            ) / (results_df["normalized_final_capital"].max() - min_norm_final_capital)
        else:
            results_df["normalized_final_capital_norm"] = (
                results_df["normalized_final_capital"]
                / results_df["normalized_final_capital"].max()
            )

        # Fill any remaining NaNs after normalization with 0 (e.g., if a metric was constant)
        results_df = results_df.fillna(0)

        # Calculate Composite Score
        weights = {
                "sharpe_ratio": 0.20,
                "profit_factor": 0.20,
                "inverse_max_drawdown": 0.20,
                "win_rate": 0.20,
                "normalized_total_profit": 0.10,
                "normalized_final_capital": 0.10, # New weight for final capital
            }

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


file_path = (
    r"D:\Sushant\Fyers_AlgoTrade\Fyers_Data\Nifty50_Index\NIFTY50_INDEX_D_Min.csv"
)

try:
    df = pd.read_csv(file_path)
    df = df.drop(columns=["Unnamed: 0", "Volume"], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    initial_rows_before_dedup = len(df)
    df.drop_duplicates(subset=["Date"], keep="first", inplace=True)
    rows_removed_dedup = initial_rows_before_dedup - len(df)
    if rows_removed_dedup > 0:
        print(f"Removed {rows_removed_dedup} duplicate date rows.")

    df = df.set_index("Date")

    # Filter data after 01-01-2021
    df = df[df.index >= "2021-01-01"]
    # df = df[df.index < "2022-01-01"]

    original_index = df.index
    df = df.reset_index(drop=True)

    # Calculate Heikin Ashi candles
    ha_df = calculate_heikin_ashi(
        df.copy()
    )  # Pass a copy to avoid modifying original df
    df = df.join(ha_df)  # Join HA columns back to the original DataFrame

    df.index = original_index  # Restore original index after HA calculation

    # Define optimization ranges
    period_ranges = [(8, 8, 1), (8, 8, 1)]
    multiplier_ranges = [(1.0, 1.0, 1), (1.0, 1.0, 1)]
    ema_fast_period_range = (20, 20, 1) # Fast EMA range (5-21)
    ema_slow_period_range = (50, 50, 1) # Slow EMA range (20-50)
    stoploss_atr_multiplier_range = (2.0, 2.0, 1)
    risk_per_trade_pct_range = (0.02, 0.02, 0.01)

    # Run the optimization
    optimization_results = run_optimization(
        df.copy(),
        period_ranges,
        multiplier_ranges,
        ema_fast_period_range, # Pass fast EMA range
        ema_slow_period_range, # Pass slow EMA range
        stoploss_atr_multiplier_range,
        risk_per_trade_pct_range,
        initial_capital=100000,
        brokerage_pct=0.0003,
        slippage_points=2,
    )

    print("\nOptimization Results:")
    if not optimization_results.empty:
        # Sort by composite score
        print(
            optimization_results.sort_values(
                by="composite_score", ascending=False
            ).head(20)
        )
        optimization_results.to_csv("supertrend_optimization_results.csv", index=False)
        print("\nOptimization results saved to 'supertrend_optimization_results.csv'")

        # Save trades and performance metrics for the best strategy based on composite score
        best_params = optimization_results.loc[
            optimization_results["composite_score"].idxmax()
        ]

        # Extract best EMA periods
        best_ema_fast_p = int(best_params["ema_fast_period"])
        best_ema_slow_p = int(best_params["ema_slow_period"])

        best_df_copy = df.copy()

        high_main_best = best_df_copy["HA_High"]
        low_main_best = best_df_copy["HA_Low"]
        close_main_best = best_df_copy["HA_Close"]

        # Calculate ATR once for the main DataFrame for best strategy run
        use_heikin_ashi = False  # Use Heikin Ashi for best strategy run

        if use_heikin_ashi:
            high = best_df_copy["HA_High"]
            low  = best_df_copy["HA_Low"]
            close = best_df_copy["HA_Close"]

        else:
            high = best_df_copy["High"]
            low = best_df_copy["Low"]
            close = best_df_copy["Close"]

        best_df_copy["TR"] = np.maximum.reduce(
            [
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1)),
            ]
        )

        period = int(best_params['period1'])
        best_df_copy["ATR"] = best_df_copy["TR"].ewm(alpha=1/period, adjust=False).mean()

        st_1_best_name = f"ST_{int(best_params['period1'])}_{str(best_params['multiplier1']).replace('.', '_')}"
        st_1_best_dir_name = f"ST_Dir_{int(best_params['period1'])}_{str(best_params['multiplier1']).replace('.', '_')}"
        st_1_best_trending_up_name = f"trendingUp_{int(best_params['period1'])}_{str(best_params['multiplier1']).replace('.', '_')}"
        st_1_best_trending_down_name = f"trendingDown_{int(best_params['period1'])}_{str(best_params['multiplier1']).replace('.', '_')}"
        st_1_best_df = calculate_supertrend(
            df.copy(),
            period=int(best_params["period1"]),
            multiplier=best_params["multiplier1"],
            trending_up_col_name=st_1_best_trending_up_name,
            trending_down_col_name=st_1_best_trending_down_name,
        )
        best_df_copy = best_df_copy.join(st_1_best_df)

        st_2_best_name = f"ST_{int(best_params['period2'])}_{str(best_params['multiplier2']).replace('.', '_')}"
        st_2_best_dir_name = f"ST_Dir_{int(best_params['period2'])}_{str(best_params['multiplier2']).replace('.', '_')}"
        st_2_best_trending_up_name = f"trendingUp_{int(best_params['period2'])}_{str(best_params['multiplier2']).replace('.', '_')}"
        st_2_best_trending_down_name = f"trendingDown_{int(best_params['period2'])}_{str(best_params['multiplier2']).replace('.', '_')}"
        st_2_best_df = calculate_supertrend(
            df.copy(),
            period=int(best_params["period2"]),
            multiplier=best_params["multiplier2"],
            trending_up_col_name=st_2_best_trending_up_name,
            trending_down_col_name=st_2_best_trending_down_name,
        )
        if st_1_best_name != st_2_best_name:
            best_df_copy = best_df_copy.join(st_2_best_df)

        # Calculate EMAs for the best parameters
        ema_fast_best_name = f"EMA_Fast_{best_ema_fast_p}"
        ema_slow_best_name = f"EMA_Slow_{best_ema_slow_p}"
        best_df_copy[ema_fast_best_name] = calculate_ema(
            best_df_copy, period=best_ema_fast_p, price_col="Close"
        )
        best_df_copy[ema_slow_best_name] = calculate_ema(
            best_df_copy, period=best_ema_slow_p, price_col="Close"
        )

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
            & (best_df_copy[ema_fast_best_name] > best_df_copy[ema_slow_best_name]) # EMA crossover for BUY
        )

        # Calculate StopLoss based on the lower period supertrend (assuming p1 is lower period)
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
            & (best_df_copy[ema_fast_best_name] < best_df_copy[ema_slow_best_name]) # EMA crossover for SELL
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

        best_trades_df, best_equity_curve_df = generate_trades_df(
            best_df_copy,
            initial_capital=initial_capital,
            risk_per_trade_pct=best_params["risk_per_trade_pct"],
            brokerage_pct=0.0003,
            slippage_points=2,
        )
        best_performance_metrics = calculate_performance_metrics(best_trades_df, best_equity_curve_df)

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
