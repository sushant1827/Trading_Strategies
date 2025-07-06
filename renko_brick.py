import pandas as pd
import numpy as np
import os

def calculate_atr(df, period=14):
    """
    Calculates the Average True Range (ATR) for a given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' columns.
        period (int): The period for ATR calculation (default is 14).

    Returns:
        pd.Series: A Series containing the ATR values.
    """
    # Calculate True Range (TR)
    # TR = max[(High - Low), abs(High - Prev. Close), abs(Low - Prev. Close)]
    df['TR1'] = abs(df['High'] - df['Low'])
    df['TR2'] = abs(df['High'] - df['Close'].shift(1))
    df['TR3'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)

    # Calculate ATR using Exponential Moving Average (EMA) for smoothing
    # Wilder's smoothing method: ATR = (Previous ATR * (period - 1) + Current TR) / period
    # pd.Series.ewm(adjust=False) simulates Wilder's smoothing.
    atr_series = df['TR'].ewm(span=period, adjust=False, min_periods=period).mean()
    
    # Clean up intermediate columns
    df.drop(columns=['TR1', 'TR2', 'TR3', 'TR'], inplace=True, errors='ignore')
    
    return atr_series

def backtest_renko_strategy(df, atr_period=14):
    """
    Backtests a Renko-based trading strategy on the given OHLC data.

    Args:
        df (pd.DataFrame): DataFrame with 'Date', 'Open', 'High', 'Low', 'Close' columns.
                           'Date' should be datetime objects.
        atr_period (int): Period for ATR calculation (default 14).

    Returns:
        tuple: A tuple containing:
               - pd.DataFrame: DataFrame of executed trades.
               - float: Total PNL of the strategy.
               - float: The calculated Renko brick size.
    """
    trades = []
    in_trade = False
    position_type = None  # 'Long' or 'Short'
    entry_price = 0
    stop_loss = 0
    trade_entry_date = None
    
    total_pnl = 0

    # --- Step 1: Calculate Average ATR for Fixed Brick Size ---
    # This creates a copy and calculates ATR for the whole dataset to get an average brick size.
    df_atr_calc = df.copy()
    df_atr_calc['ATR_Series'] = calculate_atr(df_atr_calc, period=atr_period)
    
    # Determine the fixed Renko brick size (ATR 14 period * 0.5)
    # Use the mean ATR for a robust, single brick size for the entire backtest.
    brick_size_val = df_atr_calc['ATR_Series'].mean() * 0.5
    
    # Handle cases where ATR calculation might result in NaN or zero (e.g., very flat initial data)
    if pd.isna(brick_size_val) or brick_size_val <= 0:
        # Fallback 1: Use average range if ATR fails
        brick_size_val = (df['High'] - df['Low']).replace([np.inf, -np.inf], np.nan).mean() * 0.5
        if pd.isna(brick_size_val) or brick_size_val <= 0:
            # Fallback 2: Use a small percentage of the average close price as a last resort
            brick_size_val = df['Close'].mean() * 0.001
        print(f"Warning: Calculated ATR resulted in issues or zero. Using fallback brick size: {brick_size_val:.2f}")

    # Ensure brick_size_val is not excessively small, set a minimum to prevent issues
    min_brick_size = 0.01 
    if brick_size_val < min_brick_size:
        brick_size_val = min_brick_size
        print(f"Warning: Calculated brick size was too small (<{min_brick_size}). Set to minimum: {min_brick_size}")

    print(f"Calculated Renko Brick Size: {brick_size_val:.2f}")

    # --- Step 2: Initialize Renko state and history ---
    # `renko_close`: The current effective closing price of the latest Renko brick.
    # `renko_current_direction`: Tracks the direction of the last formed Renko brick ('Up', 'Down', 'Neutral').
    # `renko_bricks_history`: Stores details of all formed Renko bricks.
    
    # Initialize with the close of the first OHLC bar
    # This forms a 'Neutral' brick to start.
    renko_close = df['Close'].iloc[0]
    renko_current_direction = 'Neutral'
    renko_bricks_history = [{
        'Date': df['Date'].iloc[0], 
        'Open': renko_close, 
        'Close': renko_close, 
        'Direction': 'Neutral'
    }]
    
    # --- Step 3: Iterate through OHLC bars to simulate Renko formation and apply strategy ---
    for i in range(1, len(df)):
        current_ohlc_bar = df.iloc[i]
        ohlc_date = current_ohlc_bar['Date']
        
        # List to store any new Renko bricks formed by the current OHLC bar
        new_bricks_this_bar = []

        # --- Renko Brick Generation Logic (on-the-fly) ---
        # Prioritize upward movement for brick formation
        while current_ohlc_bar['High'] >= renko_close + brick_size_val:
            # Check for reversal from a 'Down' (red) direction
            # A reversal brick typically requires a 2x brick size movement from the previous close
            if renko_current_direction == 'Down':
                if current_ohlc_bar['High'] < renko_close + (2 * brick_size_val):
                    # Not enough movement for a reversal brick, break and wait for more price action
                    break 
                
            # Form a new green Renko brick
            new_bricks_this_bar.append({
                'Date': ohlc_date,
                'Open': renko_close,
                'Close': renko_close + brick_size_val,
                'Direction': 'Up'
            })
            renko_close += brick_size_val # Update current Renko close
            renko_current_direction = 'Up' # Update Renko direction
            
        # If no new green bricks were formed, check for downward movement
        # (This implies that if a bar can form both up and down bricks, it prioritizes up.
        # A more complex Renko might consider mid-range or tick data for multiple bricks in one bar.)
        if not new_bricks_this_bar and current_ohlc_bar['Low'] <= renko_close - brick_size_val:
            while current_ohlc_bar['Low'] <= renko_close - brick_size_val:
                # Check for reversal from an 'Up' (green) direction
                if renko_current_direction == 'Up':
                    if current_ohlc_bar['Low'] > renko_close - (2 * brick_size_val):
                        # Not enough movement for a reversal brick
                        break 

                # Form a new red Renko brick
                new_bricks_this_bar.append({
                    'Date': ohlc_date,
                    'Open': renko_close,
                    'Close': renko_close - brick_size_val,
                    'Direction': 'Down'
                })
                renko_close -= brick_size_val # Update current Renko close
                renko_current_direction = 'Down' # Update Renko direction
        
        # Add any newly formed bricks (from this OHLC bar) to the historical list
        if new_bricks_this_bar:
            renko_bricks_history.extend(new_bricks_this_bar)
        
        # --- Strategy Logic ---
        
        # 1. Stop Loss Check (highest priority if in a trade)
        if in_trade:
            if position_type == 'Long':
                # If current OHLC bar's Low goes below the stop loss
                if current_ohlc_bar['Low'] <= stop_loss:
                    exit_price = stop_loss  # Exit exactly at the stop loss price
                    pnl = exit_price - entry_price
                    trades.append({
                        'Entry_Date': trade_entry_date,
                        'Entry_Price': entry_price,
                        'Position': position_type,
                        'Exit_Date': ohlc_date,
                        'Exit_Price': exit_price,
                        'PNL': pnl,
                        'Reason': 'Stop Loss Hit'
                    })
                    total_pnl += pnl
                    in_trade = False
                    position_type = None
                    # Continue to the next OHLC bar as the trade is now closed
                    continue 
            elif position_type == 'Short':
                # If current OHLC bar's High goes above the stop loss
                if current_ohlc_bar['High'] >= stop_loss:
                    exit_price = stop_loss  # Exit exactly at the stop loss price
                    pnl = entry_price - exit_price
                    trades.append({
                        'Entry_Date': trade_entry_date,
                        'Entry_Price': entry_price,
                        'Position': position_type,
                        'Exit_Date': ohlc_date,
                        'Exit_Price': exit_price,
                        'PNL': pnl,
                        'Reason': 'Stop Loss Hit'
                    })
                    total_pnl += pnl
                    in_trade = False
                    position_type = None
                    # Continue to the next OHLC bar as the trade is now closed
                    continue

        # 2. Reversal Exit Signal (if in trade and SL not hit)
        # This check happens if new bricks were formed by the current OHLC bar AND they indicate a reversal.
        if in_trade and new_bricks_this_bar:
            last_formed_brick_this_bar = new_bricks_this_bar[-1] # Get the very last brick formed by this OHLC bar
            if position_type == 'Long' and last_formed_brick_this_bar['Direction'] == 'Down':
                exit_price = last_formed_brick_this_bar['Close'] # Exit at the close of the red Renko brick
                pnl = exit_price - entry_price
                trades.append({
                    'Entry_Date': trade_entry_date,
                    'Entry_Price': entry_price,
                    'Position': position_type,
                    'Exit_Date': ohlc_date, # Date of the OHLC bar that formed the exit brick
                    'Exit_Price': exit_price,
                    'PNL': pnl,
                    'Reason': 'Renko Reversal (Red Brick)'
                })
                total_pnl += pnl
                in_trade = False
                position_type = None
            elif position_type == 'Short' and last_formed_brick_this_bar['Direction'] == 'Up':
                exit_price = last_formed_brick_this_bar['Close'] # Exit at the close of the green Renko brick
                pnl = entry_price - exit_price
                trades.append({
                    'Entry_Date': trade_entry_date,
                    'Entry_Price': entry_price,
                    'Position': position_type,
                    'Exit_Date': ohlc_date,
                    'Exit_Price': exit_price,
                    'PNL': pnl,
                    'Reason': 'Renko Reversal (Green Brick)'
                })
                total_pnl += pnl
                in_trade = False
                position_type = None

        # 3. Entry Signal (only if not already in a trade)
        # Entry signals are checked only if new Renko bricks were formed by the current OHLC bar
        # and there are enough historical bricks to define a 'previous brick'.
        if not in_trade and new_bricks_this_bar:
            # Need at least two bricks in history to define "previous brick" for stop-loss
            if len(renko_bricks_history) >= 2:
                last_brick_for_signal = renko_bricks_history[-1]
                second_last_brick_for_signal = renko_bricks_history[-2]

                # BUY Entry: "1 Green Renko brick closed / formed"
                if last_brick_for_signal['Direction'] == 'Up':
                    in_trade = True
                    position_type = 'Long'
                    entry_price = last_brick_for_signal['Close']
                    trade_entry_date = ohlc_date
                    
                    # Stop loss at bottom of the previous Renko brick
                    stop_loss = min(second_last_brick_for_signal['Open'], second_last_brick_for_signal['Close'])
                    
                    # print(f"Long Entry at {ohlc_date}: Price={entry_price:.2f}, SL={stop_loss:.2f}")

                # SELL Entry: "1 red Renko brick closed / formed"
                elif last_brick_for_signal['Direction'] == 'Down':
                    in_trade = True
                    position_type = 'Short'
                    entry_price = last_brick_for_signal['Close']
                    trade_entry_date = ohlc_date
                    
                    # Stop loss at top of the previous Renko brick
                    stop_loss = max(second_last_brick_for_signal['Open'], second_last_brick_for_signal['Close'])
                    
                    # print(f"Short Entry at {ohlc_date}: Price={entry_price:.2f}, SL={stop_loss:.2f}")
    
    # --- Step 4: Handle any open trades at the end of the data ---
    if in_trade:
        # Exit at the closing price of the very last OHLC bar
        exit_price = df.iloc[-1]['Close']
        if position_type == 'Long':
            pnl = exit_price - entry_price
        else: # Short
            pnl = entry_price - exit_price
        trades.append({
            'Entry_Date': trade_entry_date,
            'Entry_Price': entry_price,
            'Position': position_type,
            'Exit_Date': df.iloc[-1]['Date'],
            'Exit_Price': exit_price,
            'PNL': pnl,
            'Reason': 'End of Data'
        })
        total_pnl += pnl

    return pd.DataFrame(trades), total_pnl, brick_size_val

# --- Main execution block ---
if __name__ == "__main__":
    file_name = 'NIFTY50_INDEX_60_Min.csv'
    
    # Create a dummy CSV file for demonstration if it doesn't exist
    if not os.path.exists(file_name):
        print(f"Creating a dummy CSV file: {file_name}")
        dummy_data = """
,Date,Open,High,Low,Close,Volume
0,2017-07-17 08:15:00,9908.15,9908.15,9908.15,9908.15,0
1,2017-07-17 09:15:00,9908.15,9916.7,9894.85,9915.3,0
2,2017-07-17 10:15:00,9915.45,9920.3,9911.8,9915.25,0
3,2017-07-17 11:15:00,9915.4,9916.55,9902.75,9904.7,0
4,2017-07-17 12:15:00,9904.75,9911.1,9902.6,9907.05,0
5,2017-07-17 13:15:00,9907.25,9917.3,9904.3,9917.3,0
6,2017-07-17 14:15:00,9917.4,9928.2,9916.0,9919.9,0
7,2017-07-17 15:15:00,9919.05,9921.3,9910.4,9913.65,0
8,2017-07-18 08:15:00,9913.65,9913.65,9913.65,9913.65,0
9,2017-07-18 09:15:00,9913.65,9930.0,9900.0,9925.0,0
10,2017-07-18 10:15:00,9925.0,9928.0,9910.0,9915.0,0
11,2017-07-18 11:15:00,9915.0,9918.0,9905.0,9908.0,0
12,2017-07-18 12:15:00,9908.0,9912.0,9890.0,9895.0,0
13,2017-07-18 13:15:00,9895.0,9900.0,9880.0,9885.0,0
14,2017-07-18 14:15:00,9885.0,9890.0,9870.0,9875.0,0
15,2017-07-18 15:15:00,9875.0,9880.0,9860.0,9865.0,0
16,2017-07-19 09:15:00,9865.0,9870.0,9850.0,9855.0,0
17,2017-07-19 10:15:00,9855.0,9860.0,9840.0,9845.0,0
18,2017-07-19 11:15:00,9845.0,9850.0,9830.0,9835.0,0
19,2017-07-19 12:15:00,9835.0,9840.0,9820.0,9825.0,0
20,2017-07-19 13:15:00,9825.0,9830.0,9810.0,9815.0,0
21,2017-07-19 14:15:00,9815.0,9820.0,9800.0,9805.0,0
22,2017-07-19 15:15:00,9805.0,9810.0,9790.0,9795.0,0
23,2017-07-20 09:15:00,9795.0,9800.0,9780.0,9785.0,0
24,2017-07-20 10:15:00,9785.0,9790.0,9770.0,9775.0,0
25,2017-07-20 11:15:00,9775.0,9780.0,9760.0,9765.0,0
26,2017-07-20 12:15:00,9765.0,9770.0,9750.0,9755.0,0
27,2017-07-20 13:15:00,9755.0,9760.0,9740.0,9745.0,0
28,2017-07-20 14:15:00,9745.0,9750.0,9730.0,9735.0,0
29,2017-07-20 15:15:00,9735.0,9740.0,9720.0,9725.0,0
30,2017-07-21 09:15:00,9725.0,9730.0,9710.0,9715.0,0
31,2017-07-21 10:15:00,9715.0,9720.0,9700.0,9705.0,0
32,2017-07-21 11:15:00,9705.0,9710.0,9690.0,9695.0,0
33,2017-07-21 12:15:00,9695.0,9700.0,9680.0,9685.0,0
34,2017-07-21 13:15:00,9685.0,9690.0,9670.0,9675.0,0
35,2017-07-21 14:15:00,9675.0,9680.0,9660.0,9665.0,0
36,2017-07-21 15:15:00,9665.0,9670.0,9650.0,9655.0,0
37,2017-07-24 09:15:00,9655.0,9660.0,9640.0,9645.0,0
38,2017-07-24 10:15:00,9645.0,9650.0,9630.0,9635.0,0
39,2017-07-24 11:15:00,9635.0,9640.0,9620.0,9625.0,0
40,2017-07-24 12:15:00,9625.0,9630.0,9610.0,9615.0,0
41,2017-07-24 13:15:00,9615.0,9620.0,9600.0,9605.0,0
42,2017-07-24 14:15:00,9605.0,9610.0,9590.0,9595.0,0
43,2017-07-24 15:15:00,9595.0,9600.0,9580.0,9585.0,0
44,2017-07-25 09:15:00,9585.0,9590.0,9570.0,9575.0,0
45,2017-07-25 10:15:00,9575.0,9580.0,9560.0,9565.0,0
46,2017-07-25 11:15:00,9565.0,9570.0,9550.0,9555.0,0
47,2017-07-25 12:15:00,9555.0,9560.0,9540.0,9545.0,0
48,2017-07-25 13:15:00,9545.0,9550.0,9530.0,9535.0,0
49,2017-07-25 14:15:00,9535.0,9540.0,9520.0,9525.0,0
50,2017-07-25 15:15:00,9525.0,9530.0,9510.0,9515.0,0
51,2017-07-26 09:15:00,9515.0,9520.0,9500.0,9505.0,0
52,2017-07-26 10:15:00,9505.0,9510.0,9490.0,9495.0,0
53,2017-07-26 11:15:00,9495.0,9500.0,9480.0,9485.0,0
54,2017-07-26 12:15:00,9485.0,9490.0,9470.0,9475.0,0
55,2017-07-26 13:15:00,9475.0,9480.0,9460.0,9465.0,0
56,2017-07-26 14:15:00,9465.0,9470.0,9450.0,9455.0,0
57,2017-07-26 15:15:00,9455.0,9460.0,9440.0,9445.0,0
58,2017-07-27 09:15:00,9445.0,9450.0,9430.0,9435.0,0
59,2017-07-27 10:15:00,9435.0,9440.0,9420.0,9425.0,0
60,2017-07-27 11:15:00,9425.0,9430.0,9410.0,9415.0,0
61,2017-07-27 12:15:00,9415.0,9420.0,9400.0,9405.0,0
62,2017-07-27 13:15:00,9405.0,9410.0,9390.0,9395.0,0
63,2017-07-27 14:15:00,9395.0,9400.0,9380.0,9385.0,0
64,2017-07-27 15:15:00,9385.0,9390.0,9370.0,9375.0,0
        """
        with open(file_name, 'w') as f:
            f.write(dummy_data.strip())
    
    try:
        # Load the NIFTY index OHLC data
        # Ensure 'Date' column is parsed as datetime objects for proper handling
        df = pd.read_csv(file_name, parse_dates=['Date'])
        
        # Drop the first unnamed column if it exists (from CSV index)
        if df.columns[0] == 'Unnamed: 0':
            df = df.drop(columns=['Unnamed: 0'])
            
        print(f"Data loaded successfully from {file_name}. Shape: {df.shape}")
        # print(df.head())

        # Run the backtest
        trades_df, total_pnl, brick_size = backtest_renko_strategy(df, atr_period=14)

        print("\n--- Backtest Results ---")
        print(f"Renko Brick Size Used: {brick_size:.2f}")
        
        if not trades_df.empty:
            print("\nDetailed Trades:")
            print(trades_df.to_string()) # Use to_string() to display all rows/columns
            print(f"\nTotal Number of Trades: {len(trades_df)}")
            print(f"Total PNL: {total_pnl:.2f}")
            
            # Calculate some basic metrics
            winning_trades = trades_df[trades_df['PNL'] > 0]
            losing_trades = trades_df[trades_df['PNL'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
            
            avg_win = winning_trades['PNL'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['PNL'].mean() if not losing_trades.empty else 0
            
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Average Winning Trade: {avg_win:.2f}")
            print(f"Average Losing Trade: {avg_loss:.2f}")
            
        else:
            print("No trades were executed based on the strategy rules.")
            print(f"Final PNL: {total_pnl:.2f}")

    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        print("Please ensure the CSV file is in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred: {e}")

