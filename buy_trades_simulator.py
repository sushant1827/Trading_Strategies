import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def simulate_buy_trades(df):
    """
    Simulate buy trades based on entry/exit signals in a pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns 'datetime', 'open', 'buy_entry', 'buy_exit'
                          'datetime' should be in datetime format

    Returns:
        pd.DataFrame: DataFrame containing all completed trades with entry/exit details
    """
    # Initialize variables
    trades_list = []
    in_position = False
    entry_price = None
    entry_datetime = None

    # Ensure datetime column is properly formatted
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Detect timeframe automatically from data
    time_diffs = df['datetime'].diff().dropna()
    median_diff = time_diffs.median()

    # Extract minutes from median time difference
    timeframe_minutes = int(median_diff.total_seconds() / 60)

    print(f"Detected timeframe: {timeframe_minutes} minutes")

    # Iterate through each row
    for i, row in df.iterrows():
        current_time = row['datetime']
        current_buy_entry = row['buy_entry']

        # Check for entry signal
        if not in_position and current_buy_entry == 1:
            # Find next candle for the detected timeframe
            next_candle_time = current_time + timedelta(minutes=timeframe_minutes)

            # Look for the next row that matches the timeframe interval
            next_row_found = False
            # Allow some tolerance for the timeframe detection (±5 minutes)
            min_tolerance = max(5, timeframe_minutes - 5)
            max_tolerance = timeframe_minutes + 5

            for j in range(i + 1, len(df)):
                next_time = df.loc[j, 'datetime']
                time_diff = next_time - current_time

                # Check if this is approximately the expected timeframe later
                if timedelta(minutes=min_tolerance) <= time_diff <= timedelta(minutes=max_tolerance):
                    next_row_found = True
                    entry_price = df.loc[j, 'open']
                    entry_datetime = df.loc[j, 'datetime']
                    in_position = True
                    print(f"Entry signal at {current_time}: Position opened at {entry_datetime} with price {entry_price}")
                    break

            if not next_row_found:
                print(f"No {timeframe_minutes}-minute candle found after entry signal at {current_time}. Skipping entry.")
                continue

        # Check for exit signal while in position
        elif in_position and current_buy_entry == 0:
            # Look ahead to see if next row has buy_exit == 1
            exit_found = False
            for j in range(i + 1, len(df)):
                next_buy_exit = df.loc[j, 'buy_exit']
                if next_buy_exit == 1:
                    exit_found = True
                    exit_price = df.loc[j, 'open']
                    exit_datetime = df.loc[j, 'datetime']

                    # Calculate profit and hold time
                    profit = exit_price - entry_price
                    hold_time_seconds = (exit_datetime - entry_datetime).total_seconds()
                    hold_time_days = hold_time_seconds / 86400  # Convert to days

                    # Create trade record
                    trade_record = {
                        'Entry datetime': entry_datetime,
                        'Entry price': entry_price,
                        'Exit datetime': exit_datetime,
                        'Exit price': exit_price,
                        'Profit': profit,
                        'Hold time (days)': hold_time_days,
                        'Hold time (hours)': hold_time_seconds / 3600
                    }

                    trades_list.append(trade_record)

                    # Reset position
                    in_position = False
                    entry_price = None
                    entry_datetime = None

                    print(f"Exit signal at {current_time}: Position closed at {exit_datetime} with price {exit_price}, Profit: {profit:.2f}")
                    break

    # Handle open position at end of DataFrame
    if in_position:
        # Force exit at last row's open price
        last_row = df.iloc[-1]
        exit_price = last_row['open']
        exit_datetime = last_row['datetime']

        profit = exit_price - entry_price
        hold_time_seconds = (exit_datetime - entry_datetime).total_seconds()
        hold_time_days = hold_time_seconds / 86400

        trade_record = {
            'Entry datetime': entry_datetime,
            'Entry price': entry_price,
            'Exit datetime': exit_datetime,
            'Exit price': exit_price,
            'Profit': profit,
            'Hold time (days)': hold_time_days,
            'Hold time (hours)': hold_time_seconds / 3600
        }

        trades_list.append(trade_record)

        print(f"Open position at end: Forced exit at {exit_datetime} with price {exit_price}, Profit: {profit:.2f}")

    # Convert to DataFrame and export to CSV
    if trades_list:
        trades_df = pd.DataFrame(trades_list)

        # Remove any duplicate trades
        initial_trade_count = len(trades_df)
        trades_df = trades_df.drop_duplicates()
        if len(trades_df) < initial_trade_count:
            print(f"Removed {initial_trade_count - len(trades_df)} duplicate trades from results")

        # Reorder columns for better readability
        column_order = ['Entry datetime', 'Entry price', 'Exit datetime', 'Exit price',
                       'Profit', 'Hold time (days)', 'Hold time (hours)']
        trades_df = trades_df[column_order]

        # Export to CSV
        output_file = 'buy_trades_results.csv'
        trades_df.to_csv(output_file, index=False)

        print(f"\n{'='*50}")
        print(f"Trading simulation completed!")
        print(f"Total trades: {len(trades_df)}")
        print(f"Results exported to: {output_file}")
        print(f"{'='*50}")

        # Display summary statistics
        print(f"\nSummary Statistics:")
        print(f"Total Profit: {trades_df['Profit'].sum():.2f}")
        print(f"Average Profit per Trade: {trades_df['Profit'].mean():.2f}")
        print(f"Win Rate: {(trades_df['Profit'] > 0).mean()*100:.1f}%")
        print(f"Max Profit: {trades_df['Profit'].max():.2f}")
        print(f"Min Profit: {trades_df['Profit'].min():.2f}")
        print(f"Average Hold Time: {trades_df['Hold time (hours)'].mean():.1f} hours")

        return trades_df
    else:
        print("No trades were generated.")
        return pd.DataFrame()

def load_and_process_data(file_path):
    """
    Load data from CSV file and run the trading simulation.

    Args:
        file_path (str): Path to the CSV file containing trading data

    Returns:
        pd.DataFrame: DataFrame with trade results
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)

        # Map column names to match the actual CSV structure
        # Handle both 30min and 60min column names
        column_mapping = {}
        if 'DateTime_30min' in df.columns:
            column_mapping['DateTime_30min'] = 'datetime'
        elif 'DateTime_60min' in df.columns:
            column_mapping['DateTime_60min'] = 'datetime'

        column_mapping.update({
            'Open': 'open',
            'Buy_Entry': 'buy_entry',
            'Buy_Exit': 'buy_exit'
        })

        # Check if required columns exist (using original names)
        required_columns = []
        if 'DateTime_30min' in df.columns or 'DateTime_60min' in df.columns:
            required_columns.append('DateTime_30min' if 'DateTime_30min' in df.columns else 'DateTime_60min')
        required_columns.extend(['Open', 'Buy_Entry', 'Buy_Exit'])
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return pd.DataFrame()

        # Rename columns to match expected format
        df = df.rename(columns=column_mapping)

        # Remove duplicates from loaded data
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            print(f"Removed {initial_rows - len(df)} duplicate rows from loaded data")

        print(f"Loaded data from {file_path}")
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

        # Run simulation
        return simulate_buy_trades(df)

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Use the NIFTY50 RSI data file (located in parent directory)
    # The CSV should have columns: DateTime_30min/DateTime_60min, Open, Buy_Entry, Buy_Exit
    # Can use either RSI_D_W_H.py output or RSI_60_D_W.py output
    data_file = 'NIFTY50_RSI_Daily_Weekly_Hourly_Complete.csv'  # From RSI_D_W_H.py
    # OR use: 'NIFTY50_RSI_60min_Daily_Weekly_Complete.csv'  # From RSI_60_D_W.py

    trades_df = load_and_process_data(data_file)

    if not trades_df.empty:
        print("\nFirst few trades:")
        print(trades_df.head())