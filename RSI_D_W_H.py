import pandas as pd
import numpy as np

def calculate_rsi_rma(prices, period=14):
    """
    Calculate RSI using TradingView's RMA (Running Modified Average) method

    Args:
        prices: pandas Series of prices (typically closing prices)
        period: RSI calculation period (default 14)

    Returns:
        pandas Series with RSI values matching TradingView
    """
    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    # Convert to Series for RMA calculation
    gains = pd.Series(gains, index=prices.index)
    losses = pd.Series(losses, index=prices.index)

    # Calculate RMA (Running Modified Average) - TradingView style
    def rma(series, length):
        """Running Modified Average matching TradingView's ta.rma()"""
        rma_values = []
        alpha = 1.0 / length

        for i in range(len(series)):
            if i == 0:
                # First value is just the value itself
                rma_val = series.iloc[0]
                rma_values.append(rma_val)
            else:
                # RMA formula: (previous_RMA * (length - 1) + current_value) / length
                # Or equivalently: previous_RMA * (1 - alpha) + current_value * alpha
                rma_val = rma_values[-1] * (1 - alpha) + series.iloc[i] * alpha
                rma_values.append(rma_val)
        return pd.Series(rma_values, index=series.index)

    # Calculate RMA for gains and losses
    avg_gains = rma(gains, period)
    avg_losses = rma(losses, period)

    # Calculate RSI exactly like TradingView
    rsi = pd.Series(index=prices.index, dtype=float)

    for i in range(len(rsi)):
        up_val = avg_gains.iloc[i]
        down_val = avg_losses.iloc[i]

        if abs(down_val) < 1e-10:  # More precise zero check
            rsi.iloc[i] = 100
        elif abs(up_val) < 1e-10:  # More precise zero check
            rsi.iloc[i] = 0
        else:
            rs_ratio = up_val / down_val
            rsi.iloc[i] = 100 - (100 / (1 + rs_ratio))

    return rsi

def create_weekly_candles(daily_df):
    """
    Create weekly OHLC candles from daily data

    Args:
        daily_df: DataFrame with daily OHLC data

    Returns:
        DataFrame with weekly OHLC candles
    """
    # Ensure DateTime is datetime type
    daily_df['DateTime_Daily'] = pd.to_datetime(daily_df['DateTime_Daily'])

    # Sort by date
    daily_df = daily_df.sort_values('DateTime_Daily')

    # Create week start dates (Monday as start of week)
    daily_df['Week_Start'] = daily_df['DateTime_Daily'].dt.to_period('W').apply(lambda r: r.start_time)

    # Group by week and create OHLC
    weekly_df = daily_df.groupby('Week_Start').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'DateTime_Daily': 'last'  # Use last date as the week date
    }).reset_index()

    # Rename columns appropriately
    weekly_df = weekly_df.rename(columns={
        'Week_Start': 'DateTime_Weekly',
        'DateTime_Daily': 'Week_End_Date'
    })

    return weekly_df

def process_weekly_data(daily_df, rsi_period=14):
    """
    Process weekly data from daily data and calculate weekly RSI

    Args:
        daily_df: DataFrame with daily OHLC data
        rsi_period: RSI calculation period (default 14)

    Returns:
        DataFrame with weekly OHLC and RSI data
    """
    # Create weekly candles
    weekly_df = create_weekly_candles(daily_df)

    if len(weekly_df) < rsi_period:
        print(f"Warning: Not enough weekly data ({len(weekly_df)} weeks) for RSI calculation (need {rsi_period})")
        weekly_df[f'RSI_{rsi_period}_Weekly'] = np.nan
        return weekly_df

    # Calculate weekly RSI using the existing RSI function
    weekly_df[f'RSI_{rsi_period}_Weekly'] = calculate_rsi_rma(weekly_df['Close'], rsi_period).round(2)

    print(f"Weekly data processed. Shape: {weekly_df.shape}")
    print(f"Weekly RSI range: {weekly_df[f'RSI_{rsi_period}_Weekly'].min():.2f} to {weekly_df[f'RSI_{rsi_period}_Weekly'].max():.2f}")

    return weekly_df

def process_daily_data(csv_path):
    """
    Process daily NIFTY50 data similar to 30-minute data

    Args:
        csv_path: Path to daily CSV file

    Returns:
        Processed daily DataFrame
    """
    try:
        # Read the daily CSV file
        df_daily = pd.read_csv(csv_path)

        # Remove unwanted columns
        if 'Unnamed: 0' in df_daily.columns:
            df_daily = df_daily.drop('Unnamed: 0', axis=1)
        if 'Volume' in df_daily.columns:
            df_daily = df_daily.drop('Volume', axis=1)

        # Rename Date column to DateTime (keep as column for daily data)
        if 'Date' in df_daily.columns:
            df_daily = df_daily.rename(columns={'Date': 'DateTime_Daily'})

        # Convert DateTime to datetime type
        df_daily['DateTime_Daily'] = pd.to_datetime(df_daily['DateTime_Daily'])

        # Sort by DateTime column
        df_daily = df_daily.sort_values('DateTime_Daily')

        # Remove duplicates
        df_daily = df_daily.drop_duplicates(subset='DateTime_Daily', keep='last')

        print(f"Daily data processed. Shape: {df_daily.shape}")
        return df_daily

    except Exception as e:
        print(f"Error processing daily data: {str(e)}")
        return None

def debug_rsi_calculation(close_prices, rsi_period=14, max_rows=20):
    """
    Debug function to show step-by-step RSI calculation
    Helps identify where differences with TradingView occur
    """
    print(f"\n=== RSI Debug Info (Period: {rsi_period}) ===")
    print(f"Total data points: {len(close_prices)}")
    print(f"RSI Period: {rsi_period}")
    print(f"Min/Max Close: {close_prices.min():.2f} / {close_prices.max():.2f}")

    # Show first few close prices
    print("\nFirst 10 Close Prices:")
    for i in range(min(10, len(close_prices))):
        print(f"Row {i}: {close_prices.iloc[i]:.2f}")

    # Calculate RSI
    rsi_values = calculate_rsi_rma(close_prices, rsi_period)

    print("\nFirst 15 RSI Values:")
    for i in range(min(15, len(rsi_values))):
        if pd.isna(rsi_values.iloc[i]):
            print(f"Row {i}: NaN")
        else:
            print(f"Row {i}: {rsi_values.iloc[i]:.2f}")

    return rsi_values

def read_nifty50_csv():
    """
    Read NIFTY50 index CSV file and display head and tail rows
    """
    # RSI Configuration - Change these values to modify RSI periods
    RSI_PERIOD_30MIN = 10  # For 30-minute data
    RSI_PERIOD_DAILY = 7  # For daily data
    RSI_PERIOD_WEEKLY = 3  # For weekly data

    # Buy Signal Threshold Configuration - Change these values to modify buy signal conditions
    WEEKLY_RSI_THRESHOLD = 55   # Weekly RSI must be above this value
    DAILY_RSI_THRESHOLD = 55    # Daily RSI must be above this value
    MIN_30_RSI_THRESHOLD = 35   # 30-minute RSI must be above this value

    # Exit Signal Threshold Configuration
    DAILY_RSI_EXIT_THRESHOLD = 50  # Daily RSI exit threshold (signal when RSI crosses below this after being above)

    # Paths to CSV files
    csv_path_30min = "Nifty50_Index/NIFTY50_INDEX_60_Min.csv"
    csv_path_daily = "Nifty50_Index/NIFTY50_INDEX_D_Min.csv"

    
    # Read the 30-minute CSV file
    print("Reading 30-minute data...")
    df = pd.read_csv(csv_path_30min)

    # Remove unwanted columns
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    if 'Volume' in df.columns:
        df = df.drop('Volume', axis=1)

    # Rename Date column to DateTime (keep as column, don't set as index)
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'DateTime_30min'})

    # Convert DateTime to datetime type for filtering
    df['DateTime_30min'] = pd.to_datetime(df['DateTime_30min'])

    # Filter out rows with time 08:15:00
    df = df[~df['DateTime_30min'].dt.strftime('%H:%M:%S').str.contains('08:15:00')]

    # Sort by DateTime column
    df = df.sort_values('DateTime_30min')

    # Remove any duplicate DateTime values
    df = df.drop_duplicates(subset='DateTime_30min', keep='last')

    print(f"30-minute data processed. Shape: {df.shape}")

    # Read and process daily data
    print("\nReading daily data...")
    df_daily = process_daily_data(csv_path_daily)

    if df_daily is not None:
        # Add daily data as additional columns to the 30-minute dataframe
        # Use SAME day's data for correct RSI calculation
        df['Date'] = pd.to_datetime(df['DateTime_30min']).dt.date
        df_daily['Date'] = pd.to_datetime(df_daily['DateTime_Daily']).dt.date

        # Merge with same day's daily data (for correct RSI calculation)
        df = df.merge(df_daily, on='Date', how='left', suffixes=('', '_Daily'))

        # Drop the Date column as it's no longer needed
        df = df.drop('Date', axis=1)

        print(f"Combined data shape: {df.shape}")
        print(f"Daily columns added: {[col for col in df.columns if col.endswith('_Daily')]}")

        # Process weekly data from daily data
        print("\nProcessing weekly data...")
        df_weekly = process_weekly_data(df_daily, RSI_PERIOD_WEEKLY)

        if df_weekly is not None and not df_weekly.empty:
            # Apply previous week shift for all weekly columns (similar to daily shift)
            weekly_cols_to_shift = [col for col in df_weekly.columns if not col.startswith('DateTime')]

            # Get unique week start dates in order
            unique_weeks = sorted(df_weekly['DateTime_Weekly'].unique())

            # Create mapping from current week to previous week
            week_to_prev_week = {}
            for i in range(1, len(unique_weeks)):
                current_week = unique_weeks[i]
                prev_week = unique_weeks[i-1]
                week_to_prev_week[current_week] = prev_week

            # Create a mapping series for faster lookup
            df['Week_Date'] = pd.to_datetime(df['DateTime_30min']).dt.to_period('W').apply(lambda r: r.start_time)
            df['prev_Week_Date'] = df['Week_Date'].map(week_to_prev_week)

            # Apply the previous week shift for each weekly column using vectorized operations
            for col in weekly_cols_to_shift:
                # Create temporary dataframe with weekly data for each week start
                weekly_values = df_weekly.set_index('DateTime_Weekly')[col]

                # Map previous week date to weekly values, handling NaN values
                def get_prev_weekly_value(prev_week):
                    if pd.isna(prev_week):
                        return np.nan
                    return weekly_values.get(prev_week, np.nan)

                # Apply the mapping function
                prev_weekly_values = df['prev_Week_Date'].apply(get_prev_weekly_value).values

                # Add as new column with _Weekly suffix (avoid double suffix)
                if not col.endswith('_Weekly'):
                    df[col + '_Weekly'] = prev_weekly_values
                else:
                    df[col] = prev_weekly_values

            # Also add the DateTime_Weekly column (shifted)
            if 'DateTime_Weekly' in df_weekly.columns:
                # Create a mapping from week to its datetime
                week_to_datetime = dict(zip(df_weekly['DateTime_Weekly'], df_weekly['DateTime_Weekly']))

                def get_prev_weekly_datetime(prev_week):
                    if pd.isna(prev_week):
                        return np.nan
                    return week_to_datetime.get(prev_week, np.nan)

                prev_weekly_datetime = df['prev_Week_Date'].apply(get_prev_weekly_datetime).values
                df['DateTime_Weekly'] = prev_weekly_datetime

            # Clean up temporary columns
            df = df.drop(['Week_Date', 'prev_Week_Date'], axis=1)

            print(f"Weekly columns shifted by 1 week to show PREVIOUS week context")
            print(f"Weekly data merged. Shape: {df.shape}")
            print(f"Weekly columns added: {[col for col in df.columns if col.endswith('_Weekly')]}")

        # Apply simple previous trading day shift for all dates
        daily_cols_to_shift = [col for col in df.columns if col.endswith('_Daily')]

        # Get unique trading dates in order
        unique_dates = sorted(df['DateTime_30min'].dt.date.unique())

        # Create mapping from current date to previous trading date
        date_to_prev_date = {}
        for i in range(1, len(unique_dates)):
            current_date = unique_dates[i]
            prev_date = unique_dates[i-1]
            date_to_prev_date[current_date] = prev_date

        # Create a mapping series for faster lookup
        df['trading_date'] = df['DateTime_30min'].dt.date
        df['prev_trading_date'] = df['trading_date'].map(date_to_prev_date)

        # Apply the previous trading day shift for each daily column using vectorized operations
        for col in daily_cols_to_shift:
            # Create temporary dataframe with daily data for each trading date
            daily_values = df.groupby('trading_date')[col].last()

            # Map previous trading date to daily values, handling NaN values
            def get_prev_daily_value(prev_date):
                if pd.isna(prev_date):
                    return np.nan
                return daily_values.get(prev_date, np.nan)

            # Apply the mapping function
            prev_daily_values = df['prev_trading_date'].apply(get_prev_daily_value).values

            # Update the original column with shifted values
            df[col] = prev_daily_values

        # Clean up temporary columns
        df = df.drop(['trading_date', 'prev_trading_date'], axis=1)

        print(f"Daily columns shifted by 1 trading day to show PREVIOUS trading day context")

    # Calculate 30-minute RSI
    if 'Close' in df.columns and len(df) > RSI_PERIOD_30MIN:
        df[f'RSI_{RSI_PERIOD_30MIN}'] = calculate_rsi_rma(df['Close'], RSI_PERIOD_30MIN).round(2)

    # Calculate Daily RSI properly (accounting for the 1-day shift)
    if 'Close_Daily' in df.columns and df['Close_Daily'].notna().sum() > RSI_PERIOD_DAILY:
        # Create a proper daily series using the shifted daily data but with correct date mapping
        # Group by the shifted dates to get unique daily values for RSI calculation
        shifted_daily_data = []
        for date, group in df.groupby(df['DateTime_30min'].dt.date):
            daily_row = group[['DateTime_Daily', 'Close_Daily']].iloc[0]  # Get first row for each 30-min date
            if not pd.isna(daily_row['Close_Daily']):
                shifted_daily_data.append({
                    'date': date,  # Use the 30-minute date as index
                    'datetime_daily': daily_row['DateTime_Daily'],
                    'close_daily': daily_row['Close_Daily']
                })

        if shifted_daily_data:
            daily_df = pd.DataFrame(shifted_daily_data)
            daily_df = daily_df.sort_values('date')
            daily_series = pd.Series(daily_df['close_daily'].values,
                                    index=pd.to_datetime(daily_df['date']))

            # Calculate RSI on the daily series
            daily_rsi = calculate_rsi_rma(daily_series, RSI_PERIOD_DAILY).round(2)

            # Map back to 30-minute data using the 30-minute dates
            date_rsi_map = dict(zip(daily_series.index.date, daily_rsi))
            df[f'RSI_{RSI_PERIOD_DAILY}_Daily'] = df['DateTime_30min'].dt.date.map(date_rsi_map)

            # Fill NaN values with forward fill (for dates without enough history)
            df[f'RSI_{RSI_PERIOD_DAILY}_Daily'] = df[f'RSI_{RSI_PERIOD_DAILY}_Daily'].ffill()

            print(f"Daily RSI calculation: {len(daily_rsi)} unique days, range: {daily_rsi.min():.2f} to {daily_rsi.max():.2f}")
            print(f"Daily RSI sample values: {daily_rsi.head().to_dict()}")
            print(f"Available dates in daily RSI: {list(date_rsi_map.keys())[:5]}...{list(date_rsi_map.keys())[-5:]}")

    # Create Buy Signal Column based on RSI conditions
    print("\nCreating Buy Signal column...")

    # Initialize buy entry column to 0
    df['Buy_Entry'] = 0

    # Check if all required RSI columns exist
    required_rsi_columns = [
        f'RSI_{RSI_PERIOD_30MIN}',  # 30-minute RSI
        f'RSI_{RSI_PERIOD_DAILY}_Daily',  # Daily RSI
        f'RSI_{RSI_PERIOD_WEEKLY}_Weekly'  # Weekly RSI
    ]

    missing_columns = [col for col in required_rsi_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing RSI columns for buy signal calculation: {missing_columns}")
        print("Buy signal column will contain only zeros")
    else:
        # Create conditions for buy signal
        weekly_rsi_condition = df[f'RSI_{RSI_PERIOD_WEEKLY}_Weekly'] > WEEKLY_RSI_THRESHOLD
        daily_rsi_condition = df[f'RSI_{RSI_PERIOD_DAILY}_Daily'] > DAILY_RSI_THRESHOLD
        min_30_rsi_condition = df[f'RSI_{RSI_PERIOD_30MIN}'] > MIN_30_RSI_THRESHOLD

        # Combine all conditions - all must be true for buy entry
        df['Buy_Entry'] = ((weekly_rsi_condition) &
                            (daily_rsi_condition) &
                            (min_30_rsi_condition)).astype(int)

        # Print buy entry statistics
        total_signals = df['Buy_Entry'].sum()
        total_rows = len(df)
        signal_percentage = (total_signals / total_rows * 100) if total_rows > 0 else 0

        print(f"Buy Signal Configuration:")
        print(f"  - Weekly RSI > {WEEKLY_RSI_THRESHOLD}")
        print(f"  - Daily RSI > {DAILY_RSI_THRESHOLD}")
        print(f"  - 30-min RSI > {MIN_30_RSI_THRESHOLD}")
        print(f"Buy Signal Statistics:")
        print(f"  - Total signals: {total_signals}")
        print(f"  - Total rows: {total_rows}")
        print(f"  - Signal percentage: {signal_percentage:.2f}%")

        # Show sample of recent buy entries
        recent_signals = df[df['Buy_Entry'] == 1].tail(10)
        if not recent_signals.empty:
            print(f"\nRecent Buy Entry Signals (last 10):")
            signal_cols = ['DateTime_30min', f'RSI_{RSI_PERIOD_30MIN}',
                            f'RSI_{RSI_PERIOD_DAILY}_Daily', f'RSI_{RSI_PERIOD_DAILY}_Weekly', 'Buy_Entry']
            available_signal_cols = [col for col in signal_cols if col in df.columns]
            print(recent_signals[available_signal_cols].to_string())
        else:
            print("\nNo recent buy signals found")

    # Create Buy Exit Signal Column based on daily RSI crossing below threshold
    print("\nCreating Buy Exit Signal column...")

    # Initialize buy exit signal column to 0
    df['Buy_Exit'] = 0

    # Check if daily RSI column exists
    daily_rsi_col = f'RSI_{RSI_PERIOD_DAILY}_Daily'
    if daily_rsi_col not in df.columns:
        print(f"Warning: Daily RSI column '{daily_rsi_col}' not found for exit signal calculation")
        print("Buy exit column will contain only zeros")
    else:
        # Create exit signal: current RSI <= threshold AND previous RSI > threshold
        exit_signals = []

        for i in range(len(df)):
            current_rsi = df[daily_rsi_col].iloc[i]

            # Check if current RSI is below or equal to threshold
            if pd.isna(current_rsi) or current_rsi > DAILY_RSI_EXIT_THRESHOLD:
                exit_signals.append(0)
                continue

            # Check if any previous RSI was above threshold
            prev_rsi_above_threshold = False

            # Look back through previous rows to find if RSI was ever above threshold
            for j in range(i):
                prev_rsi = df[daily_rsi_col].iloc[j]
                if not pd.isna(prev_rsi) and prev_rsi > DAILY_RSI_EXIT_THRESHOLD:
                    prev_rsi_above_threshold = True
                    break

            # If current RSI <= threshold and previous RSI was > threshold, it's an exit signal
            if prev_rsi_above_threshold:
                exit_signals.append(1)
            else:
                exit_signals.append(0)

        # Apply the exit signals to the dataframe
        df['Buy_Exit'] = exit_signals

        # Print exit signal statistics
        total_exit_signals = df['Buy_Exit'].sum()
        exit_signal_percentage = (total_exit_signals / total_rows * 100) if total_rows > 0 else 0

        print(f"Buy Exit Signal Configuration:")
        print(f"  - Daily RSI crosses below {DAILY_RSI_EXIT_THRESHOLD} after being above it")
        print(f"Buy Exit Signal Statistics:")
        print(f"  - Total exit signals: {total_exit_signals}")
        print(f"  - Exit signal percentage: {exit_signal_percentage:.2f}%")

        # Show sample of recent exit signals
        recent_exit_signals = df[df['Buy_Exit'] == 1].tail(10)
        if not recent_exit_signals.empty:
            print(f"\nRecent Buy Exit Signals (last 10):")
            exit_cols = ['DateTime_30min', daily_rsi_col, 'Buy_Exit']
            available_exit_cols = [col for col in exit_cols if col in df.columns]
            print(recent_exit_signals[available_exit_cols].to_string())
        else:
            print("\nNo recent buy exit signals found")

    # Print only essential columns for the last 20 rows (including weekly data)
    essential_cols = ['DateTime_30min', 'Close', 'RSI_14', 'DateTime_Daily', 'Close_Daily', 'RSI_14_Daily', 'DateTime_Weekly', 'Close_Weekly', 'RSI_14_Weekly', 'Buy_Entry', 'Buy_Exit']
    available_cols = [col for col in essential_cols if col in df.columns]
    print(f"Available columns: {available_cols}")
    print(df.tail(30)[available_cols].to_string())

    # Save the complete dataframe to CSV file
    output_filename = "NIFTY50_RSI_Daily_Weekly_Hourly_Complete.csv"
    try:
        df.to_csv(output_filename, index=False)
        print(f"\nDataFrame saved successfully to: {output_filename}")
        print(f"File contains {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")

    return df


if __name__ == "__main__":
    df = read_nifty50_csv()