import pandas as pd
import numpy as np
import pandas_ta as ta # A common library for technical analysis indicators

# --- Configuration (Equivalent to Pine Script Inputs) ---
RSI_LENGTH = 9
STOCH_LENGTH = 9
K_LENGTH = 3
D_LENGTH = 3
# DI_LENGTH = 10 # Not used in the provided Pine Script for signals
# ADX_SMOOTHING = 14 # Not used in the provided Pine Script for signals
HIGH_TIMEFRAME = '30T' # Assuming '30' means 30 minutes, adjust as per your data frequency

# --- Backtesting Parameters (Equivalent to Pine Script strategy settings) ---
DEFAULT_QTY_TYPE = 'percent_of_equity' # Not directly implemented in this snippet
DEFAULT_QTY_VALUE = 50                 # Not directly implemented in this snippet
BROKERAGE_PCT = 0.001 # Example brokerage for PnL calculation
SLIPPAGE_POINTS = 0.0 # Example slippage for PnL calculation (adjust based on instrument)

# --- Date Range (Equivalent to Pine Script timestamp) ---
START_DATE = pd.Timestamp(2020, 1, 1, 0, 0, 0)
END_DATE = pd.Timestamp(2022, 12, 31, 23, 59, 59)

# --- Helper function for trade profit (simplified, can be expanded) ---
def calculate_pnl(entry_price, exit_price, trade_type, brokerage_pct, slippage_points):
    adjusted_entry_price = entry_price
    adjusted_exit_price = exit_price

    if trade_type == 'long':
        adjusted_entry_price += slippage_points
        adjusted_exit_price -= slippage_points
        gross_pnl = adjusted_exit_price - adjusted_entry_price
    elif trade_type == 'short':
        adjusted_entry_price -= slippage_points
        adjusted_exit_price += slippage_points
        gross_pnl = adjusted_entry_price - adjusted_exit_price
    else:
        return 0

    transaction_cost = (adjusted_entry_price + adjusted_exit_price) * brokerage_pct
    net_pnl = gross_pnl - transaction_cost
    return net_pnl

# --- Main Strategy Function ---
def run_stochrsi_strategy(df_current_tf: pd.DataFrame, df_high_tf: pd.DataFrame):
    """
    Applies the StochRSI strategy to the provided OHLC data.

    Args:
        df_current_tf (pd.DataFrame): DataFrame with OHLC data for the current timeframe.
                                      Must have 'Open', 'High', 'Low', 'Close' and a DateTime index.
        df_high_tf (pd.DataFrame): DataFrame with OHLC data for the higher timeframe.
                                    Must have 'Open', 'High', 'Low', 'Close' and a DateTime index.

    Returns:
        pd.DataFrame: Original DataFrame with StochRSI, signals, and trade history.
    """
    df = df_current_tf.copy()

    # Ensure index is datetime for date range filtering
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # --- Calculate Stochastic RSI (Current Timeframe) ---
    # Pine Script: rsiValue = ta.rsi(close, rsiLength)
    df['rsiValue'] = ta.rsi(df['Close'], length=RSI_LENGTH)

    # Pine Script: rsiMax = ta.highest(ta.rsi(close, rsiLength), stochLength)
    # Equivalent to rolling max of RSI
    df['rsiMax'] = df['rsiValue'].rolling(window=STOCH_LENGTH).max()

    # Pine Script: rsiMin = ta.lowest(ta.rsi(close, rsiLength), stochLength)
    # Equivalent to rolling min of RSI
    df['rsiMin'] = df['rsiValue'].rolling(window=STOCH_LENGTH).min()

    # Pine Script: stochRsiK = ((rsiValue - rsiMin) / (rsiMax - rsiMin)) * 100
    # Note: Handle division by zero for rsiMax - rsiMin
    denominator_k = df['rsiMax'] - df['rsiMin']
    df['stochRsiK'] = np.where(denominator_k != 0, ((df['rsiValue'] - df['rsiMin']) / denominator_k) * 100, 0)
    df['stochRsiK'].fillna(0, inplace=True) # Fill NaNs that might arise from initial periods

    # Pine Script: stochRsiD = ta.sma(stochRsiK, d)
    df['stochRsiD'] = ta.sma(df['stochRsiK'], length=D_LENGTH)


    # --- Calculate Stochastic RSI (Higher Timeframe) ---
    # This section requires careful data handling for real-world application.
    # The Pine Script `request.security(..., lookahead=barmerge.lookahead_on)` is tricky.
    # For backtesting, you'd typically calculate indicators on the higher timeframe
    # and then 'map' them back to the lower timeframe bars.
    # This means, for any bar on the current_tf, you use the *completed* high_tf value
    # from the high_tf bar that *precedes or aligns* with the current_tf bar.

    # Calculate RSI on higher timeframe
    df_high_tf['rsiValue1'] = ta.rsi(df_high_tf['Close'], length=RSI_LENGTH)
    df_high_tf['rsiMax1'] = df_high_tf['rsiValue1'].rolling(window=STOCH_LENGTH).max()
    df_high_tf['rsiMin1'] = df_high_tf['rsiValue1'].rolling(window=STOCH_LENGTH).min()
    denominator_k1 = df_high_tf['rsiMax1'] - df_high_tf['rsiMin1']
    df_high_tf['stochRsiK1_raw'] = np.where(denominator_k1 != 0, ((df_high_tf['rsiValue1'] - df_high_tf['rsiMin1']) / denominator_k1) * 100, 0)
    df_high_tf['stochRsiK1_raw'].fillna(0, inplace=True)
    df_high_tf['stochRsiD1_raw'] = ta.sma(df_high_tf['stochRsiK1_raw'], length=D_LENGTH)

    # Merge higher timeframe indicators back to the current timeframe.
    # This uses `asof` merge, which is suitable for getting the *latest available* value.
    # This simulates `barmerge.lookahead_off` behavior more directly, which is generally safer for live trading.
    # To simulate `lookahead_on`, you'd need to shift this data one bar back or handle it in a way
    # that makes higher timeframe current bar's close data available at lower timeframe's open.
    # For simplicity and general realism, we'll use the last *completed* higher timeframe bar.

    # Align higher timeframe data to current timeframe's index
    # Ensure higher_tf index is also datetime and sorted
    if not isinstance(df_high_tf.index, pd.DatetimeIndex):
        df_high_tf.index = pd.to_datetime(df_high_tf.index)
    df_high_tf = df_high_tf.sort_index()

    # Rename columns to avoid clashes during merge and to clearly indicate source
    df_high_tf_indicators = df_high_tf[['stochRsiK1_raw', 'stochRsiD1_raw']].rename(columns={
        'stochRsiK1_raw': 'stochRsiK1',
        'stochRsiD1_raw': 'stochRsiD1'
    })

    # Perform an asof merge to bring higher TF indicators into the current TF dataframe
    # This will take the value of the most recent completed high_tf bar for each current_tf bar.
    df = pd.merge_asof(df, df_high_tf_indicators, left_index=True, right_index=True, direction='backward')

    # Fill NaNs that occur at the beginning if high_tf data starts later
    df[['stochRsiK1', 'stochRsiD1']] = df[['stochRsiK1', 'stochRsiD1']].fillna(method='ffill')
    df[['stochRsiK1', 'stochRsiD1']] = df[['stochRsiK1', 'stochRsiD1']].fillna(0) # In case ffill leaves NaNs


    # --- Date Range Filtering ---
    in_date_range = (df.index >= START_DATE) & (df.index <= END_DATE)

    # --- Generate Signals ---
    # Pine Script: ta.crossover(stochRsiK, stochRsiD)
    # Pine Script: ta.crossunder(stochRsiK, stochRsiD)
    df['crossover_K_D'] = (df['stochRsiK'].shift(1) <= df['stochRsiD'].shift(1)) & (df['stochRsiK'] > df['stochRsiD'])
    df['crossunder_K_D'] = (df['stochRsiK'].shift(1) >= df['stochRsiD'].shift(1)) & (df['stochRsiK'] < df['stochRsiD'])

    df['crossover_K1_D1'] = (df['stochRsiK1'].shift(1) <= df['stochRsiD1'].shift(1)) & (df['stochRsiK1'] > df['stochRsiD1'])
    df['crossunder_K1_D1'] = (df['stochRsiK1'].shift(1) >= df['stochRsiD1'].shift(1)) & (df['stochRsiK1'] < df['stochRsiD1'])

    # Define Buy Signal
    # Pine Script: buySignal_on = ta.crossover(stochRsiK, stochRsiD) and (ta.crossover(stochRsiK1, stochRsiD1) or stochRsiK1 > stochRsiD1)
    df['buySignal_on'] = df['crossover_K_D'] & (df['crossover_K1_D1'] | (df['stochRsiK1'] > df['stochRsiD1']))

    # Pine Script: buySignal_off = ta.crossunder(stochRsiK, stochRsiD)
    df['buySignal_off'] = df['crossunder_K_D']

    # Define Sell Signal
    # Pine Script: sellSignal_on = ta.crossunder(stochRsiK, stochRsiD) and (ta.crossunder(stochRsiK1, stochRsiD1) or stochRsiK1 < stochRsiD1)
    df['sellSignal_on'] = df['crossunder_K_D'] & (df['crossunder_K1_D1'] | (df['stochRsiK1'] < df['stochRsiD1']))

    # Pine Script: sellSignal_off = ta.crossover(stochRsiK, stochRsiD)
    df['sellSignal_off'] = df['crossover_K_D']

    # --- Backtesting Trade Execution ---
    trades = []
    position = None # 'long', 'short', or None
    entry_price = None
    entry_date = None

    for i in range(len(df)):
        current_date = df.index[i]
        current_open = df['Open'].iloc[i]
        
        # Apply date range filter
        if not in_date_range.iloc[i]:
            # If outside date range and in a position, close it
            if position is not None:
                pnl = calculate_pnl(entry_price, current_open, position, BROKERAGE_PCT, SLIPPAGE_POINTS)
                trades.append({
                    'type': position,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': current_date,
                    'exit_price': current_open,
                    'pnl': pnl
                })
                position = None
                entry_price = None
                entry_date = None
            continue # Skip trading outside the date range

        # Exit logic (execute before entry logic if signals conflict on same bar)
        if position == 'long' and df['buySignal_off'].iloc[i]:
            pnl = calculate_pnl(entry_price, current_open, 'long', BROKERAGE_PCT, SLIPPAGE_POINTS)
            trades.append({
                'type': 'long_exit',
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': current_date,
                'exit_price': current_open,
                'pnl': pnl
            })
            position = None
            entry_price = None
            entry_date = None

        if position == 'short' and df['sellSignal_off'].iloc[i]:
            pnl = calculate_pnl(entry_price, current_open, 'short', BROKERAGE_PCT, SLIPPAGE_POINTS)
            trades.append({
                'type': 'short_exit',
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': current_date,
                'exit_price': current_open,
                'pnl': pnl
            })
            position = None
            entry_price = None
            entry_date = None

        # Entry logic
        if position is None: # Only enter if not already in a position
            if df['buySignal_on'].iloc[i]:
                position = 'long'
                entry_price = current_open
                entry_date = current_date
                trades.append({
                    'type': 'long_entry',
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': np.nan, # To be filled on exit
                    'exit_price': np.nan,
                    'pnl': np.nan
                })
            elif df['sellSignal_on'].iloc[i]:
                position = 'short'
                entry_price = current_open
                entry_date = current_date
                trades.append({
                    'type': 'short_entry',
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': np.nan, # To be filled on exit
                    'exit_price': np.nan,
                    'pnl': np.nan
                })

    # Close any open position at the end of the data if still in one within the date range
    if position is not None and in_date_range.iloc[-1]:
        pnl = calculate_pnl(entry_price, df['Open'].iloc[-1], position, BROKERAGE_PCT, SLIPPAGE_POINTS)
        trades.append({
            'type': f'{position}_exit_final', # Mark as final exit
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': df.index[-1],
            'exit_price': df['Open'].iloc[-1],
            'pnl': pnl
        })

    trades_df = pd.DataFrame(trades)
    return df, trades_df

# --- Example Usage (How to call the function) ---
if __name__ == "__main__":
    # 1. Create Dummy OHLC Data for Current Timeframe (e.g., 5-minute)
    dates_current_tf = pd.date_range(start='2020-01-01', end='2023-01-01', freq='5T')
    df_current_tf = pd.DataFrame({
        'Open': np.random.rand(len(dates_current_tf)) * 100 + 1000,
        'High': np.random.rand(len(dates_current_tf)) * 10 + df_current_tf['Open'] + 5,
        'Low': np.random.rand(len(dates_current_tf)) * 10 + df_current_tf['Open'] - 10,
        'Close': np.random.rand(len(dates_current_tf)) * 10 + df_current_tf['Open']
    }, index=dates_current_tf)
    # Ensure Close is within High/Low
    df_current_tf['Close'] = np.clip(df_current_tf['Close'], df_current_tf['Low'], df_current_tf['High'])
    df_current_tf['Open'] = df_current_tf['Open'].round(2)
    df_current_tf['High'] = df_current_tf['High'].round(2)
    df_current_tf['Low'] = df_current_tf['Low'].round(2)
    df_current_tf['Close'] = df_current_tf['Close'].round(2)

    # 2. Create Dummy OHLC Data for Higher Timeframe (e.g., 30-minute)
    # This simulates fetching 30-minute data.
    # For real data, you would resample your 5-minute data or load actual 30-minute data.
    df_high_tf = df_current_tf.resample(HIGH_TIMEFRAME).ohlc().dropna()
    df_high_tf.columns = df_high_tf.columns.droplevel(0) # Flatten multi-index
    df_high_tf = df_high_tf[['open', 'high', 'low', 'close']].rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
    })


    # Run the strategy
    df_with_indicators, trades_history = run_stochrsi_strategy(df_current_tf, df_high_tf)

    print("DataFrame with Indicators (first 5 rows):")
    print(df_with_indicators.head())

    print("\nDataFrame with Indicators (last 5 rows):")
    print(df_with_indicators.tail())

    print("\nTrades History:")
    if not trades_history.empty:
        print(trades_history.head())
        print(f"\nTotal PnL: {trades_history['pnl'].sum():.2f}")
        print(f"Number of trades: {len(trades_history[trades_history['type'].isin(['long_exit', 'short_exit', 'long_exit_final', 'short_exit_final'])])}")
        print(f"Number of winning trades: {len(trades_history[trades_history['pnl'] > 0])}")
        print(f"Number of losing trades: {len(trades_history[trades_history['pnl'] < 0])}")
    else:
        print("No trades generated.")

    # You can plot these columns to visualize
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(15, 8))
    # plt.plot(df_with_indicators.index, df_with_indicators['Close'], label='Close Price')
    # plt.plot(df_with_indicators.index, df_with_indicators['stochRsiK'], label='StochRSI %K (Current TF)')
    # plt.plot(df_with_indicators.index, df_with_indicators['stochRsiD'], label='StochRSI %D (Current TF)')
    # plt.plot(df_with_indicators.index, df_with_indicators['stochRsiK1'], label='StochRSI %K (High TF)', linestyle='--')
    # plt.plot(df_with_indicators.index, df_with_indicators['stochRsiD1'], label='StochRSI %D (High TF)', linestyle='--')
    # plt.legend()
    # plt.show()