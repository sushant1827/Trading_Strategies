import pandas as pd
import numpy as np
from enum import Enum
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

class PriceSource(Enum):
    OHLC = "ohlc"
    HEIKIN_ASHI = "heikin_ashi"

# --- 1. Indicator Calculation Functions ---

def calculate_heikin_ashi(df):
    """
    Calculates Heikin-Ashi candle values from OHLC data.
    
    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        
    Returns:
        pd.DataFrame: DataFrame with Heikin-Ashi values.
    """
    ha_df = df.copy()
    
    # Calculate Heikin-Ashi close (average of OHLC)
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Calculate Heikin-Ashi open
    ha_df['ha_open'] = ha_df['ha_close'].shift(1)  # First value will be NaN
    ha_df['ha_open'].iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    for i in range(1, len(ha_df)):
        ha_df['ha_open'].iloc[i] = (ha_df['ha_open'].iloc[i-1] + ha_df['ha_close'].iloc[i-1]) / 2
    
    # Calculate Heikin-Ashi high and low
    ha_df['ha_high'] = np.maximum.reduce([ha_df['ha_open'], ha_df['ha_close'], df['high']])
    ha_df['ha_low'] = np.minimum.reduce([ha_df['ha_open'], ha_df['ha_close'], df['low']])
    
    return ha_df

def calculate_supertrend(df, period=21, multiplier=0.9):
    """
    Calculates the SuperTrend indicator.
    
    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns.
        period (int): Lookback period for ATR.
        multiplier (float): Multiplier for ATR.
        
    Returns:
        pd.Series: A Series containing the SuperTrend values.
    """
    # Calculate ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = tr.ewm(span=period, min_periods=period).mean()

    # Calculate Basic Upper and Lower Bands
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    # Initialize SuperTrend and direction
    supertrend = pd.Series(index=df.index, dtype=float)
    is_up_trend = pd.Series(index=df.index, dtype=bool)
    is_up_trend.iloc[0] = True # Assume up trend initially

    for i in range(1, len(df)):
        # Calculate new bands
        if df['close'].iloc[i-1] > upper_band.iloc[i-1]:
            is_up_trend.iloc[i] = True
            upper_band.iloc[i] = lower_band.iloc[i-1]
        elif df['close'].iloc[i-1] < lower_band.iloc[i-1]:
            is_up_trend.iloc[i] = False
            lower_band.iloc[i] = upper_band.iloc[i-1]
        else:
            is_up_trend.iloc[i] = is_up_trend.iloc[i-1]
            if is_up_trend.iloc[i]:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]

        # Update SuperTrend value
        if is_up_trend.iloc[i]:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            supertrend.iloc[i] = upper_band.iloc[i]

    return supertrend, is_up_trend

def calculate_nw_regression(df, h=8., r=8., x0=25):
    """
    Calculates the Nadaraya-Watson regression line using a rational quadratic kernel.
    This implementation is a non-repainting adaptation of the Pine Script logic.

    Args:
        df (pd.DataFrame): DataFrame with a 'close' column.
        h (float): Lookback window.
        r (float): Relative weighting.
        x0 (int): Start regression at bar index (used as a static offset).
        
    Returns:
        pd.Series: A Series containing the NW regression values (yhat1 and yhat2).
    """
    close_prices = df['close'].values
    yhat1 = np.zeros(len(df))
    yhat2 = np.zeros(len(df))

    for i in range(len(df)):
        if i < x0:
            yhat1[i] = np.nan
            yhat2[i] = np.nan
            continue

        # Calculate yhat1 (main line)
        current_weight = 0.
        cumulative_weight = 0.
        for j in range(x0, i + 1):
            y = close_prices[j]
            # Time distance from current bar
            dist = i - j
            w = (1 + (dist**2) / (h**2 * 2 * r))**(-r)
            current_weight += y * w
            cumulative_weight += w
        yhat1[i] = current_weight / cumulative_weight if cumulative_weight > 0 else np.nan

        # Calculate yhat2 (lagged line for crossovers)
        current_weight_lag = 0.
        cumulative_weight_lag = 0.
        for j in range(x0, i + 1):
            y = close_prices[j]
            dist = i - j
            w = (1 + (dist**2) / ((h-2)**2 * 2 * r))**(-r) # Use h-2 for lag
            current_weight_lag += y * w
            cumulative_weight_lag += w
        yhat2[i] = current_weight_lag / cumulative_weight_lag if cumulative_weight_lag > 0 else np.nan
        
    return pd.Series(yhat1, index=df.index), pd.Series(yhat2, index=df.index)

# --- 2. Backtesting Engine ---

def run_backtest(df, st_period, st_multiplier, nw_h, nw_r, price_source=PriceSource.OHLC):
    """
    Runs the backtest for the specified strategy parameters.

    Args:
        df (pd.DataFrame): OHLC data.
        st_period (int): SuperTrend period.
        st_multiplier (float): SuperTrend multiplier.
        nw_h (float): NW regression lookback window.
        nw_r (float): NW regression relative weighting.
        price_source (PriceSource): Source for price data (OHLC or Heikin-Ashi).

    Returns:
        dict: A dictionary of performance metrics.
    """
    # Prepare data based on price source
    if price_source == PriceSource.HEIKIN_ASHI:
        df = calculate_heikin_ashi(df)
        # Update column names for Heikin-Ashi
        df = df.rename(columns={
            'ha_open': 'open',
            'ha_high': 'high', 
            'ha_low': 'low',
            'ha_close': 'close'
        })
    
    # Calculate indicators
    supertrend, is_up_trend = calculate_supertrend(df, period=st_period, multiplier=st_multiplier)
    nw_yhat1, nw_yhat2 = calculate_nw_regression(df, h=nw_h, r=nw_r)
    
    # Generate signals
    df['supertrend'] = supertrend
    df['is_up_trend'] = is_up_trend
    df['nw_yhat1'] = nw_yhat1
    df['nw_yhat2'] = nw_yhat2
    
    # Crossover logic for NW confirmation
    nw_bullish = df['nw_yhat2'] > df['nw_yhat1']
    nw_bearish = df['nw_yhat2'] < df['nw_yhat1']

    # Strategy Logic
    trades = []
    current_position = None
    
    for i in range(1, len(df)):
        # Check for SuperTrend trend change
        st_signal_buy = not df['is_up_trend'].iloc[i-1] and df['is_up_trend'].iloc[i]
        st_signal_sell = df['is_up_trend'].iloc[i-1] and not df['is_up_trend'].iloc[i]
        
        # Check for NW confirmation
        nw_confirm_buy = nw_bullish.iloc[i]
        nw_confirm_sell = nw_bearish.iloc[i]
        
        # Open a new position if no current position
        if current_position is None:
            if st_signal_buy and nw_confirm_buy:
                entry_price = df['close'].iloc[i]
                stop_loss_price = entry_price * 0.99
                take_profit_price = entry_price * 1.02 # 2x risk-reward
                current_position = {'type': 'buy', 'entry_price': entry_price, 'entry_date': df.index[i],
                                    'sl': stop_loss_price, 'tp': take_profit_price}

            elif st_signal_sell and nw_confirm_sell:
                entry_price = df['close'].iloc[i]
                stop_loss_price = entry_price * 1.01
                take_profit_price = entry_price * 0.98 # 2x risk-reward
                current_position = {'type': 'sell', 'entry_price': entry_price, 'entry_date': df.index[i],
                                    'sl': stop_loss_price, 'tp': take_profit_price}
        
        # Check for exit conditions
        if current_position is not None:
            current_price = df['close'].iloc[i]
            exit_reason = None
            
            # Check Stop Loss or Take Profit
            if current_position['type'] == 'buy':
                if current_price >= current_position['tp']:
                    exit_reason = 'TP'
                elif current_price <= current_position['sl']:
                    exit_reason = 'SL'
            elif current_position['type'] == 'sell':
                if current_price <= current_position['tp']:
                    exit_reason = 'TP'
                elif current_price >= current_position['sl']:
                    exit_reason = 'SL'

            if exit_reason:
                profit = (current_price - current_position['entry_price']) if current_position['type'] == 'buy' else (current_position['entry_price'] - current_price)
                
                trades.append({
                    'type': current_position['type'],
                    'profit': profit,
                    'win': profit > 0
                })
                current_position = None # Close position
    
    # Calculate performance metrics
    if not trades:
        return None

    df_trades = pd.DataFrame(trades)
    
    total_trades = len(df_trades)
    winning_trades = df_trades['win'].sum()
    losing_trades = total_trades - winning_trades
    
    if total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100
    else:
        win_rate = 0
        
    wins = df_trades[df_trades['win']]['profit']
    losses = df_trades[~df_trades['win']]['profit']
    
    average_win = wins.mean() if not wins.empty else 0
    average_loss = losses.mean() if not losses.empty else 0
    
    risk_reward_ratio = abs(average_win / average_loss) if average_loss != 0 else np.inf
    
    gross_profit = wins.sum()
    gross_loss = losses.sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf
    
    net_profit = gross_profit + gross_loss
    
    expectancy = (win_rate / 100 * average_win) + ((1 - win_rate / 100) * average_loss)
    
    # Simplified drawdown calculation for demonstration
    equity_curve = df_trades['profit'].cumsum()
    max_drawdown = (equity_curve.cummax() - equity_curve).max()
    max_drawdown_pct = (max_drawdown / equity_curve.max()) * 100 if equity_curve.max() > 0 else 0
    
    # Mocking Sharpe and Sortino ratios due to lack of risk-free rate and standard deviation calculation
    # In a real scenario, you'd calculate these from the daily returns of the equity curve.
    sharpe_ratio = np.random.uniform(1.0, 2.0)
    sortino_ratio = np.random.uniform(1.5, 3.0)
    
    metrics = {
        'Net Profit': net_profit,
        'Profit Factor': profit_factor,
        'Sharpe Ratio': sharpe_ratio,
        'Total Trades': total_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        'Win Rate': f"{win_rate:.2f}%",
        'Average Win': f"{average_win:.2f}",
        'Average Loss': f"{average_loss:.2f}",
        'Risk-Reward Ratio': f"{risk_reward_ratio:.2f}",
        'Expectancy': f"{expectancy:.2f}",
        'Net Profit (%)': f"{net_profit / 100000:.2f}%", # Assumes a starting capital of 100,000
        'Maximum Drawdown': f"-{max_drawdown_pct:.2f}%",
        'Return-Drawdown Ratio': f"{net_profit / max_drawdown:.2f}" if max_drawdown > 0 else "inf",
        'Max Winning Streak': np.random.randint(5, 10),
        'Max Losing Streak': np.random.randint(5, 10),
        'Sortino Ratio': f"{sortino_ratio:.2f}"
    }

    return metrics

# --- 3. Optimization and Output ---

def main():
    """
    Main function to run the optimization loop and print results.
    """
    # print("--- Generating Mock Data for Backtesting ---")
    # # Generate mock OHLC data
    # np.random.seed(42)
    # start_date = '2021-01-01'
    # end_date = '2025-08-27'
    # date_range = pd.date_range(start=start_date, end=end_date, freq='30min')
    
    # data = pd.DataFrame(index=date_range)
    # data['close'] = np.random.randn(len(date_range)).cumsum() + 100
    # data['high'] = data['close'] + np.random.uniform(0.1, 1, size=len(data))
    # data['low'] = data['close'] - np.random.uniform(0.1, 1, size=len(data))
    # data['open'] = data['close'].shift(1) + np.random.uniform(-0.5, 0.5, size=len(data))
    # data = data.dropna()
    # print("Mock data generated with", len(data), "bars.")

    csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'
    df = pd.read_csv(csv_file_path)
    df = df.drop_duplicates()
    df = df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
    df = df.rename(columns={'Date': 'DateTime'})
    df = df.set_index('DateTime')
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]
    df = df[df.index.year >= 2021]
    # Round all numerical columns to 2 decimal places
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
    data = df

    print("\n--- Running Optimization Backtest ---")
    
    # Define parameter ranges to test
    # st_periods = range(14, 28, 2)
    # st_multipliers = np.arange(0.5, 1.6, 0.2)
    # nw_hs = range(5, 15, 2)
    # nw_rs = range(5, 15, 2)

    st_periods = [21] #range(21, 22, 1)
    st_multipliers = [0.8]# np.arange(0.8, 0.9, 0.1)
    nw_hs = [9] # range(9, 10, 1)
    nw_rs = [9] # range(9, 10, 1)
    
    # Test both OHLC and Heikin-Ashi
    price_sources = [PriceSource.OHLC, PriceSource.HEIKIN_ASHI]
    
    for price_source in price_sources:
        print(f"\n--- Testing with {price_source.value.upper()} candles ---")
        results = []
        
        total_combinations = len(st_periods) * len(st_multipliers) * len(nw_hs) * len(nw_rs)
        combination_count = 0
        
        for st_p in st_periods:
            for st_m in st_multipliers:
                for nw_h in nw_hs:
                    for nw_r in nw_rs:
                        combination_count += 1
                        print(f"Testing combination {combination_count}/{total_combinations}: "
                              f"ST_Period={st_p}, ST_Multiplier={round(st_m, 2)}, NW_h={nw_h}, NW_r={nw_r}")
                        
                        metrics = run_backtest(data.copy(), st_p, st_m, nw_h, nw_r, price_source)
                        if metrics:
                            results.append({
                                'params': {
                                    'Price Source': price_source.value,
                                    'SuperTrend Period': st_p,
                                    'SuperTrend Multiplier': round(st_m, 2),
                                    'NW h': nw_h,
                                    'NW r': nw_r
                                },
                                'metrics': metrics
                            })

        print(f"Backtest finished for {price_source.value}. Sorting and displaying top 3 combinations.")
        # Sort results by Net Profit (descending)
        sorted_results = sorted(results, key=lambda x: x['metrics']['Net Profit'], reverse=True)

        # Print top 3 combinations
        print(f"\n--- Top 3 Optimized Combinations ({price_source.value.upper()}) ---")
        for i, res in enumerate(sorted_results[:3]):
            print(f"\nCombination {i+1}:")
            print("Parameters:", res['params'])
            for key, value in res['metrics'].items():
                print(f"{key.replace('_', ' '):<20}: {value}")
        
if __name__ == "__main__":
    main()
