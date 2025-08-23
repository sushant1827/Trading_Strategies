import pandas as pd
import numpy as np
from numba import jit
import itertools

# Define the path to the CSV file
csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_15_Min.csv'

@jit(nopython=True)
def _calculate_dmi_numba(high, low, close, period):
    """Calculate DMI indicators using Numba for performance"""
    # Initialize arrays
    tr = np.zeros_like(high)
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    # Calculate True Range and Directional Movement
    for i in range(1, len(high)):
        # True Range
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Calculate Smoothed values using Wilder's smoothing
    atr = np.zeros_like(tr)
    plus_di_smooth = np.zeros_like(plus_dm)
    minus_di_smooth = np.zeros_like(minus_dm)
    
    # Initialize first values
    atr[0] = tr[0]
    plus_di_smooth[0] = plus_dm[0]
    minus_di_smooth[0] = minus_dm[0]
    
    # Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / period
    for i in range(1, len(tr)):
        atr[i] = (tr[i] * alpha) + (atr[i-1] * (1 - alpha))
        plus_di_smooth[i] = (plus_dm[i] * alpha) + (plus_di_smooth[i-1] * (1 - alpha))
        minus_di_smooth[i] = (minus_dm[i] * alpha) + (minus_di_smooth[i-1] * (1 - alpha))
    
    # Calculate DI+ and DI-
    plus_di = np.zeros_like(plus_di_smooth)
    minus_di = np.zeros_like(minus_di_smooth)
    
    for i in range(len(tr)):
        if atr[i] != 0:
            plus_di[i] = (plus_di_smooth[i] / atr[i]) * 100
            minus_di[i] = (minus_di_smooth[i] / atr[i]) * 100
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
    # Calculate ADX
    adx = np.zeros_like(plus_di)
    dx = np.zeros_like(plus_di)
    
    for i in range(len(plus_di)):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        else:
            dx[i] = 0
    
    # Smooth DX to get ADX
    for i in range(period, len(dx)):
        if i == period:
            adx[i] = np.mean(dx[1:period+1])
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx[i]) / period
    
    return plus_di, minus_di, adx

def calculate_dmi(df, period=14):
    """Calculate DMI indicators"""
    original_index = df.index
    df_reset = df.reset_index(drop=True)
    
    if not all(col in df_reset.columns for col in ["High", "Low", "Close"]):
        raise ValueError("DataFrame must contain 'High', 'Low', 'Close' columns for DMI calculation.")
    
    high = df_reset["High"].to_numpy()
    low = df_reset["Low"].to_numpy()
    close = df_reset["Close"].to_numpy()
    
    plus_di, minus_di, adx = _calculate_dmi_numba(high, low, close, period)
    
    result_df = pd.DataFrame({
        f"DI_Plus_{period}": plus_di,
        f"DI_Minus_{period}": minus_di,
        f"ADX_{period}": adx
    }, index=original_index)
    
    return result_df

def calculate_ema(df, period, column='Close'):
    """Calculate Exponential Moving Average"""
    return df[column].ewm(span=period, adjust=False).mean()

@jit(nopython=True)
def _run_dmi_ema_strategy_numba(di_plus, di_minus, adx, ema_short, ema_long, close_prices, adx_threshold=20):
    """Run DMI+EMA strategy using Numba for performance"""
    buy_entry_prices = np.full_like(close_prices, np.nan)
    buy_exit_prices = np.full_like(close_prices, np.nan)
    sell_entry_prices = np.full_like(close_prices, np.nan)
    sell_exit_prices = np.full_like(close_prices, np.nan)
    
    in_buy_position = False
    in_sell_position = False
    
    for i in range(1, len(close_prices)):
        # Buy signal: DI+ crosses above DI- AND DI+ > DI- AND ADX > threshold AND EMA short > EMA long
        buy_condition = (di_plus[i] > di_minus[i] and 
                        di_plus[i-1] <= di_minus[i-1] and  # crossover
                        adx[i] > adx_threshold and
                        ema_short[i] > ema_long[i])
        
        # Sell signal: DI- crosses above DI+ AND DI- > DI+ AND ADX > threshold AND EMA short < EMA long
        sell_condition = (di_minus[i] > di_plus[i] and 
                         di_minus[i-1] <= di_plus[i-1] and  # crossover
                         adx[i] > adx_threshold and
                         ema_short[i] < ema_long[i])
        
        # Check for Buy_Entry condition
        if not in_buy_position and not in_sell_position and buy_condition:
            buy_entry_prices[i] = close_prices[i]
            in_buy_position = True
        # Check for Buy_Exit condition (reverse signal or opposite signal)
        elif in_buy_position and (sell_condition or di_plus[i] < di_minus[i]):
            buy_exit_prices[i] = close_prices[i]
            in_buy_position = False

        # Check for Sell_Entry condition
        if not in_sell_position and not in_buy_position and sell_condition:
            sell_entry_prices[i] = close_prices[i]
            in_sell_position = True
        # Check for Sell_Exit condition (reverse signal or opposite signal)
        elif in_sell_position and (buy_condition or di_minus[i] < di_plus[i]):
            sell_exit_prices[i] = close_prices[i]
            in_sell_position = False
            
    return buy_entry_prices, buy_exit_prices, sell_entry_prices, sell_exit_prices

def run_strategy_and_get_metrics(df_original, dmi_period=14, ema_short_period=12, ema_long_period=50, adx_threshold=20):
    """Run DMI+EMA strategy and calculate performance metrics"""
    df = df_original.copy()
    
    # Calculate DMI indicators
    dmi_result = calculate_dmi(df.copy(), period=dmi_period)
    df = pd.concat([df, dmi_result], axis=1)
    
    # Calculate EMAs
    df[f'EMA_{ema_short_period}'] = calculate_ema(df, ema_short_period)
    df[f'EMA_{ema_long_period}'] = calculate_ema(df, ema_long_period)
    
    # Convert to numpy arrays for Numba
    di_plus = df[f'DI_Plus_{dmi_period}'].to_numpy()
    di_minus = df[f'DI_Minus_{dmi_period}'].to_numpy()
    adx = df[f'ADX_{dmi_period}'].to_numpy()
    ema_short = df[f'EMA_{ema_short_period}'].to_numpy()
    ema_long = df[f'EMA_{ema_long_period}'].to_numpy()
    close_prices = df['Close'].to_numpy()
    
    # Run strategy
    buy_entry_prices, buy_exit_prices, sell_entry_prices, sell_exit_prices = _run_dmi_ema_strategy_numba(
        di_plus, di_minus, adx, ema_short, ema_long, close_prices, adx_threshold
    )
    
    df['Buy_Entry'] = buy_entry_prices
    df['Buy_Exit'] = buy_exit_prices
    df['Sell_Entry'] = sell_entry_prices
    df['Sell_Exit'] = sell_exit_prices
    
    # Calculate trades
    trades = []
    current_buy_entry = None
    current_sell_entry = None
    
    for i in range(len(df)):
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
            current_buy_entry = None

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
            current_sell_entry = None

    trades_df = pd.DataFrame(trades)
    
    # Calculate metrics
    metrics = {}
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

        returns_series = trades_df.set_index('exit_date')['profit'].cumsum()
        net_profit = returns_series.iloc[-1] if not returns_series.empty else 0

        initial_capital = 100000
        if not returns_series.empty:
            equity_curve = initial_capital + returns_series.fillna(0)
            peak = equity_curve.expanding(min_periods=1).max()
            drawdown = (equity_curve - peak) / peak
            max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0
        else:
            max_drawdown = 0

        net_profit_percent = (net_profit / initial_capital) * 100 if initial_capital != 0 else 0
        return_drawdown_ratio = (net_profit_percent / abs(max_drawdown)) if max_drawdown != 0 else np.nan

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

        daily_returns = trades_df.set_index('exit_date')['profit'].resample('D').sum().fillna(0)
        annualized_returns = daily_returns.mean() * 252
        annualized_std_dev = daily_returns.std() * np.sqrt(252)

        sharpe_ratio = annualized_returns / annualized_std_dev if annualized_std_dev != 0 else np.nan

        downside_returns = daily_returns[daily_returns < 0]
        downside_std_dev = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
        sortino_ratio = annualized_returns / downside_std_dev if downside_std_dev != 0 else np.nan

        metrics = {
            'total_trades': num_total_trades,
            'winning_trades': num_winning_trades,
            'losing_trades': num_losing_trades,
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2) if not np.isnan(risk_reward_ratio) else np.nan,
            'profit_factor': round(profit_factor, 2) if not np.isnan(profit_factor) else np.nan,
            'expectancy': round(expectancy, 2),
            'net_profit': round(net_profit, 2),
            'net_profit_percent': round(net_profit_percent, 2),
            'max_drawdown': round(max_drawdown, 2),
            'return_drawdown_ratio': round(return_drawdown_ratio, 2) if not np.isnan(return_drawdown_ratio) else np.nan,
            'max_winning_streak': max_winning_streak,
            'max_losing_streak': max_losing_streak,
            'sharpe_ratio': round(sharpe_ratio, 2) if not np.isnan(sharpe_ratio) else np.nan,
            'sortino_ratio': round(sortino_ratio, 2) if not np.isnan(sortino_ratio) else np.nan
        }
    else:
        metrics = {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0,
            'avg_win': 0, 'avg_loss': 0, 'risk_reward_ratio': np.nan, 'profit_factor': np.nan,
            'expectancy': 0, 'net_profit': 0, 'net_profit_percent': 0, 'max_drawdown': 0,
            'return_drawdown_ratio': np.nan, 'max_winning_streak': 0, 'max_losing_streak': 0,
            'sharpe_ratio': np.nan, 'sortino_ratio': np.nan
        }
    
    return metrics

def optimize_parameters(df_original):
    """Optimize strategy parameters"""
    print("Starting parameter optimization...")
    
    # Parameter ranges for optimization
    dmi_periods = [7, 14, 21]
    ema_short_periods = [9, 12, 15]
    ema_long_periods = [26, 50, 75]
    adx_thresholds = [15, 20, 25]
    
    best_metrics = None
    best_params = None
    best_score = -np.inf
    
    total_combinations = len(dmi_periods) * len(ema_short_periods) * len(ema_long_periods) * len(adx_thresholds)
    current_combination = 0
    
    results = []
    
    for dmi_period in dmi_periods:
        for ema_short_period in ema_short_periods:
            for ema_long_period in ema_long_periods:
                for adx_threshold in adx_thresholds:
                    current_combination += 1
                    if ema_short_period >= ema_long_period:  # Skip invalid combinations
                        continue
                    
                    print(f"Testing combination {current_combination}/{total_combinations}: "
                          f"DMI={dmi_period}, EMA_S={ema_short_period}, EMA_L={ema_long_period}, ADX_T={adx_threshold}")
                    
                    try:
                        metrics = run_strategy_and_get_metrics(
                            df_original.copy(),
                            dmi_period=dmi_period,
                            ema_short_period=ema_short_period,
                            ema_long_period=ema_long_period,
                            adx_threshold=adx_threshold
                        )
                        
                        # Use Sharpe ratio as optimization metric (you can change this)
                        score = metrics.get('sharpe_ratio', 0) if not np.isnan(metrics.get('sharpe_ratio', np.nan)) else 0
                        
                        results.append({
                            'DMI_Period': dmi_period,
                            'EMA_Short_Period': ema_short_period,
                            'EMA_Long_Period': ema_long_period,
                            'ADX_Threshold': adx_threshold,
                            'Score': score,
                            **metrics
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_metrics = metrics
                            best_params = {
                                'dmi_period': dmi_period,
                                'ema_short_period': ema_short_period,
                                'ema_long_period': ema_long_period,
                                'adx_threshold': adx_threshold
                            }
                        
                    except Exception as e:
                        print(f"Error with parameters DMI={dmi_period}, EMA_S={ema_short_period}, "
                              f"EMA_L={ema_long_period}, ADX_T={adx_threshold}: {e}")
                        continue
    
    # Save optimization results
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Score', ascending=False)
        results_df.to_csv('dmi_ema_optimization_results.csv', index=False)
        print(f"Optimization results saved to 'dmi_ema_optimization_results.csv'")
    
    return best_params, best_metrics

# Main execution
if __name__ == "__main__":
    try:
        # Load data
        df_original = pd.read_csv(csv_file_path)
        df_original = df_original.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
        df_original = df_original.rename(columns={'Date': 'DateTime'})
        df_original = df_original.set_index('DateTime')
        df_original.index = pd.to_datetime(df_original.index)
        df_original = df_original[~df_original.index.duplicated(keep='first')]
        print(f"After dropping duplicates, df_original index is unique: {df_original.index.is_unique}")
        df_original = df_original[df_original.index.year >= 2021]
        print(f"After filtering for 2021 onwards, df_original shape: {df_original.shape}")

        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df_original.columns:
                df_original[col] = df_original[col].round(2)

        # Test with fixed parameters first
        print("\n=== Testing with Fixed Parameters ===")
        fixed_metrics = run_strategy_and_get_metrics(
            df_original.copy(),
            dmi_period=14,
            ema_short_period=12,
            ema_long_period=50,
            adx_threshold=20
        )
        
        print("\n--- Trading Metrics with Fixed Parameters ---")
        for key, value in fixed_metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Save fixed parameter results
        fixed_results_df = pd.DataFrame([{
            'DMI_Period': 14,
            'EMA_Short_Period': 12,
            'EMA_Long_Period': 50,
            'ADX_Threshold': 20,
            **fixed_metrics
        }])
        fixed_results_df.to_csv('dmi_ema_fixed_results.csv', index=False)
        print(f"\nFixed parameter results saved to 'dmi_ema_fixed_results.csv'")

        # Run optimization
        print("\n=== Running Parameter Optimization ===")
        best_params, best_metrics = optimize_parameters(df_original.copy())
        
        if best_params:
            print(f"\n--- Best Parameters Found ---")
            print(f"DMI Period: {best_params['dmi_period']}")
            print(f"EMA Short Period: {best_params['ema_short_period']}")
            print(f"EMA Long Period: {best_params['ema_long_period']}")
            print(f"ADX Threshold: {best_params['adx_threshold']}")
            
            print(f"\n--- Best Trading Metrics ---")
            for key, value in best_metrics.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
            
            # Save best results
            best_results_df = pd.DataFrame([{
                'DMI_Period': best_params['dmi_period'],
                'EMA_Short_Period': best_params['ema_short_period'],
                'EMA_Long_Period': best_params['ema_long_period'],
                'ADX_Threshold': best_params['adx_threshold'],
                **best_metrics
            }])
            best_results_df.to_csv('dmi_ema_best_results.csv', index=False)
            print(f"\nBest results saved to 'dmi_ema_best_results.csv'")
        else:
            print("No valid parameters found during optimization.")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
