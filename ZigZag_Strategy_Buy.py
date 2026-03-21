import pandas as pd
import numpy as np
import quantstats as qs
from itertools import product

def heikin_ashi(df):
    """
    Calculates Heikin-Ashi candles from a DataFrame with OHLC data.
    """
    ha_df = pd.DataFrame(index=df.index)
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_df['HA_Open'] = 0.0
    ha_df.iloc[0, ha_df.columns.get_loc('HA_Open')] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2

    for i in range(1, len(df)):
        ha_df.iloc[i, ha_df.columns.get_loc('HA_Open')] = (ha_df.iloc[i-1, ha_df.columns.get_loc('HA_Open')] + ha_df.iloc[i-1, ha_df.columns.get_loc('HA_Close')]) / 2

    ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close']].join(df['High']).max(axis=1)
    ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close']].join(df['Low']).min(axis=1)

    # Rename columns to be compatible with the backtester
    ha_df.rename(columns={'HA_Open': 'Open', 'HA_High': 'High', 'HA_Low': 'Low', 'HA_Close': 'Close'}, inplace=True)
    
    return ha_df

# def zigzag(df, depth=12, deviation=5, backstep=2):
#     """
#     Calculates the ZigZag indicator.
#     """
#     highs = df['High'].values
#     lows = df['Low'].values
    
#     last_high_price = 0
#     last_low_price = 0
#     last_high_bar = 0
#     last_low_bar = 0
    
#     direction = 0
    
#     pivots = []

#     for i in range(len(df)):
#         # Find highest high in the lookback period
#         highest_high = highs[i]
#         highest_bar = i
#         for j in range(max(0, i - depth), i):
#             if highs[j] > highest_high:
#                 highest_high = highs[j]
#                 highest_bar = j

#         # Find lowest low in the lookback period
#         lowest_low = lows[i]
#         lowest_bar = i
#         for j in range(max(0, i - depth), i):
#             if lows[j] < lowest_low:
#                 lowest_low = lows[j]
#                 lowest_bar = j

#         if direction == 0: # Undefined trend
#             if highest_high > last_high_price:
#                 last_high_price = highest_high
#                 last_high_bar = highest_bar
#             if lowest_low < last_low_price:
#                 last_low_price = lowest_low
#                 last_low_bar = lowest_bar
            
#             if last_high_price > 0 and last_low_price > 0:
#                 if (last_high_price - lowest_low) / lowest_low * 100 > deviation:
#                     pivots.append({'bar': last_low_bar, 'price': lowest_low, 'type': 'Low'})
#                     direction = 1
#                 elif (last_high_price - lowest_low) / lowest_low * 100 > deviation:
#                     pivots.append({'bar': last_high_bar, 'price': last_high_price, 'type': 'High'})
#                     direction = -1
        
#         elif direction == 1: # Uptrend
#             if highest_high > last_high_price:
#                 last_high_price = highest_high
#                 last_high_bar = highest_bar
            
#             if (last_high_price - lowest_low) / lowest_low * 100 > deviation and i - last_high_bar >= backstep:
#                 pivots.append({'bar': last_high_bar, 'price': last_high_price, 'type': 'High'})
#                 direction = -1
#                 last_low_price = lowest_low
#                 last_low_bar = lowest_bar

#         elif direction == -1: # Downtrend
#             if lowest_low < last_low_price:
#                 last_low_price = lowest_low
#                 last_low_bar = lowest_bar
            
#             if (last_high_price - last_low_price) / last_low_price * 100 > deviation and i - last_low_bar >= backstep:
#                 pivots.append({'bar': last_low_bar, 'price': last_low_price, 'type': 'Low'})
#                 direction = 1
#                 last_high_price = highest_high
#                 last_high_bar = highest_bar

#     return pivots


def zigzag(df, depth=12, deviation=5, backstep=2):
    """
    Calculates the ZigZag indicator.
    """
    highs = df['High'].values
    lows = df['Low'].values
    
    last_high_price = 0
    last_low_price = float('inf')  # Initialize to infinity for proper comparison
    last_high_bar = 0
    last_low_bar = 0
    
    direction = 0
    
    pivots = []

    for i in range(depth, len(df)):  # Start from depth to have enough lookback data
        # Find highest high in the lookback period
        lookback_start = max(0, i - depth)
        highest_high = np.max(highs[lookback_start:i+1])
        highest_bar = lookback_start + np.argmax(highs[lookback_start:i+1])

        # Find lowest low in the lookback period
        lowest_low = np.min(lows[lookback_start:i+1])
        lowest_bar = lookback_start + np.argmin(lows[lookback_start:i+1])

        if direction == 0:  # Undefined trend - initialization phase
            # Update highest high seen so far
            if highest_high > last_high_price:
                last_high_price = highest_high
                last_high_bar = highest_bar
            
            # Update lowest low seen so far
            if lowest_low < last_low_price:
                last_low_price = lowest_low
                last_low_bar = lowest_bar
            
            # Check if we have enough deviation to establish trend
            if last_high_price > 0 and last_low_price < float('inf'):
                price_change_pct = (last_high_price - last_low_price) / last_low_price * 100
                if price_change_pct > deviation:
                    # Determine which came first to establish initial direction
                    if last_low_bar < last_high_bar:
                        pivots.append({'bar': last_low_bar, 'price': last_low_price, 'type': 'Low'})
                        direction = 1  # Now looking for next high
                    else:
                        pivots.append({'bar': last_high_bar, 'price': last_high_price, 'type': 'High'})
                        direction = -1  # Now looking for next low
        
        elif direction == 1:  # Uptrend - looking for higher high or significant reversal
            # Update if we find a higher high
            if highest_high > last_high_price:
                last_high_price = highest_high
                last_high_bar = highest_bar
            
            # Check for reversal: significant decline from the high
            price_change_pct = (last_high_price - lowest_low) / last_high_price * 100
            if price_change_pct > deviation and i - last_high_bar >= backstep:
                pivots.append({'bar': last_high_bar, 'price': last_high_price, 'type': 'High'})
                direction = -1  # Now looking for next low
                last_low_price = lowest_low
                last_low_bar = lowest_bar

        elif direction == -1:  # Downtrend - looking for lower low or significant reversal
            # Update if we find a lower low
            if lowest_low < last_low_price:
                last_low_price = lowest_low
                last_low_bar = lowest_bar
            
            # Check for reversal: significant rise from the low
            price_change_pct = (highest_high - last_low_price) / last_low_price * 100
            if price_change_pct > deviation and i - last_low_bar >= backstep:
                pivots.append({'bar': last_low_bar, 'price': last_low_price, 'type': 'Low'})
                direction = 1  # Now looking for next high
                last_high_price = highest_high
                last_high_bar = highest_bar

    return pivots


# def backtest(df, pivots):
#     """
#     Backtests the ZigZag trading strategy.
#     """
#     if len(pivots) < 4:
#         # Return a stats dictionary with default zero values if not enough pivots
#         stats = {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0}
#         default_metrics = {'win_rate': 0, 'avg_win': 0, 'avg_loss': 0, 'risk_reward_ratio': 0, 'profit_factor': 0, 'expectancy': 0, 'net_profit': 0, 'net_profit_percent': 0, 'Sharpe Ratio': 0, 'Sortino Ratio': 0, 'Max Drawdown': 0, 'Max Drawdown %': 0}
#         stats.update(default_metrics)
#         return stats

#     signals = []
#     last_highs = []
#     last_lows = []

#     for pivot in pivots:
#         if pivot['type'] == 'High':
#             if len(last_highs) > 0:
#                 if pivot['price'] > last_highs[-1]['price']:
#                     signals.append({'bar': pivot['bar'], 'signal': 'HH'})
#                 else:
#                     signals.append({'bar': pivot['bar'], 'signal': 'LH'})
#             last_highs.append(pivot)
#         else:
#             if len(last_lows) > 0:
#                 if pivot['price'] < last_lows[-1]['price']:
#                     signals.append({'bar': pivot['bar'], 'signal': 'LL'})
#                 else:
#                     signals.append({'bar': pivot['bar'], 'signal': 'HL'})
#             last_lows.append(pivot)
    
#     df['signal'] = np.nan
#     if signals:
#         signals_df = pd.DataFrame(signals)
#         signals_df.set_index('bar', inplace=True)
#         df['signal'] = signals_df['signal']

#     positions = []
#     in_position = False
#     for i in range(len(df)):
#         if df['signal'].iloc[i] == 'HL' and not in_position:
#             positions.append({'entry_bar': i, 'entry_price': df['Close'].iloc[i], 'direction': 'long'})
#             in_position = True
#         elif df['signal'].iloc[i] == 'LH' and in_position:
#             if positions and 'exit_bar' not in positions[-1]:
#                 positions[-1]['exit_bar'] = i
#                 positions[-1]['exit_price'] = df['Close'].iloc[i]
#                 in_position = False
    
#     positions = [p for p in positions if 'exit_bar' in p]
#     trades = pd.DataFrame(positions)

#     stats = {
#         'total_trades': len(trades),
#         'winning_trades': (trades['pnl'] > 0).sum() if not trades.empty else 0,
#         'losing_trades': (trades['pnl'] <= 0).sum() if not trades.empty else 0,
#     }
    
#     if stats['total_trades'] > 0:
#         trades['pnl'] = (trades['exit_price'] - trades['entry_price'])
#         trades['pnl_pct'] = (trades['pnl'] / trades['entry_price']) * 100
#         returns = df['Close'].pct_change()
        
#         stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
#         stats['avg_win'] = trades[trades['pnl'] > 0]['pnl'].mean()
#         stats['avg_loss'] = trades[trades['pnl'] <= 0]['pnl'].mean()
#         stats['risk_reward_ratio'] = abs(stats['avg_win'] / stats['avg_loss']) if stats['avg_loss'] != 0 else np.inf
#         stats['profit_factor'] = trades[trades['pnl'] > 0]['pnl'].sum() / abs(trades[trades['pnl'] <= 0]['pnl'].sum()) if trades[trades['pnl'] <= 0]['pnl'].sum() != 0 else np.inf
#         stats['expectancy'] = (stats['win_rate'] * stats['avg_win']) - ((1 - stats['win_rate']) * abs(stats['avg_loss']))
#         stats['net_profit'] = trades['pnl'].sum()
#         stats['net_profit_percent'] = trades['pnl_pct'].sum()
        
#         qs.extend_pandas()
#         stats['Sharpe Ratio'] = returns.sharpe()
#         stats['Sortino Ratio'] = returns.sortino()
#         stats['Max Drawdown'] = returns.max_drawdown()
#         stats['Max Drawdown %'] = returns.max_drawdown() * 100
#     else:
#         # --- FIX: Ensure all metric keys exist even if there are no trades ---
#         default_metrics = {
#             'win_rate': 0, 'avg_win': 0, 'avg_loss': 0, 'risk_reward_ratio': 0,
#             'profit_factor': 0, 'expectancy': 0, 'net_profit': 0, 'net_profit_percent': 0,
#             'Sharpe Ratio': 0, 'Sortino Ratio': 0, 'Max Drawdown': 0, 'Max Drawdown %': 0
#         }
#         stats.update(default_metrics)
        
#     return stats


def backtest(df, pivots):
    """
    Backtests the ZigZag trading strategy.
    """
    if len(pivots) < 4:
        # Return a stats dictionary with default zero values if not enough pivots
        stats = {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0}
        default_metrics = {'win_rate': 0, 'avg_win': 0, 'avg_loss': 0, 'risk_reward_ratio': 0, 'profit_factor': 0, 'expectancy': 0, 'net_profit': 0, 'net_profit_percent': 0, 'Sharpe Ratio': 0, 'Sortino Ratio': 0, 'Max Drawdown': 0, 'Max Drawdown %': 0}
        stats.update(default_metrics)
        return stats

    signals = []
    last_highs = []
    last_lows = []

    for pivot in pivots:
        if pivot['type'] == 'High':
            if len(last_highs) > 0:
                if pivot['price'] > last_highs[-1]['price']:
                    signals.append({'bar': pivot['bar'], 'signal': 'HH'})
                else:
                    signals.append({'bar': pivot['bar'], 'signal': 'LH'})
            last_highs.append(pivot)
        else:
            if len(last_lows) > 0:
                if pivot['price'] < last_lows[-1]['price']:
                    signals.append({'bar': pivot['bar'], 'signal': 'LL'})
                else:
                    signals.append({'bar': pivot['bar'], 'signal': 'HL'})
            last_lows.append(pivot)
    
    # Initialize signal column with NaN
    df['signal'] = np.nan
    
    # Handle signals with potential duplicate indices
    if signals:
        signals_df = pd.DataFrame(signals)
        
        # Remove duplicate bar indices, keeping the last signal for each bar
        signals_df = signals_df.drop_duplicates(subset=['bar'], keep='last')
        
        # Ensure bar indices are within the DataFrame range
        signals_df = signals_df[signals_df['bar'] < len(df)]
        
        # Set signals using .loc to avoid reindexing issues
        for _, row in signals_df.iterrows():
            bar_idx = int(row['bar'])
            if 0 <= bar_idx < len(df):
                # df.iloc[bar_idx, df.columns.get_loc('signal')] = row['signal']
                df.iloc[bar_idx, df.columns.get_loc('signal')] = str(row['signal'])
                # df['signal'] = pd.Series(dtype=object, index=df.index)  # Creates object dtype column

    positions = []
    in_position = False
    
    for i in range(len(df)):
        current_signal = df['signal'].iloc[i]
        
        if pd.notna(current_signal):  # Check if signal exists at this bar
            if current_signal == 'HL' and not in_position:
                positions.append({
                    'entry_bar': i, 
                    'entry_price': df['Close'].iloc[i], 
                    'direction': 'long'
                })
                in_position = True
            elif current_signal == 'LH' and in_position:
                if positions and 'exit_bar' not in positions[-1]:
                    positions[-1]['exit_bar'] = i
                    positions[-1]['exit_price'] = df['Close'].iloc[i]
                    in_position = False
    
    # Filter out incomplete positions
    positions = [p for p in positions if 'exit_bar' in p]
    
    # Calculate stats
    stats = {
        'total_trades': len(positions),
        'winning_trades': 0,
        'losing_trades': 0,
    }
    
    if stats['total_trades'] > 0:
        trades = pd.DataFrame(positions)
        trades['pnl'] = (trades['exit_price'] - trades['entry_price'])
        trades['pnl_pct'] = (trades['pnl'] / trades['entry_price']) * 100
        
        stats['winning_trades'] = (trades['pnl'] > 0).sum()
        stats['losing_trades'] = (trades['pnl'] <= 0).sum()
        
        # Calculate performance metrics
        returns = df['Close'].pct_change().dropna()
        
        stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
        
        winning_trades = trades[trades['pnl'] > 0]['pnl']
        losing_trades = trades[trades['pnl'] <= 0]['pnl']
        
        stats['avg_win'] = winning_trades.mean() if len(winning_trades) > 0 else 0
        stats['avg_loss'] = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        stats['risk_reward_ratio'] = abs(stats['avg_win'] / stats['avg_loss']) if stats['avg_loss'] != 0 else np.inf
        
        total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        
        stats['profit_factor'] = total_wins / total_losses if total_losses != 0 else np.inf
        stats['expectancy'] = (stats['win_rate'] * stats['avg_win']) - ((1 - stats['win_rate']) * abs(stats['avg_loss']))
        stats['net_profit'] = trades['pnl'].sum()
        stats['net_profit_percent'] = trades['pnl_pct'].sum()
        
        # Calculate risk metrics using quantstats
        try:
            import quantstats as qs
            qs.extend_pandas()
            stats['Sharpe Ratio'] = returns.sharpe() if len(returns) > 1 else 0
            stats['Sortino Ratio'] = returns.sortino() if len(returns) > 1 else 0
            stats['Max Drawdown'] = returns.max_drawdown() if len(returns) > 1 else 0
            stats['Max Drawdown %'] = returns.max_drawdown() * 100 if len(returns) > 1 else 0
        except:
            # Fallback if quantstats fails
            stats['Sharpe Ratio'] = 0
            stats['Sortino Ratio'] = 0
            stats['Max Drawdown'] = 0
            stats['Max Drawdown %'] = 0
    else:
        # No trades - set all metrics to zero
        default_metrics = {
            'win_rate': 0, 'avg_win': 0, 'avg_loss': 0, 'risk_reward_ratio': 0,
            'profit_factor': 0, 'expectancy': 0, 'net_profit': 0, 'net_profit_percent': 0,
            'Sharpe Ratio': 0, 'Sortino Ratio': 0, 'Max Drawdown': 0, 'Max Drawdown %': 0
        }
        stats.update(default_metrics)
        
    return stats


def optimize(df):
    """
    Optimizes the ZigZag parameters.
    """
    depth_range = [10, 12, 15]
    deviation_range = [3, 5, 7]
    backstep_range = [2, 3, 4]

    results = []

    for candle_type in ['normal', 'heikin_ashi']:
        if candle_type == 'heikin_ashi':
            data = heikin_ashi(df.copy())
        else:
            data = df.copy()

        for depth, deviation, backstep in product(depth_range, deviation_range, backstep_range):
            pivots = zigzag(data, depth, deviation, backstep)
            stats = backtest(data.copy(), pivots)
            
            results.append({
                'candle_type': candle_type,
                'depth': depth,
                'deviation': deviation,
                'backstep': backstep,
                **stats
            })

    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("Warning: Optimization produced no results.")
        return results_df

    results_df.to_csv('all_optimization_results.csv', index=False)
    return results_df.sort_values(by='Sharpe Ratio', ascending=False).head(5)

# if __name__ == '__main__':
#     # Create a sample OHLC DataFrame
#     csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_15_Min.csv'

#     df = pd.read_csv(csv_file_path)    
#     df = df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')    
#     df = df.rename(columns={'Date': 'DateTime'})    
#     df = df.set_index('DateTime')
#     df.index = pd.to_datetime(df.index)
#     df = df[~df.index.duplicated(keep='first')]   

#     top_results = optimize(df)

#     print("Top 5 Optimization Results:")
#     print(top_results.to_string())

#     if not top_results.empty:
#         best_params = top_results.iloc[0]
        
#         if best_params['candle_type'] == 'heikin_ashi':
#             signal_df = heikin_ashi(df.copy())
#         else:
#             signal_df = df.copy()

#         best_pivots = zigzag(signal_df, 
#                                depth=int(best_params['depth']), 
#                                deviation=int(best_params['deviation']), 
#                                backstep=int(best_params['backstep']))

#         if best_pivots:
#             signals = []
#             last_highs = []
#             last_lows = []

#             for pivot in best_pivots:
#                 if pivot['type'] == 'High':
#                     if len(last_highs) > 0:
#                         if pivot['price'] > last_highs[-1]['price']:
#                             signals.append({'bar': pivot['bar'], 'signal': 'HH'})
#                         else:
#                             signals.append({'bar': pivot['bar'], 'signal': 'LH'})
#                     last_highs.append(pivot)
#                 else:
#                     if len(last_lows) > 0:
#                         if pivot['price'] < last_lows[-1]['price']:
#                             signals.append({'bar': pivot['bar'], 'signal': 'LL'})
#                         else:
#                             signals.append({'bar': pivot['bar'], 'signal': 'HL'})
#                     last_lows.append(pivot)
            
#             if signals:
#                 signals_df = pd.DataFrame(signals)
#                 signals_df.set_index('bar', inplace=True)
#                 signal_df['signal'] = signals_df['signal']

#         signal_df.to_csv('best_strategy_signals.csv')
#         print("\n✅ Saved all optimization results to 'all_optimization_results.csv'")
#         print("✅ Saved the dataframe with signals for the best strategy to 'best_strategy_signals.csv'")

if __name__ == '__main__':
    # Create a sample OHLC DataFrame
    csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_15_Min.csv'

    df = pd.read_csv(csv_file_path)    
    df = df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')    
    df = df.rename(columns={'Date': 'DateTime'})    
    df = df.set_index('DateTime')
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]   

    top_results = optimize(df)

    print("Top 5 Optimization Results:")
    print(top_results.to_string())

    if not top_results.empty:
        best_params = top_results.iloc[0]
        
        if best_params['candle_type'] == 'heikin_ashi':
            signal_df = heikin_ashi(df.copy())
        else:
            signal_df = df.copy()

        best_pivots = zigzag(signal_df, 
                               depth=int(best_params['depth']), 
                               deviation=int(best_params['deviation']), 
                               backstep=int(best_params['backstep']))

        if best_pivots:
            signals = []
            last_highs = []
            last_lows = []

            for pivot in best_pivots:
                if pivot['type'] == 'High':
                    if len(last_highs) > 0:
                        if pivot['price'] > last_highs[-1]['price']:
                            signals.append({'bar': pivot['bar'], 'signal': 'HH'})
                        else:
                            signals.append({'bar': pivot['bar'], 'signal': 'LH'})
                    last_highs.append(pivot)
                else:
                    if len(last_lows) > 0:
                        if pivot['price'] < last_lows[-1]['price']:
                            signals.append({'bar': pivot['bar'], 'signal': 'LL'})
                        else:
                            signals.append({'bar': pivot['bar'], 'signal': 'HL'})
                    last_lows.append(pivot)
            
            # Initialize signal column
            signal_df['signal'] = np.nan
            
            if signals:
                signals_df = pd.DataFrame(signals)
                
                # Remove duplicate bar indices, keeping the last signal for each bar
                signals_df = signals_df.drop_duplicates(subset=['bar'], keep='last')
                
                # Ensure bar indices are within the DataFrame range
                signals_df = signals_df[signals_df['bar'] < len(signal_df)]
                
                # Safe signal assignment using .iloc instead of reindexing
                for _, row in signals_df.iterrows():
                    bar_idx = int(row['bar'])
                    if 0 <= bar_idx < len(signal_df):
                        signal_df.iloc[bar_idx, signal_df.columns.get_loc('signal')] = row['signal']
        else:
            # No pivots found, initialize empty signal column
            signal_df['signal'] = np.nan

        signal_df.to_csv('best_strategy_signals.csv')
        print("\n✅ Saved all optimization results to 'all_optimization_results.csv'")
        print("✅ Saved the dataframe with signals for the best strategy to 'best_strategy_signals.csv'")
    else:
        print("No optimization results found. Check your data and parameters.")