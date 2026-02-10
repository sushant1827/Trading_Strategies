import time
import pandas as pd
import numpy as np
from Multitime_Supertrend import SupertrendStrategy as OriginalStrategy
from Multitime_Supertrend_Optimized import SupertrendStrategy as OptimizedStrategy

def load_data():
    """Load and prepare test data"""
    csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_1_Min.csv'
    df = pd.read_csv(csv_file_path)    
    df = df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
    df = df.rename(columns={'Date': 'DateTime'})
    df = df.set_index('DateTime')
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]
    df = df[df.index.year >= 2021]
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].round(2)
    return df

def run_benchmark():
    """Run performance comparison between original and optimized versions"""
    print("Loading data...")
    data = load_data()
    
    # Test parameters
    params_30m = {'period': 20, 'multiplier': 0.9, 'source': 'hl2'}
    params_120m = {'period': 10, 'multiplier': 1.5, 'source': 'hl2'}
    
    # Initialize strategies
    original = OriginalStrategy(initial_capital=100000)
    optimized = OptimizedStrategy(initial_capital=100000)
    
    print("\nRunning original strategy...")
    start_time = time.time()
    original_results = original.backtest_strategy(data, params_30m, params_120m)
    original_time = time.time() - start_time
    
    print("\nRunning optimized strategy...")
    start_time = time.time()
    optimized_results = optimized.backtest_strategy(data, params_30m, params_120m)
    optimized_time = time.time() - start_time
    
    # Print comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"Original Version Time  : {original_time:.2f} seconds")
    print(f"Optimized Version Time : {optimized_time:.2f} seconds")
    print(f"Speed Improvement      : {(original_time/optimized_time):.2f}x faster")
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print("Original Strategy Results:")
    print(f"Total Trades: {original_results.total_trades}")
    print(f"Win Rate: {original_results.win_rate:.2f}%")
    print(f"Net Profit: {original_results.net_profit_pct:.2f}%")
    print(f"Sharpe Ratio: {original_results.sharpe_ratio:.2f}")
    
    print("\nOptimized Strategy Results:")
    print(f"Total Trades: {optimized_results.total_trades}")
    print(f"Win Rate: {optimized_results.win_rate:.2f}%")
    print(f"Net Profit: {optimized_results.net_profit_pct:.2f}%")
    print(f"Sharpe Ratio: {optimized_results.sharpe_ratio:.2f}")

if __name__ == "__main__":
    run_benchmark()
