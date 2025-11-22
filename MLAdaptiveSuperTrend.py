import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

class MLAdaptiveSuperTrend:
    def __init__(self, atr_length=10, st_factor=3.0, training_period=100,
                 high_vol_percentile=0.75, med_vol_percentile=0.5, low_vol_percentile=0.25,
                 cluster_update_frequency=10):
        self.atr_length = atr_length
        self.st_factor = st_factor
        self.training_period = training_period
        self.high_vol_percentile = high_vol_percentile
        self.med_vol_percentile = med_vol_percentile
        self.low_vol_percentile = low_vol_percentile
        self.cluster_update_frequency = cluster_update_frequency  # Update clusters every N bars
        self._cluster_cache = {}  # Cache for cluster centroids
        
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range - Vectorized version"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))

        # Handle first value where shift would be NaN
        high_close[0] = high_low[0]
        low_close[0] = high_low[0]

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return pd.Series(true_range, index=df.index).rolling(window=self.atr_length).mean()
    
    def kmeans_volatility_clustering(self, atr_values: np.array) -> Tuple[np.array, np.array]:
        """Perform K-means clustering on ATR values"""
        # Initial centroids based on percentiles
        atr_range = atr_values.max() - atr_values.min()
        initial_centroids = np.array([
            [atr_values.min() + atr_range * self.high_vol_percentile],   # High vol
            [atr_values.min() + atr_range * self.med_vol_percentile],    # Med vol
            [atr_values.min() + atr_range * self.low_vol_percentile]     # Low vol
        ])
        
        # K-means clustering
        kmeans = KMeans(n_clusters=3, init=initial_centroids, random_state=42, n_init=1)
        clusters = kmeans.fit_predict(atr_values.reshape(-1, 1))
        centroids = kmeans.cluster_centers_.flatten()
        
        # Sort centroids: 0=high, 1=medium, 2=low
        sorted_indices = np.argsort(centroids)[::-1]
        centroid_mapping = {sorted_indices[i]: i for i in range(3)}
        clusters = np.array([centroid_mapping[c] for c in clusters])
        centroids = np.sort(centroids)[::-1]
        
        return clusters, centroids
    
    def calculate_supertrend_vectorized(self, df: pd.DataFrame, adaptive_atr: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate SuperTrend using vectorized operations for better performance"""
        hl2 = (df['high'] + df['low']) / 2

        upper_band = hl2 + (self.st_factor * adaptive_atr)
        lower_band = hl2 - (self.st_factor * adaptive_atr)

        # Initialize arrays
        final_upper_band = upper_band.copy()
        final_lower_band = lower_band.copy()
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        # Initialize first values
        direction.iloc[0] = 1
        supertrend.iloc[0] = final_lower_band.iloc[0]

        # Vectorized calculation for remaining values
        close = df['close'].values
        final_upper = final_upper_band.values
        final_lower = final_lower_band.values
        upper_vals = upper_band.values
        lower_vals = lower_band.values
        supertrend_vals = supertrend.values
        direction_vals = direction.values

        for i in range(1, len(df)):
            # Calculate final bands (with persistence logic)
            if final_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                final_lower[i] = lower_vals[i]
            else:
                final_lower[i] = final_lower[i-1]

            if final_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                final_upper[i] = upper_vals[i]
            else:
                final_upper[i] = final_upper[i-1]

            # Determine SuperTrend direction and value
            prev_supertrend = supertrend_vals[i-1]
            prev_final_upper = final_upper[i-1]

            if prev_supertrend == prev_final_upper:
                direction_vals[i] = -1 if close[i] <= final_lower[i] else 1
            else:
                direction_vals[i] = 1 if close[i] >= final_upper[i] else -1

            if direction_vals[i] == 1:
                supertrend_vals[i] = final_lower[i]
            else:
                supertrend_vals[i] = final_upper[i]

        return pd.Series(supertrend_vals, index=df.index), pd.Series(direction_vals, index=df.index)
    
    def generate_signals_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using vectorized ML Adaptive SuperTrend"""
        df = df.copy()

        # Calculate ATR
        atr = self.calculate_atr(df)
        df['atr'] = atr

        # Initialize result arrays
        df['cluster'] = np.nan
        df['adaptive_atr'] = np.nan
        df['supertrend'] = np.nan
        df['direction'] = np.nan
        df['signal'] = 0

        # Pre-calculate rolling ATR windows for clustering
        print("Pre-calculating volatility clusters...")
        start_time = time.time()

        # Use rolling windows to create training data for each point
        cluster_update_indices = range(self.training_period, len(df), self.cluster_update_frequency)

        for i in tqdm(cluster_update_indices, desc="Processing clusters"):
            # Get training data for this window
            start_idx = i - self.training_period
            end_idx = i
            training_atr = atr.iloc[start_idx:end_idx].dropna()

            if len(training_atr) < self.training_period * 0.8:
                continue

            # Perform clustering (cached to avoid repeated computation)
            cache_key = f"{len(training_atr)}_{training_atr.iloc[0]:.6f}_{training_atr.iloc[-1]:.6f}"
            if cache_key not in self._cluster_cache:
                clusters, centroids = self.kmeans_volatility_clustering(training_atr.values)
                self._cluster_cache[cache_key] = (clusters, centroids)
            else:
                clusters, centroids = self._cluster_cache[cache_key]

            # Apply cluster to next N bars (cluster_update_frequency)
            end_apply = min(i + self.cluster_update_frequency, len(df))
            current_atr_values = atr.iloc[i:end_apply].dropna()

            for j, (idx, current_atr_val) in enumerate(current_atr_values.items()):
                if pd.isna(current_atr_val):
                    continue

                distances = np.abs(current_atr_val - centroids)
                current_cluster = np.argmin(distances)
                current_centroid = centroids[current_cluster]

                df.loc[idx, 'cluster'] = current_cluster
                df.loc[idx, 'adaptive_atr'] = current_centroid

        cluster_time = time.time() - start_time
        print(f"Cluster calculation completed in {cluster_time:.2f} seconds")

        # Calculate SuperTrend using adaptive ATR
        valid_data = df.dropna(subset=['adaptive_atr'])
        if len(valid_data) > 10:
            print("Calculating SuperTrend...")
            st_start = time.time()
            st, direction = self.calculate_supertrend_vectorized(valid_data, valid_data['adaptive_atr'])
            st_time = time.time() - st_start
            print(f"SuperTrend calculation completed in {st_time:.2f} seconds")

            df.loc[valid_data.index, 'supertrend'] = st
            df.loc[valid_data.index, 'direction'] = direction

            # Generate signals based on direction changes (vectorized)
            df['direction_prev'] = df['direction'].shift(1)

            # Buy signal: direction changes from -1 to 1 (bearish to bullish)
            buy_condition = (df['direction'] == 1) & (df['direction_prev'] == -1)
            # Sell signal: direction changes from 1 to -1 (bullish to bearish)
            sell_condition = (df['direction'] == -1) & (df['direction_prev'] == 1)

            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1

            # Add signal filtering to reduce noise
            df['signal'] = self.filter_signals(df)

        return df
    
    def filter_signals(self, df: pd.DataFrame) -> pd.Series:
        """Filter signals to reduce noise and overtrading"""
        signals = df['signal'].copy()
        
        # Remove consecutive same signals (keep only the first)
        signals = signals * (signals != signals.shift(1))
        
        # Minimum bars between signals (prevents excessive trading)
        min_bars_between_signals = 3
        filtered_signals = signals.copy()
        last_signal_idx = None
        
        for i in range(len(signals)):
            if signals.iloc[i] != 0:
                if last_signal_idx is not None and i - last_signal_idx < min_bars_between_signals:
                    filtered_signals.iloc[i] = 0
                else:
                    last_signal_idx = i
        
        return filtered_signals

class BacktestEngine:
    def __init__(self, initial_capital=10000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        
    def run_backtest(self, df: pd.DataFrame, strategy_signals: pd.Series) -> Dict:
        """Run backtest and calculate performance metrics"""
        df = df.copy()
        df['signal'] = strategy_signals
        df['returns'] = df['close'].pct_change()
        
        # Calculate positions (hold position until opposite signal)
        df['raw_position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Only change position on actual signals (not every bar)
        df['position'] = 0
        current_position = 0
        
        for i in range(len(df)):
            signal = df['signal'].iloc[i]
            if signal != 0:  # New signal
                current_position = signal
            df['position'].iloc[i] = current_position
        
        # Calculate strategy returns
        df['position_shifted'] = df['position'].shift(1).fillna(0)
        df['strategy_returns'] = df['position_shifted'] * df['returns']
        
        # Apply commission costs only on position changes
        df['position_change'] = df['position'].diff().abs()
        df['commission_cost'] = df['position_change'] * self.commission
        df['strategy_returns'] = df['strategy_returns'] - df['commission_cost']
        
        # Calculate cumulative returns
        df['cum_strategy_returns'] = (1 + df['strategy_returns']).cumprod()
        df['cum_market_returns'] = (1 + df['returns']).cumprod()
        
        # Calculate metrics
        metrics = self.calculate_metrics(df)
        
        return {
            'df': df,
            'metrics': metrics,
            'trades': self.analyze_trades(df)
        }
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        strategy_returns = df['strategy_returns'].dropna()
        market_returns = df['returns'].dropna()
        
        # Basic metrics
        total_return = df['cum_strategy_returns'].iloc[-1] - 1
        market_return = df['cum_market_returns'].iloc[-1] - 1
        
        # Risk metrics
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() != 0 else 0
        
        # Drawdown
        cumulative = df['cum_strategy_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade metrics
        positions = df['position']
        trades = positions.diff().abs().sum() / 2  # Number of trades
        
        # Win rate
        trade_returns = []
        in_position = False
        entry_price = 0
        
        for i, row in df.iterrows():
            if row['position'] != 0 and not in_position:
                entry_price = row['close']
                in_position = True
                position_type = row['position']
            elif row['position'] == 0 and in_position:
                exit_price = row['close']
                trade_return = (exit_price - entry_price) / entry_price * position_type
                trade_returns.append(trade_return)
                in_position = False
        
        trade_returns = np.array(trade_returns)
        win_rate = len(trade_returns[trade_returns > 0]) / len(trade_returns) if len(trade_returns) > 0 else 0
        avg_win = trade_returns[trade_returns > 0].mean() if len(trade_returns[trade_returns > 0]) > 0 else 0
        avg_loss = trade_returns[trade_returns < 0].mean() if len(trade_returns[trade_returns < 0]) > 0 else 0
        profit_factor = abs(avg_win * len(trade_returns[trade_returns > 0]) / 
                           (avg_loss * len(trade_returns[trade_returns < 0]))) if avg_loss != 0 else np.inf
        
        return {
            'total_return': total_return,
            'market_return': market_return,
            'excess_return': total_return - market_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def analyze_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze individual trades"""
        trades = []
        position = 0
        entry_idx = None
        entry_price = None
        
        for i, row in df.iterrows():
            current_signal = row['signal']
            current_position = row['position']
            
            # Entry condition: signal changes from 0 position to non-zero
            if current_signal != 0 and position == 0:
                position = current_signal
                entry_idx = i
                entry_price = row['close']
                cluster_entry = row.get('cluster', np.nan)
                
            # Exit condition: signal changes to opposite direction
            elif current_signal != 0 and current_signal == -position:
                # Close previous position and open new one
                exit_price = row['close']
                duration = (i - entry_idx).days if hasattr(i - entry_idx, 'days') else len(df.loc[entry_idx:i])
                pnl = (exit_price - entry_price) / entry_price * position
                
                trades.append({
                    'entry_date': entry_idx,
                    'exit_date': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_type': 'Long' if position == 1 else 'Short',
                    'duration': duration,
                    'pnl': pnl,
                    'cluster_entry': cluster_entry
                })
                
                # Start new position
                position = current_signal
                entry_idx = i
                entry_price = row['close']
                cluster_entry = row.get('cluster', np.nan)
        
        # Handle final open position
        if position != 0 and entry_idx is not None:
            final_row = df.iloc[-1]
            exit_price = final_row['close']
            duration = (df.index[-1] - entry_idx).days if hasattr(df.index[-1] - entry_idx, 'days') else len(df.loc[entry_idx:])
            pnl = (exit_price - entry_price) / entry_price * position
            
            trades.append({
                'entry_date': entry_idx,
                'exit_date': df.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_type': 'Long' if position == 1 else 'Short',
                'duration': duration,
                'pnl': pnl,
                'cluster_entry': cluster_entry
            })
        
        return pd.DataFrame(trades)

class StrategyOptimizer:
    def __init__(self, backtest_engine: BacktestEngine):
        self.backtest_engine = backtest_engine
        
    def optimize_parameters(self, df: pd.DataFrame, param_grid: Dict) -> Dict:
        """Optimize strategy parameters using grid search"""
        best_sharpe = -np.inf
        best_params = None
        results = []
        
        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            try:
                # Create strategy with current parameters
                strategy = MLAdaptiveSuperTrend(**params)
                df_with_signals = strategy.generate_signals_vectorized(df.copy())
                
                # Run backtest
                backtest_results = self.backtest_engine.run_backtest(
                    df_with_signals, df_with_signals['signal']
                )
                
                metrics = backtest_results['metrics']
                metrics.update(params)
                results.append(metrics)
                
                # Track best parameters
                if metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = metrics['sharpe_ratio']
                    best_params = params.copy()
                    
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_sharpe': best_sharpe,
            'all_results': pd.DataFrame(results)
        }

def create_sample_data(n_days=1000):
    """Create sample OHLC data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Generate realistic price data with trends and volatility clustering
    returns = np.random.normal(0.001, 0.02, n_days)
    
    # Add volatility clustering
    volatility = np.ones(n_days) * 0.02
    for i in range(1, n_days):
        volatility[i] = 0.95 * volatility[i-1] + 0.05 * abs(returns[i-1]) + np.random.normal(0, 0.001)
        returns[i] = np.random.normal(0.001, volatility[i])
    
    # Create OHLC from returns
    close_prices = 100 * np.cumprod(1 + returns)
    
    # Generate OHLC data
    data = []
    for i in range(n_days):
        base_price = close_prices[i]
        daily_range = base_price * volatility[i] * 2
        
        open_price = base_price + np.random.uniform(-daily_range/4, daily_range/4)
        high_price = max(open_price, base_price) + np.random.uniform(0, daily_range/2)
        low_price = min(open_price, base_price) - np.random.uniform(0, daily_range/2)
        
        data.append({
            'date': dates[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': base_price
        })
    
    return pd.DataFrame(data).set_index('date')

# Example usage and analysis
if __name__ == "__main__":
    # # Create sample data
    # print("Creating sample data...")
    # df = create_sample_data(1000)

    csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'

    df = pd.read_csv(csv_file_path)
    df = df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')    
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]
    # df = df[df.index.year >= 2021]    

    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].round(2)

    df.columns = df.columns.str.lower()
    
    # Initialize strategy and backtest engine
    strategy = MLAdaptiveSuperTrend()
    backtest_engine = BacktestEngine()
    
    # Generate signals
    print("Generating ML Adaptive SuperTrend signals...")
    df_with_signals = strategy.generate_signals_vectorized(df)
    
    # Run backtest
    print("Running backtest...")
    results = backtest_engine.run_backtest(df_with_signals, df_with_signals['signal'])
    
    # Print results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    
    metrics = results['metrics']
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Analyze trades by volatility cluster
    trades_df = results['trades']
    if len(trades_df) > 0:
        print(f"\n{len(trades_df)} trades executed")
        print(f"Profitable trades: {len(trades_df[trades_df['pnl'] > 0])}")
        print(f"Losing trades: {len(trades_df[trades_df['pnl'] < 0])}")
        
        # Performance by volatility cluster
        if 'cluster_entry' in trades_df.columns:
            cluster_performance = trades_df.groupby('cluster_entry')['pnl'].agg(['count', 'mean', 'std'])
            cluster_performance.columns = ['num_trades', 'avg_return', 'volatility']
            print("\nPerformance by Volatility Cluster:")
            print(cluster_performance)
    
    # Parameter optimization example
    print("\nRunning parameter optimization...")
    optimizer = StrategyOptimizer(backtest_engine)
    
    param_grid = {
        'atr_length': [10, 14, 20],
        'st_factor': [2.0, 3.0, 4.0],
        'training_period': [50, 100, 150]
    }
    
    optimization_results = optimizer.optimize_parameters(df, param_grid)
    
    print(f"\nBest parameters: {optimization_results['best_params']}")
    print(f"Best Sharpe ratio: {optimization_results['best_sharpe']:.4f}")
    
    # Show top 5 parameter combinations
    results_df = optimization_results['all_results'].sort_values('sharpe_ratio', ascending=False)
    print("\nTop 5 parameter combinations:")
    print(results_df.head()[['atr_length', 'st_factor', 'training_period', 'sharpe_ratio', 'total_return']].to_string())

    print("\nBacktest analysis complete!")