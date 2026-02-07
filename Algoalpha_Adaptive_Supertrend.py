import pandas as pd
import numpy as np
import numba as nb
from datetime import datetime, timedelta
import warnings
import itertools
import multiprocessing as mp
import functools
warnings.filterwarnings('ignore')

@nb.jit(nopython=True)
def supertrend_calculation_nb(high, low, close, factor, atr_array, n):
    hl2 = (high + low) / 2
    upper_band = hl2 + factor * atr_array
    lower_band = hl2 - factor * atr_array
    supertrend = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    for i in range(n):
        if i == 0:
            final_upper[i] = upper_band[i]
            final_lower[i] = lower_band[i]
            direction[i] = 1
            supertrend[i] = final_lower[i]
        else:
            if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                final_lower[i] = lower_band[i]
            else:
                final_lower[i] = final_lower[i-1]
            if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                final_upper[i] = upper_band[i]
            else:
                final_upper[i] = final_upper[i-1]
            if np.isnan(supertrend[i-1]):
                direction[i] = 1
            elif supertrend[i-1] == final_upper[i-1] and close[i] > final_upper[i]:
                direction[i] = -1
            elif supertrend[i-1] == final_lower[i-1] and close[i] < final_lower[i]:
                direction[i] = 1
            else:
                direction[i] = direction[i-1]
            if direction[i] == -1:
                supertrend[i] = final_lower[i]
            else:
                supertrend[i] = final_upper[i]
    return supertrend, direction


@nb.jit(nopython=True)
def calc_ha_open_nb(ha_close, open_, orig_close):
    n = len(ha_close)
    ha_open = np.zeros(n)
    ha_open[0] = (open_[0] + orig_close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    return ha_open

@nb.jit(nopython=True)
def kmeans_clustering_nb(train_data, centroids, max_iterations=20):
    for _ in range(max_iterations):
        prev_centroids = centroids.copy()

        # Vectorized assignment and centroid update
        distances = np.abs(train_data[:, np.newaxis] - centroids)
        clusters = np.argmin(distances, axis=1)
        new_centroids = np.array([
            train_data[clusters == 0].mean() if np.sum(clusters == 0) > 0 else centroids[0],
            train_data[clusters == 1].mean() if np.sum(clusters == 1) > 0 else centroids[1],
            train_data[clusters == 2].mean() if np.sum(clusters == 2) > 0 else centroids[2]
        ])

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, clusters

class MLAdaptiveSuperTrend:
    def __init__(self, atr_len=10, fact=3, training_data_period=100, 
                 highvol=0.75, midvol=0.5, lowvol=0.25):
        self.atr_len = atr_len
        self.fact = fact
        self.training_data_period = training_data_period
        self.highvol = highvol
        self.midvol = midvol
        self.lowvol = lowvol

    def calculate_atr(self, df, period):
        """
        Calculate Average True Range
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.ewm(span=period, adjust=False).mean()  # EMA
        return atr
    
    def heikin_ashi_transform(self, df):
        """Convert OHLC to Heikin Ashi candles"""
        ha_df = df.copy()

        # Calculate Heikin Ashi values
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_open = calc_ha_open_nb(ha_close.values, df['open'].values, df['close'].values)

        ha_high = np.maximum(df['high'], np.maximum(ha_open, ha_close))
        ha_low = np.minimum(df['low'], np.minimum(ha_open, ha_close))

        ha_df['open'] = ha_open
        ha_df['high'] = ha_high
        ha_df['low'] = ha_low
        ha_df['close'] = ha_close

        return ha_df
    
    def kmeans_clustering(self, volatility_data, training_period):
        """
        K-means clustering for volatility-based SuperTrend adaptation
        
        Args:
            volatility_data: Array of ATR values
            training_period: Number of recent periods to use for clustering
        
        Returns:
            Dictionary containing cluster centroids, assignments, and current classification
        """
        # Get recent volatility data for training
        if len(volatility_data) < training_period:
            train_data = volatility_data.copy().values
        else:
            train_data = volatility_data[-training_period:].values

        # Remove any NaN values
        train_data = train_data[~np.isnan(train_data)]

        if len(train_data) < 10:  # Need minimum data points
            return self._default_clustering_result(volatility_data.iloc[-1])

        # Initialize centroids based on percentiles
        upper = np.max(train_data)
        lower = np.min(train_data)

        high_vol_init = lower + (upper - lower) * self.highvol
        mid_vol_init = lower + (upper - lower) * self.midvol
        low_vol_init = lower + (upper - lower) * self.lowvol

        # K-means iterations
        centroids = np.array([high_vol_init, mid_vol_init, low_vol_init])
        centroids, clusters = kmeans_clustering_nb(train_data, centroids)

         # Sort centroids: [high_vol, medium_vol, low_vol]
        sorted_indices = np.argsort(centroids)[::-1]  # Descending order
        sorted_centroids = centroids[sorted_indices]

        # Classify current volatility
        current_vol = volatility_data.iloc[-1]
        distances_current = np.abs(current_vol - sorted_centroids)
        cluster = np.argmin(distances_current)
        assigned_centroid = sorted_centroids[cluster]

        return assigned_centroid, cluster, sorted_centroids[0], sorted_centroids[1], sorted_centroids[2]
    
    def calculate_signals(self, df):
        """Calculate ML Adaptive SuperTrend signals"""
        # Convert to Heikin Ashi
        ha_df = self.heikin_ashi_transform(df)

        # Calculate ATR on Heikin Ashi data
        volatility = self.calculate_atr(ha_df, self.atr_len)

        # Initialize arrays
        supertrend_values = np.full(len(df), np.nan)
        directions = np.full(len(df), np.nan)
        clusters = np.full(len(df), np.nan)

        # Calculate signals
        centroids = np.full(len(df), np.nan)
        for i in range(self.training_data_period, len(df)):
            # Get volatility data up to current point
            vol_data = volatility[:i+1]

            # Perform K-means clustering
            assigned_centroid, cluster, hv, mv, lv = self.kmeans_clustering(
                vol_data, self.training_data_period
            )

            centroids[i] = assigned_centroid
            clusters[i] = cluster
        # Now calculate SuperTrend for the entire period
        ATR_series = pd.Series(centroids)
        st, dir_ = self.supertrend_calculation(ha_df, self.fact, ATR_series)
        supertrend_values = st
        directions = dir_

        return supertrend_values, directions, clusters
    
    def _default_clustering_result(self, current_vol):
        high = current_vol * 1.5
        med = current_vol
        low = current_vol * 0.5
        return med, 1, high, med, low
    

    def supertrend_calculation(self, df, factor, atr, n=None):
        """Calculate SuperTrend indicator"""
        if n is None:
            n = len(df)
        # atr_val = atr.iloc[0]  # ATR is constant - commented for per-bar ATR
        atr_array = atr.values[:n]  # Use ATR array for per-bar calculation
        high = df['high'].values[:n]
        low = df['low'].values[:n]
        close = df['close'].values[:n]
        st, dir_ = supertrend_calculation_nb(high, low, close, factor, atr_array, n)
        return st, dir_


class Backtester:
    def __init__(self, initial_capital=1000000, commission_rate=0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.reset()
    
    def reset(self):
        self.trades = []
        self.equity_curve = []
        self.positions = []
        self.current_position = 0
        self.entry_price = 0
        self.entry_time = None
        self.cash = self.initial_capital
        self.equity = self.initial_capital
    
    def calculate_position_size(self, price):
        """Calculate position size based on available capital"""
        return int(self.cash / price)
    
    def execute_trade(self, price, timestamp, signal):
        """Execute buy/sell trades"""
        if signal == 'buy' and self.current_position <= 0:
            # Close short position if exists
            if self.current_position < 0:
                pnl = (-self.current_position) * (self.entry_price - price)
                commission = 20
                self.cash -= (-self.current_position) * price + commission
                
                # Record trade
                self.trades.append({
                    'entry_time': self.entry_time,
                    'exit_time': timestamp,
                    'entry_price': self.entry_price,
                    'exit_price': price,
                    'position_size': -self.current_position,
                    'pnl': pnl,
                    'commission': commission,
                    'type': 'short',
                    'return_pct': (self.entry_price - price) / self.entry_price * 100
                })
            
            # Open long position
            position_size = self.calculate_position_size(price)
            if position_size > 0:
                commission = 20
                cost = position_size * price + commission
                if self.cash < cost:
                    position_size = max(0, int((self.cash - commission) / price))
                    cost = position_size * price + commission
                if position_size > 0:
                    self.cash -= cost
                    self.current_position = position_size
                    self.entry_price = price
                    self.entry_time = timestamp
        
        elif signal == 'sell' and self.current_position >= 0:
            # Close long position if exists
            if self.current_position > 0:
                pnl = self.current_position * (price - self.entry_price)
                commission = 20
                self.cash += self.current_position * price - commission
                
                # Record trade
                self.trades.append({
                    'entry_time': self.entry_time,
                    'exit_time': timestamp,
                    'entry_price': self.entry_price,
                    'exit_price': price,
                    'position_size': self.current_position,
                    'pnl': pnl,
                    'commission': commission,
                    'type': 'long',
                    'return_pct': (price - self.entry_price) / self.entry_price * 100
                })
            
            # Open short position
            position_size = self.calculate_position_size(price)
            if position_size > 0:
                commission = 20
                self.cash += position_size * price - commission
                self.current_position = -position_size
                self.entry_price = price
                self.entry_time = timestamp
    
    def update_equity(self, price):
        """Update current equity value"""
        if self.current_position > 0:
            self.equity = self.cash + self.current_position * price
        elif self.current_position < 0:
            self.equity = self.cash + self.current_position * (2 * self.entry_price - price)
        else:
            self.equity = self.cash
    
    def run_backtest(self, df, supertrend, directions):
        """Run the backtest"""
        self.reset()
        self.equity_curve.append(self.equity)  # Include initial equity
        
        for i in range(1, len(df)):
            if np.isnan(directions[i]) or np.isnan(directions[i-1]):
                price = df['close'].iloc[i]
                self.update_equity(price)
                self.equity_curve.append(self.equity)
                continue
            
            price = df['close'].iloc[i]
            timestamp = df.index[i]
            
            # Generate signals based on direction changes
            if directions[i-1] == 1 and directions[i] == -1:  # Bullish signal
                self.execute_trade(price, timestamp, 'buy')
            elif directions[i-1] == -1 and directions[i] == 1:  # Bearish signal
                self.execute_trade(price, timestamp, 'sell')
            
            self.update_equity(price)
            self.equity_curve.append(self.equity)
        
        # Close final position if exists
        if self.current_position != 0:
            final_price = df['close'].iloc[-1]
            final_timestamp = df.index[-1]
            
            if self.current_position > 0:
                pnl = self.current_position * (final_price - self.entry_price)
                commission = 20
                self.cash += self.current_position * final_price - commission
                
                self.trades.append({
                    'entry_time': self.entry_time,
                    'exit_time': final_timestamp,
                    'entry_price': self.entry_price,
                    'exit_price': final_price,
                    'position_size': self.current_position,
                    'pnl': pnl,
                    'commission': commission,
                    'type': 'long',
                    'return_pct': (final_price - self.entry_price) / self.entry_price * 100
                })
            else:
                pnl = (-self.current_position) * (self.entry_price - final_price)
                commission = 20
                self.cash += pnl - commission
                
                self.trades.append({
                    'entry_time': self.entry_time,
                    'exit_time': final_timestamp,
                    'entry_price': self.entry_price,
                    'exit_price': final_price,
                    'position_size': abs(self.current_position),
                    'pnl': pnl,
                    'commission': commission,
                    'type': 'short',
                    'return_pct': (self.entry_price - final_price) / self.entry_price * 100
                })
            
            self.current_position = 0
            self.update_equity(final_price)
            # Equity curve already appended in the loop for the last bar


class PerformanceAnalyzer:
    def __init__(self, backtester, df):
        self.backtester = backtester
        self.df = df
        self.trades_df = pd.DataFrame(backtester.trades) if backtester.trades else pd.DataFrame()
        self.equity_curve = np.array(backtester.equity_curve)
        
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if len(self.trades_df) == 0:
            return {}
        
        # Basic metrics
        start_date = self.df.index[0]
        end_date = self.df.index[-1]
        duration = end_date - start_date
        
        initial_capital = self.backtester.initial_capital
        final_equity = self.backtester.equity
        equity_peak = np.max(self.equity_curve) if len(self.equity_curve) > 0 else initial_capital
        
        total_commissions = self.trades_df['commission'].sum()
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        # Buy & Hold return
        buy_hold_return = (self.df['close'].iloc[-1] - self.df['close'].iloc[0]) / self.df['close'].iloc[0] * 100
        
        # Time-based calculations
        years = duration.days / 365.25
        annual_return = ((final_equity / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        cagr = annual_return
        
        # Volatility calculations
        equity_returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        daily_vol = np.std(equity_returns) if len(equity_returns) > 0 else 0
        annual_vol = daily_vol * np.sqrt(252) * 100
        
        # Risk metrics
        risk_free_rate = 0.02  # 2% risk-free rate
        excess_returns = equity_returns - risk_free_rate/252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Sortino ratio
        negative_returns = equity_returns[equity_returns < 0]
        downside_vol = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = np.mean(excess_returns) / downside_vol * np.sqrt(252) if downside_vol > 0 else 0
        
        # Drawdown calculations
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak * 100
        max_drawdown = np.min(drawdown)
        avg_drawdown = np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0
        
        # Drawdown duration
        dd_periods = []
        in_drawdown = False
        dd_start = 0
        
        for i, dd in enumerate(drawdown):
            if dd < -0.01 and not in_drawdown:  # Start of drawdown
                in_drawdown = True
                dd_start = i
            elif dd >= -0.01 and in_drawdown:  # End of drawdown
                in_drawdown = False
                dd_periods.append(i - dd_start)
        
        max_dd_duration = max(dd_periods) if dd_periods else 0
        avg_dd_duration = np.mean(dd_periods) if dd_periods else 0
        
        # Convert to timedelta
        max_dd_duration_td = timedelta(days=max_dd_duration) if max_dd_duration > 0 else timedelta(0)
        avg_dd_duration_td = timedelta(days=avg_dd_duration) if avg_dd_duration > 0 else timedelta(0)
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Alpha and Beta (simplified calculation)
        market_returns = self.df['close'].pct_change().dropna().values
        if len(market_returns) == len(equity_returns):
            beta = np.cov(equity_returns, market_returns)[0,1] / np.var(market_returns) if np.var(market_returns) > 0 else 0
            alpha = (np.mean(equity_returns) - beta * np.mean(market_returns)) * 252 * 100
        else:
            beta = 0
            alpha = 0
        
        # Trade metrics
        num_trades = len(self.trades_df)
        winning_trades = self.trades_df[self.trades_df['return_pct'] > 0]
        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
        
        best_trade = self.trades_df['return_pct'].max() if num_trades > 0 else 0
        worst_trade = self.trades_df['return_pct'].min() if num_trades > 0 else 0
        avg_trade = self.trades_df['return_pct'].mean() if num_trades > 0 else 0
        
        # Trade duration
        if num_trades > 0:
            self.trades_df['duration'] = (
                pd.to_datetime(self.trades_df['exit_time']) - 
                pd.to_datetime(self.trades_df['entry_time'])
            )
            max_trade_duration = self.trades_df['duration'].max()
            avg_trade_duration = self.trades_df['duration'].mean()
        else:
            max_trade_duration = timedelta(0)
            avg_trade_duration = timedelta(0)
        
        # Profit factor
        gross_profit = winning_trades['return_pct'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(self.trades_df[self.trades_df['return_pct'] < 0]['return_pct'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        expectancy = avg_trade
        
        # System Quality Number (SQN)
        sqn = np.sqrt(num_trades) * avg_trade / np.std(self.trades_df['return_pct']) if num_trades > 1 and np.std(self.trades_df['return_pct']) > 0 else 0
        
        # Kelly Criterion
        if num_trades > 0 and win_rate > 0:
            avg_win = winning_trades['return_pct'].mean()
            avg_loss = abs(self.trades_df[self.trades_df['return_pct'] < 0]['return_pct'].mean())
            win_prob = win_rate / 100
            loss_prob = 1 - win_prob
            kelly = (win_prob * avg_win - loss_prob * avg_loss) / avg_win if avg_win > 0 else 0
        else:
            kelly = 0
        
        # Exposure time
        total_bars = len(self.df)
        in_position_bars = sum(1 for i in range(len(self.equity_curve)) if abs(self.equity_curve[i] - self.backtester.cash) > 1)
        exposure_time = in_position_bars / total_bars * 100 if total_bars > 0 else 0
        
        return {
            'Start': start_date,
            'End': end_date,
            'Duration': duration,
            'Exposure Time [%]': round(exposure_time, 5),
            'Equity Final [$]': round(final_equity, 4),
            'Equity Peak [$]': round(equity_peak, 4),
            'Commissions [$]': round(total_commissions, 4),
            'Return [%]': round(total_return, 5),
            'Buy & Hold Return [%]': round(buy_hold_return, 5),
            'Return (Ann.) [%]': round(annual_return, 5),
            'Volatility (Ann.) [%]': round(annual_vol, 5),
            'CAGR [%]': round(cagr, 5),
            'Sharpe Ratio': round(sharpe_ratio, 5),
            'Sortino Ratio': round(sortino_ratio, 5),
            'Calmar Ratio': round(calmar_ratio, 5),
            'Alpha [%]': round(alpha, 5),
            'Beta': round(beta, 5),
            'Max. Drawdown [%]': round(max_drawdown, 5),
            'Avg. Drawdown [%]': round(avg_drawdown, 5),
            'Max. Drawdown Duration': max_dd_duration_td,
            'Avg. Drawdown Duration': avg_dd_duration_td,
            '# Trades': num_trades,
            'Win Rate [%]': round(win_rate, 5),
            'Best Trade [%]': round(best_trade, 5),
            'Worst Trade [%]': round(worst_trade, 5),
            'Avg. Trade [%]': round(avg_trade, 5),
            'Max. Trade Duration': max_trade_duration,
            'Avg. Trade Duration': avg_trade_duration,
            'Profit Factor': round(profit_factor, 5),
            'Expectancy [%]': round(expectancy, 5),
            'SQN': round(sqn, 5),
            'Kelly Criterion': round(kelly, 5)
        }

# Parameter ranges for optimization
param_ranges = {
    'atr_len': [5, 10, 15],
    'fact': [2, 3, 4],
    'training_data_period': [50, 100, 150],
    'highvol': [0.7, 0.75, 0.8],
    'midvol': [0.4, 0.5, 0.6],
    'lowvol': [0.2, 0.25, 0.3]
}

def worker(params, df):
    """Worker function for parallel optimization"""
    atr_len, fact, training_data_period, highvol, midvol, lowvol = params

    # Initialize strategy with current parameters
    ml_st = MLAdaptiveSuperTrend(
        atr_len=atr_len,
        fact=fact,
        training_data_period=training_data_period,
        highvol=highvol,
        midvol=midvol,
        lowvol=lowvol
    )

    # Calculate signals
    supertrend_values, directions, clusters = ml_st.calculate_signals(df)

    # Run backtest
    backtester = Backtester(initial_capital=100000, commission_rate=0.003)
    backtester.run_backtest(df, supertrend_values, directions)

    # Analyze performance
    analyzer = PerformanceAnalyzer(backtester, df)
    metrics = analyzer.calculate_metrics()

    return params, metrics

def optimize_params(df):
    """Optimize parameters using parallel grid search"""
    # Generate all parameter combinations
    param_combos = list(itertools.product(*param_ranges.values()))

    print(f"Starting optimization with {len(param_combos)} parameter combinations...")

    # Create partial function with fixed df
    partial_worker = functools.partial(worker, df=df)

    # Run optimization in parallel
    with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:  # Limit to 8 processes to avoid overload
        results = pool.map(partial_worker, param_combos)

    # Find best result based on Sharpe Ratio
    best_result = max(results, key=lambda x: x[1].get('Sharpe Ratio', -np.inf))
    best_params, best_metrics = best_result

    print("Optimization complete!")
    print(f"Best parameters: {dict(zip(param_ranges.keys(), best_params))}")
    print(f"Best Sharpe Ratio: {best_metrics.get('Sharpe Ratio', 'N/A')}")

    # Save results to CSV
    results_df = pd.DataFrame([
        dict(zip(param_ranges.keys(), params)) | metrics
        for params, metrics in results
    ])
    results_df.to_csv('adaptive_supertrend_optimization_results.csv', index=False)
    print("Results saved to adaptive_supertrend_optimization_results.csv")

    return best_params, best_metrics

def run_ml_supertrend_backtest(df):
    """Main function to run the ML Adaptive SuperTrend backtest"""
    print("Initializing ML Adaptive SuperTrend...")
    
    # Initialize the indicator
    ml_st = MLAdaptiveSuperTrend(
        atr_len=10,
        fact=3,
        training_data_period=100,
        highvol=0.75,
        midvol=0.5,
        lowvol=0.25
    )
    
    print("Calculating signals...")
    # Calculate signals
    supertrend_values, directions, clusters = ml_st.calculate_signals(df)
    
    print("Running backtest...")
    # Initialize backtester
    backtester = Backtester(initial_capital=100000, commission_rate=0.003)
    
    # Run backtest
    backtester.run_backtest(df, supertrend_values, directions)
    
    print("Analyzing performance...")
    # Analyze performance
    analyzer = PerformanceAnalyzer(backtester, df)
    metrics = analyzer.calculate_metrics()
    
    # Save trades to CSV
    if len(analyzer.trades_df) > 0:
        analyzer.trades_df.to_csv('adaptive_supertrend_trades.csv', index=False)
        print(f"\nTrades saved to adaptive_supertrend_trades.csv")

    # Print results
    print("\nBacktest complete")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<30} {value:>15.2f}")
        elif isinstance(value, int):
            print(f"{key:<30} {value:>15}")
        else:
            print(f"{key:<30} {str(value):>15}")

    return backtester, analyzer, metrics

if __name__ == "__main__":

    # Define the path to the CSV file
    csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'

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

    df.columns = df.columns.str.lower()

    print("Data shape:", df.shape)
    print(df.head())

    # Run optimization
    best_params, best_metrics = optimize_params(df)

    # Run backtest with best parameters
    print("\nRunning backtest with optimized parameters...")
    atr_len, fact, training_data_period, highvol, midvol, lowvol = best_params

    ml_st = MLAdaptiveSuperTrend(
        atr_len=atr_len,
        fact=fact,
        training_data_period=training_data_period,
        highvol=highvol,
        midvol=midvol,
        lowvol=lowvol
    )

    supertrend_values, directions, clusters = ml_st.calculate_signals(df)
    backtester = Backtester(initial_capital=100000, commission_rate=0.003)
    backtester.run_backtest(df, supertrend_values, directions)

    analyzer = PerformanceAnalyzer(backtester, df)
    metrics = analyzer.calculate_metrics()

    # Save trades to CSV
    if len(analyzer.trades_df) > 0:
        analyzer.trades_df.to_csv('adaptive_supertrend_optimized_trades.csv', index=False)
        print("Optimized trades saved to adaptive_supertrend_optimized_trades.csv")

    # Print results
    print("\nOptimized Backtest Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<30} {value:>15.2f}")
        elif isinstance(value, int):
            print(f"{key:<30} {value:>15}")
        else:
            print(f"{key:<30} {str(value):>15}")