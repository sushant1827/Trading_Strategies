import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

class VIDYAStrategy:
    """
    Volumatic Variable Index Dynamic Average (VIDYA) Trading Strategy
    Adapted for OHLC data only (without volume)
    """
    
    def __init__(self, vidya_length=10, vidya_momentum=20, band_distance=2.0, 
                 atr_period=200, pivot_bars=3):
        self.vidya_length = vidya_length
        self.vidya_momentum = vidya_momentum
        self.band_distance = band_distance
        self.atr_period = atr_period
        self.pivot_bars = pivot_bars
        
    def calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_cmo(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Chande Momentum Oscillator"""
        momentum = data.diff()
        
        pos_momentum = momentum.where(momentum >= 0, 0)
        neg_momentum = momentum.where(momentum < 0, 0).abs()
        
        sum_pos = pos_momentum.rolling(window=period).sum()
        sum_neg = neg_momentum.rolling(window=period).sum()
        
        cmo = 100 * (sum_pos - sum_neg) / (sum_pos + sum_neg)
        return cmo.fillna(0)
    
    def calculate_vidya(self, data: pd.Series, length: int, momentum_period: int) -> pd.Series:
        """Calculate Variable Index Dynamic Average"""
        cmo = self.calculate_cmo(data, momentum_period)
        abs_cmo = np.abs(cmo)
        
        alpha = 2 / (length + 1)
        
        # Initialize VIDYA
        vidya = pd.Series(index=data.index, dtype=float)
        vidya.iloc[0] = data.iloc[0]
        
        for i in range(1, len(data)):
            if pd.isna(abs_cmo.iloc[i]):
                vidya.iloc[i] = vidya.iloc[i-1]
            else:
                alpha_adjusted = alpha * abs_cmo.iloc[i] / 100
                vidya.iloc[i] = (alpha_adjusted * data.iloc[i] + 
                               (1 - alpha_adjusted) * vidya.iloc[i-1])
        
        # Apply additional smoothing (SMA of 15 as in original)
        return vidya.rolling(window=15).mean()
    
    def find_pivots(self, data: pd.Series, left_bars: int, right_bars: int, 
                   pivot_type: str = 'High') -> pd.Series:
        """Find pivot highs or lows"""
        pivots = pd.Series(index=data.index, dtype=bool)
        pivots[:] = False
        
        for i in range(left_bars, len(data) - right_bars):
            if pivot_type == 'High':
                is_pivot = all(data.iloc[i] >= data.iloc[j] 
                             for j in range(i - left_bars, i + right_bars + 1) 
                             if j != i)
            else:  # pivot_type == 'Low'
                is_pivot = all(data.iloc[i] <= data.iloc[j] 
                             for j in range(i - left_bars, i + right_bars + 1) 
                             if j != i)
            
            pivots.iloc[i] = is_pivot
            
        return pivots
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on VIDYA strategy"""
        df = data.copy()
        
        # Calculate VIDYA
        df['vidya'] = self.calculate_vidya(df['Close'], self.vidya_length, 
                                          self.vidya_momentum)
        
        # Calculate ATR and bands
        df['atr'] = self.calculate_atr(df, self.atr_period)
        df['upper_band'] = df['vidya'] + df['atr'] * self.band_distance
        df['lower_band'] = df['vidya'] - df['atr'] * self.band_distance
        
        # Trend detection
        df['trend_up'] = False
        df['trend_up'] = np.where(
            (df['Close'] > df['upper_band']) & 
            (df['Close'].shift(1) <= df['upper_band'].shift(1)), 
            True, df['trend_up']
        )
        df['trend_up'] = np.where(
            (df['Close'] < df['lower_band']) & 
            (df['Close'].shift(1) >= df['lower_band'].shift(1)), 
            False, df['trend_up']
        )
        
        # Forward fill trend
        df['trend_up'] = df['trend_up'].replace(False, np.nan).fillna(method='ffill')
        df['trend_up'] = df['trend_up'].fillna(False)
        
        # Calculate trend line (smoothed value)
        df['trend_line'] = np.where(df['trend_up'], 
                                   df['lower_band'], 
                                   df['upper_band'])
        
        # Find pivot points
        df['pivot_high'] = self.find_pivots(df['High'], self.pivot_bars, 
                                           self.pivot_bars, 'High')
        df['pivot_low'] = self.find_pivots(df['Low'], self.pivot_bars, 
                                          self.pivot_bars, 'Low')
        
        # Generate trading signals
        df['signal'] = 0
        
        # Long signals: Close crosses above upper band
        df['signal'] = np.where(
            (df['Close'] > df['upper_band']) & 
            (df['Close'].shift(1) <= df['upper_band'].shift(1)), 
            1, df['signal']
        )
        
        # Short signals: Close crosses below lower band
        df['signal'] = np.where(
            (df['Close'] < df['lower_band']) & 
            (df['Close'].shift(1) >= df['lower_band'].shift(1)), 
            -1, df['signal']
        )
        
        # Additional filters using price-based volume proxies
        df['range'] = df['High'] - df['Low']
        df['body_size'] = np.abs(df['Close'] - df['Open'])
        df['range_ma'] = df['range'].rolling(window=20).mean()
        df['body_ma'] = df['body_size'].rolling(window=20).mean()
        
        # Filter signals based on volatility
        df['volatility_filter'] = (df['range'] > df['range_ma'] * 0.5) & \
                                 (df['body_size'] > df['body_ma'] * 0.3)
        
        # Apply volatility filter to signals
        df['signal'] = np.where(df['volatility_filter'], df['signal'], 0)
        
        # Create position column
        df['position'] = df['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        return df
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000, 
                transaction_cost: float = 0.001) -> Dict:
        """Backtest the strategy"""
        df = self.generate_signals(data)
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        
        # Apply transaction costs
        position_changes = df['position'].diff().abs()
        df['transaction_costs'] = position_changes * transaction_cost
        df['net_strategy_returns'] = df['strategy_returns'] - df['transaction_costs']
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['cumulative_strategy_returns'] = (1 + df['net_strategy_returns']).cumprod()
        
        # Calculate performance metrics
        total_return = df['cumulative_strategy_returns'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1
        
        # Volatility
        strategy_vol = df['net_strategy_returns'].std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / strategy_vol if strategy_vol > 0 else 0
        
        # Maximum drawdown
        rolling_max = df['cumulative_strategy_returns'].expanding().max()
        drawdown = (df['cumulative_strategy_returns'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = df[df['net_strategy_returns'] > 0]
        total_trades = df[df['net_strategy_returns'] != 0]
        win_rate = len(winning_trades) / len(total_trades) if len(total_trades) > 0 else 0
        
        # Number of trades
        num_trades = len(total_trades)
        
        results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': strategy_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'data': df
        }
        
        return results

class VIDYAOptimizer:
    """Parameter optimizer for VIDYA strategy"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.best_params = None
        self.best_score = float('-inf')
        
    def objective_function(self, params):
        """Objective function for optimization"""
        vidya_length, vidya_momentum, band_distance, atr_period = params
        
        # Convert to integers where needed
        vidya_length = int(vidya_length)
        vidya_momentum = int(vidya_momentum)
        atr_period = int(atr_period)
        
        # Create strategy with parameters
        strategy = VIDYAStrategy(
            vidya_length=vidya_length,
            vidya_momentum=vidya_momentum,
            band_distance=band_distance,
            atr_period=atr_period
        )
        
        try:
            # Backtest
            results = strategy.backtest(self.data)
            
            # Multi-objective optimization score
            # Combine Sharpe ratio, return, and drawdown
            score = (results['sharpe_ratio'] * 0.4 + 
                    results['annualized_return'] * 0.3 - 
                    abs(results['max_drawdown']) * 0.3)
            
            # Penalize strategies with too few trades
            if results['num_trades'] < 10:
                score *= 0.5
                
            return -score  # Negative because we minimize
            
        except Exception as e:
            return 1000  # Large penalty for failed backtests
    
    def optimize(self, bounds=None):
        """Optimize strategy parameters"""
        if bounds is None:
            bounds = [
                (5, 10),     # vidya_length
                (10, 20),   # vidya_momentum
                (0.5, 1.0),  # band_distance
                (50, 100)    # atr_period
            ]
        
        print("Starting parameter optimization...")
        print("This may take several minutes...")
        
        # Use differential evolution for global optimization
        result = differential_evolution(
            self.objective_function,
            bounds,
            seed=42,
            maxiter=50,
            popsize=10,
            disp=True
        )
        
        if result.success:
            self.best_params = {
                'vidya_length': int(result.x[0]),
                'vidya_momentum': int(result.x[1]),
                'band_distance': round(result.x[2], 2),
                'atr_period': int(result.x[3])
            }
            self.best_score = -result.fun
            
            print(f"\nOptimization completed successfully!")
            print(f"Best parameters: {self.best_params}")
            print(f"Best score: {self.best_score:.4f}")
            
        else:
            print("Optimization failed!")
            
        return self.best_params

def load_sample_data():
    """Generate sample OHLC data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Generate realistic price data
    returns = np.random.normal(0.0005, 0.02, len(dates))
    price = 100
    prices = []
    
    for ret in returns:
        price *= (1 + ret)
        prices.append(price)
    
    # Generate OHLC from close prices
    closes = np.array(prices)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    
    # Add some noise for highs and lows
    noise_factor = 0.01
    highs = closes * (1 + np.abs(np.random.normal(0, noise_factor, len(closes))))
    lows = closes * (1 - np.abs(np.random.normal(0, noise_factor, len(closes))))
    
    # Ensure OHLC logic is maintained
    for i in range(len(closes)):
        if opens[i] > closes[i]:  # Red candle
            highs[i] = max(highs[i], opens[i])
            lows[i] = min(lows[i], closes[i])
        else:  # Green candle
            highs[i] = max(highs[i], closes[i])
            lows[i] = min(lows[i], opens[i])
    
    data = pd.DataFrame({
        'date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes
    })
    
    data.set_index('date', inplace=True)
    return data

def plot_results(results_dict: Dict, title: str = "VIDYA Strategy Results"):
    """Plot backtest results"""
    df = results_dict['data']
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Price with VIDYA bands and signals
    axes[0].plot(df.index, df['Close'], label='Close Price', alpha=0.7)
    axes[0].plot(df.index, df['vidya'], label='VIDYA', linewidth=2)
    axes[0].plot(df.index, df['upper_band'], label='Upper Band', linestyle='--', alpha=0.5)
    axes[0].plot(df.index, df['lower_band'], label='Lower Band', linestyle='--', alpha=0.5)
    
    # Plot buy/sell signals
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    
    axes[0].scatter(buy_signals.index, buy_signals['Close'], 
                   color='green', marker='^', s=100, label='Buy Signal', zorder=5)
    axes[0].scatter(sell_signals.index, sell_signals['Close'], 
                   color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    axes[0].set_title(f'{title} - Price and Signals')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative returns
    axes[1].plot(df.index, (df['cumulative_returns'] - 1) * 100, 
                label='Buy & Hold', linewidth=2)
    axes[1].plot(df.index, (df['cumulative_strategy_returns'] - 1) * 100, 
                label='VIDYA Strategy', linewidth=2)
    axes[1].set_title('Cumulative Returns (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Drawdown
    rolling_max = df['cumulative_strategy_returns'].expanding().max()
    drawdown = (df['cumulative_strategy_returns'] - rolling_max) / rolling_max * 100
    
    axes[2].fill_between(df.index, drawdown, 0, alpha=0.3, color='red')
    axes[2].plot(df.index, drawdown, color='red', linewidth=1)
    axes[2].set_title('Strategy Drawdown (%)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_performance_summary(results: Dict):
    """Print performance metrics"""
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Volatility: {results['volatility']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
    print("="*50)

# Example usage
if __name__ == "__main__":

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

    data = df
    # print(f"Data loaded: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
    
    # Test with default parameters
    print("\nTesting with default parameters...")
    strategy = VIDYAStrategy()
    results = strategy.backtest(data)
    print_performance_summary(results)
    
    # Optimize parameters
    print("\nStarting parameter optimization...")
    optimizer = VIDYAOptimizer(data)
    best_params = optimizer.optimize()
    
    if best_params:
        # Test with optimized parameters
        print(f"\nTesting with optimized parameters: {best_params}")
        optimized_strategy = VIDYAStrategy(**best_params)
        optimized_results = optimized_strategy.backtest(data)
        print_performance_summary(optimized_results)
        
        # Plot results
        plot_results(optimized_results, "Optimized VIDYA Strategy")
    
    print("\nStrategy implementation completed!")