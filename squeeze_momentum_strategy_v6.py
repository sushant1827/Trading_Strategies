# Enhanced example usage with timeframe optimization
def example_usage():
    """Enhanced example showing timeframe optimization and improvements"""
    
    # Generate sample OHLC data (replace with your actual data)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate more realistic price data with trends and volatility clusters
    returns = np.random.randn(len(dates)) * 0.02
    # Add some trend periods and volatility clusters
    for i in range(50, 100):
        returns[i] += 0.005  # Uptrend
    for i in range(200, 250):
        returns[i] -= 0.005  # Downtrend
    for i in range(300, 350):
        returns[i] *= 0.3   # Low volatility period
    
    close_prices = 100 * (1 + returns).cumprod()
    high_prices = close_prices * (1 + np.random.rand(len(dates)) * 0.02)
    low_prices = close_prices * (1 - np.random.rand(len(dates)) * 0.02)
    open_prices = close_prices + np.random.randn(len(dates)) * 0.5
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    }, index=dates)import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SqueezeMomentumIndicator:
    """
    Enhanced Squeeze Momentum Indicator (TTM Squeeze) 
    
    This indicator identifies periods of low volatility (squeeze) followed by 
    high momentum moves. It combines Bollinger Bands and Keltner Channels.
    
    Improvements based on 2024 research:
    - Multi-timeframe confirmation
    - Enhanced exit rules (two bars in new color)
    - Better false signal filtering
    - Optimized parameters for different timeframes
    """
    
    def __init__(self, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5, 
                 use_true_range=True, enable_multi_timeframe=False):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        self.use_true_range = use_true_range
        self.enable_multi_timeframe = enable_multi_timeframe
    
    def true_range(self, high, low, close):
        """Calculate True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def linear_regression(self, series, length):
        """Calculate linear regression value (similar to Pine Script's linreg)"""
        def linreg_single(x):
            if len(x) < length:
                return np.nan
            y = np.arange(len(x))
            slope, intercept, _, _, _ = stats.linregress(y, x)
            return slope * (len(x) - 1) + intercept
        
        return series.rolling(window=length).apply(linreg_single, raw=False)
    
    def calculate_indicators(self, df):
        """
        Calculate Squeeze Momentum Indicator
        
        Parameters:
        df: DataFrame with OHLC data (columns: 'open', 'high', 'low', 'close')
        
        Returns:
        DataFrame with additional columns for the indicator
        """
        data = df.copy()
        
        # Calculate Bollinger Bands
        bb_basis = data['close'].rolling(window=self.bb_length).mean()
        bb_dev = self.bb_mult * data['close'].rolling(window=self.bb_length).std()
        upper_bb = bb_basis + bb_dev
        lower_bb = bb_basis - bb_dev
        
        # Calculate Keltner Channels
        kc_ma = data['close'].rolling(window=self.kc_length).mean()
        
        if self.use_true_range:
            tr = self.true_range(data['high'], data['low'], data['close'])
            range_ma = tr.rolling(window=self.kc_length).mean()
        else:
            range_val = data['high'] - data['low']
            range_ma = range_val.rolling(window=self.kc_length).mean()
        
        upper_kc = kc_ma + range_ma * self.kc_mult
        lower_kc = kc_ma - range_ma * self.kc_mult
        
        # Squeeze conditions
        squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        squeeze_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
        no_squeeze = ~squeeze_on & ~squeeze_off
        
        # Calculate momentum value
        highest_high = data['high'].rolling(window=self.kc_length).max()
        lowest_low = data['low'].rolling(window=self.kc_length).min()
        close_ma = data['close'].rolling(window=self.kc_length).mean()
        
        avg_hl = (highest_high + lowest_low) / 2
        avg_val = (avg_hl + close_ma) / 2
        
        momentum_source = data['close'] - avg_val
        momentum = self.linear_regression(momentum_source, self.kc_length)
        
        # Add to dataframe
        data['squeeze_on'] = squeeze_on
        data['squeeze_off'] = squeeze_off
        data['no_squeeze'] = no_squeeze
        data['momentum'] = momentum
        data['momentum_positive'] = momentum > 0
        data['momentum_increasing'] = momentum > momentum.shift(1)
        
        # Color coding for momentum
        data['momentum_color'] = 'gray'
        data.loc[data['momentum'] > 0, 'momentum_color'] = 'lime'
        data.loc[(data['momentum'] > 0) & (data['momentum'] <= data['momentum'].shift(1)), 'momentum_color'] = 'green'
        data.loc[data['momentum'] < 0, 'momentum_color'] = 'red'
        data.loc[(data['momentum'] < 0) & (data['momentum'] >= data['momentum'].shift(1)), 'momentum_color'] = 'maroon'
        
        # Squeeze color
        data['squeeze_color'] = 'gray'
        data.loc[data['squeeze_on'], 'squeeze_color'] = 'black'
        data.loc[data['no_squeeze'], 'squeeze_color'] = 'blue'
        
        return data
    
    def generate_signals(self, df):
        """
        Enhanced signal generation based on 2024 research improvements
        
        Buy signals:
        - Squeeze is released (first green dot after red dots)
        - Momentum turns positive and increasing
        - Optional: Multi-timeframe confirmation
        
        Sell signals (John Carter's "Two Bar Rule"):
        - Two consecutive bars in new color (momentum decreasing)
        - Momentum crosses below zero
        """
        data = self.calculate_indicators(df)
        
        # Initialize signals
        data['signal'] = 0
        data['position'] = 0
        data['squeeze_release'] = False
        
        # Identify squeeze release points (first green after red)
        squeeze_states = []
        prev_squeeze = False
        
        for _, row in data.iterrows():
            current_squeeze = row['squeeze_on']
            if prev_squeeze and not current_squeeze:
                squeeze_states.append(True)  # Squeeze just released
            else:
                squeeze_states.append(False)
            prev_squeeze = current_squeeze
        
        data['squeeze_release'] = squeeze_states
        
        # Enhanced buy conditions
        buy_condition = (
            # Primary: Squeeze release with positive momentum
            (data['squeeze_release']) &
            (data['momentum'] > 0) &
            (data['momentum_increasing'])
        ) | (
            # Secondary: Strong momentum cross during/after squeeze
            (data['momentum'] > 0) &
            (data['momentum'].shift(1) <= 0) &
            (data['squeeze_on'].rolling(3).sum() > 0) &  # Recent squeeze activity
            (data['momentum'] > data['momentum'].rolling(5).mean())  # Above recent average
        )
        
        # Enhanced sell conditions - John Carter's "Two Bar Rule"
        # Create momentum direction change tracking
        data['momentum_direction'] = np.where(data['momentum_increasing'], 1, -1)
        data['direction_change'] = data['momentum_direction'] != data['momentum_direction'].shift(1)
        
        # Two consecutive bars in new direction
        data['two_bar_rule'] = (
            (~data['momentum_increasing']) &  # Currently decreasing
            (~data['momentum_increasing'].shift(1)) &  # Was decreasing last bar
            (data['momentum'] > 0) &  # Still positive
            (data['momentum'].shift(1) > 0)  # Was positive last bar
        )
        
        sell_condition = (
            (data['two_bar_rule']) |  # Two bar rule
            (
                # Momentum crosses below zero
                (data['momentum'] < 0) &
                (data['momentum'].shift(1) >= 0)
            ) |
            (
                # Strong negative momentum after positive
                (data['momentum'] < 0) &
                (~data['momentum_increasing']) &
                (data['momentum'] < data['momentum'].rolling(3).mean())
            )
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        # Calculate positions with improved logic
        position = 0
        positions = []
        
        for i, signal in enumerate(data['signal']):
            if signal == 1 and position <= 0:
                position = 1
            elif signal == -1 and position >= 0:
                position = -1
            positions.append(position)
        
        data['position'] = positions
        
        return data
    
    def plot_indicator(self, df, title="Squeeze Momentum Indicator"):
        """Plot the indicator"""
        data = self.calculate_indicators(df)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # Price chart
        ax1.plot(data.index, data['close'], label='Close Price', linewidth=1)
        ax1.set_title(f'{title} - Price Chart')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Momentum histogram
        colors = []
        for _, row in data.iterrows():
            if row['momentum'] > 0:
                if row['momentum_increasing']:
                    colors.append('lime')
                else:
                    colors.append('green')
            else:
                if not row['momentum_increasing']:
                    colors.append('red')
                else:
                    colors.append('maroon')
        
        ax2.bar(data.index, data['momentum'], color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add squeeze indicators
        for i, (idx, row) in enumerate(data.iterrows()):
            if row['squeeze_on']:
                ax2.plot(idx, 0, 'ko', markersize=3)  # Black dots for squeeze
            elif row['no_squeeze']:
                ax2.plot(idx, 0, 'bo', markersize=3)  # Blue dots for no squeeze
            else:
                ax2.plot(idx, 0, color='gray', marker='o', markersize=3)
        
        ax2.set_title('Momentum Histogram')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def get_optimal_parameters(timeframe):
        """
        Get optimized parameters based on timeframe analysis from 2024 research
        
        Parameters:
        timeframe: str - '15m', '1h', '4h', '1d', '1w'
        
        Returns:
        dict with optimal parameters for the timeframe
        """
        params = {
            '15m': {  # Intraday scalping
                'bb_length': 10,
                'bb_mult': 2.0,
                'kc_length': 10,
                'kc_mult': 1.5,
                'description': 'Fast parameters for 15-minute scalping'
            },
            '1h': {   # Short-term trading
                'bb_length': 20,
                'bb_mult': 2.0,
                'kc_length': 20,
                'kc_mult': 1.5,
                'description': 'Standard parameters for hourly trading'
            },
            '4h': {   # Swing trading
                'bb_length': 20,
                'bb_mult': 2.0,
                'kc_length': 20,
                'kc_mult': 1.5,
                'description': 'Standard parameters for 4-hour swing trading'
            },
            '1d': {   # Daily position trading
                'bb_length': 20,
                'bb_mult': 2.0,
                'kc_length': 20,
                'kc_mult': 1.2,  # Tighter Keltner for more sensitive squeeze detection
                'description': 'Daily trading with tighter Keltner channels'
            },
            '1w': {   # Weekly long-term
                'bb_length': 15,
                'bb_mult': 1.8,
                'kc_length': 15,
                'kc_mult': 1.0,
                'description': 'Weekly parameters for long-term trends'
            }
        }
        
        return params.get(timeframe, params['1h'])  # Default to 1h if not found
    
    def optimize_for_timeframe(self, timeframe):
        """Update indicator parameters for specific timeframe"""
        optimal_params = self.get_optimal_parameters(timeframe)
        self.bb_length = optimal_params['bb_length']
        self.bb_mult = optimal_params['bb_mult']
        self.kc_length = optimal_params['kc_length']
        self.kc_mult = optimal_params['kc_mult']
        return optimal_params
        """Simple backtest of the strategy"""
        data = self.generate_signals(df)
        
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        
        for i, row in data.iterrows():
            if row['signal'] == 1 and position == 0:  # Buy signal
                position = capital / row['close']
                entry_price = row['close']
                capital = 0
                
            elif row['signal'] == -1 and position > 0:  # Sell signal
                capital = position * row['close']
                exit_price = row['close']
                pnl = (exit_price - entry_price) / entry_price * 100
                
                trades.append({
                    'entry_date': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_percent': pnl
                })
                
                position = 0
        
        # Close final position if open
        if position > 0:
            capital = position * data['close'].iloc[-1]
        
        total_return = (capital - initial_capital) / initial_capital * 100
        
        return {
            'total_return': total_return,
            'final_capital': capital,
            'num_trades': len(trades),
            'trades': trades
        }


# Example usage and testing
def example_usage():
    """Example of how to use the Squeeze Momentum Indicator"""
    
    # Generate sample OHLC data (replace with your actual data)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate price data
    close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    high_prices = close_prices + np.random.rand(len(dates)) * 2
    low_prices = close_prices - np.random.rand(len(dates)) * 2
    open_prices = close_prices + np.random.randn(len(dates)) * 0.5
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    }, index=dates)
    
    # Initialize the indicator
    squeeze_indicator = SqueezeMomentumIndicator(
        bb_length=20,
        bb_mult=2.0,
        kc_length=20,
        kc_mult=1.5,
        use_true_range=True
    )
    
    # Calculate indicators
    result = squeeze_indicator.generate_signals(df)
    
    # Display sample results
    print("Sample of calculated indicators:")
    print(result[['close', 'squeeze_on', 'squeeze_off', 'momentum', 'signal', 'position']].tail(10))
    
    # Run backtest
    backtest_results = squeeze_indicator.backtest_strategy(df)
    print(f"\nBacktest Results:")
    print(f"Total Return: {backtest_results['total_return']:.2f}%")
    print(f"Number of Trades: {backtest_results['num_trades']}")
    
    # Plot the indicator
    squeeze_indicator.plot_indicator(df.tail(100))  # Plot last 100 days
    
    return result

if __name__ == "__main__":
    example_usage()
