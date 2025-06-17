import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SqueezeMomentumIndicator:
    """
    Squeeze Momentum Indicator (TTM Squeeze) by LazyBear
    
    This indicator identifies periods of low volatility (squeeze) followed by 
    high momentum moves. It combines Bollinger Bands and Keltner Channels.
    """
    
    def __init__(self, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5, use_true_range=True):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        self.use_true_range = use_true_range
    
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
        Generate trading signals based on Squeeze Momentum
        
        Buy signals:
        - Squeeze is released (squeeze_off)
        - Momentum turns positive and increasing
        
        Sell signals:
        - Momentum turns negative and decreasing
        - Or stop loss conditions
        """
        data = self.calculate_indicators(df)
        
        # Initialize signals
        data['signal'] = 0
        data['position'] = 0
        
        # Buy conditions
        buy_condition = (
            (data['squeeze_off'] | data['squeeze_off'].shift(1)) &
            (data['momentum'] > 0) &
            (data['momentum_increasing'])
        ) | (
            # Alternative: momentum crosses above zero during or after squeeze
            (data['momentum'] > 0) &
            (data['momentum'].shift(1) <= 0) &
            (data['squeeze_on'].rolling(5).sum() > 0)  # Recent squeeze
        )
        
        # Sell conditions
        sell_condition = (
            (data['momentum'] < 0) &
            (~data['momentum_increasing'])
        ) | (
            # Momentum crosses below zero
            (data['momentum'] < 0) &
            (data['momentum'].shift(1) >= 0)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        # Calculate positions
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
    
    def backtest_strategy(self, df, initial_capital=10000):
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
