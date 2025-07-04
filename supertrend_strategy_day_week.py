import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Literal
from enum import Enum

class SourceType(Enum):
    """Source calculation types"""
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    HL2 = "hl2"  # (H + L) / 2
    HLC3 = "hlc3"  # (H + L + C) / 3
    OHLC4 = "ohlc4"  # (O + H + L + C) / 4
    HLCC4 = "hlcc4"  # (H + L + C + C) / 4

class SupertrendStrategy:
    def __init__(self, 
                 atr_period: int = 10,
                 multiplier: float = 3.0,
                 change_atr_method: bool = True,
                 source_type: SourceType = SourceType.HL2,
                 weekly_atr_period: int = 10,
                 weekly_multiplier: float = 3.0,
                 weekly_source_type: SourceType = SourceType.HL2,
                 use_dual_timeframe: bool = True):
        """
        Initialize Enhanced Supertrend Strategy with dual timeframe support
        
        Parameters:
        atr_period: Period for daily ATR calculation
        multiplier: ATR multiplier for daily bands
        change_atr_method: True for exponential ATR, False for simple moving average
        source_type: Source calculation method for daily timeframe
        weekly_atr_period: Period for weekly ATR calculation
        weekly_multiplier: ATR multiplier for weekly bands
        weekly_source_type: Source calculation method for weekly timeframe
        use_dual_timeframe: Whether to use both daily and weekly Supertrend for signals
        """
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.change_atr_method = change_atr_method
        self.source_type = source_type
        self.weekly_atr_period = weekly_atr_period
        self.weekly_multiplier = weekly_multiplier
        self.weekly_source_type = weekly_source_type
        self.use_dual_timeframe = use_dual_timeframe
    
    def calculate_source(self, df: pd.DataFrame, source_type: SourceType) -> pd.Series:
        """Calculate source price based on selected method"""
        if source_type == SourceType.OPEN:
            return df['Open']
        elif source_type == SourceType.HIGH:
            return df['High']
        elif source_type == SourceType.LOW:
            return df['Low']
        elif source_type == SourceType.CLOSE:
            return df['Close']
        elif source_type == SourceType.HL2:
            return (df['High'] + df['Low']) / 2
        elif source_type == SourceType.HLC3:
            return (df['High'] + df['Low'] + df['Close']) / 3
        elif source_type == SourceType.OHLC4:
            return (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        elif source_type == SourceType.HLCC4:
            return (df['High'] + df['Low'] + df['Close'] + df['Close']) / 4
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    
    def calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
        low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        return true_range
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        tr = self.calculate_true_range(df)
        
        if self.change_atr_method:
            # Exponential ATR (similar to Pine Script atr() function)
            atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        else:
            # Simple Moving Average ATR
            atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def resample_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert daily OHLC data to weekly OHLC data"""
        weekly_ohlc = df.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()
        
        return weekly_ohlc
    
    def calculate_supertrend_single(self, df: pd.DataFrame, atr_period: int, 
                                  multiplier: float, source_type: SourceType) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Supertrend for a single timeframe
        
        Returns:
        Tuple of (supertrend_values, trend_direction, atr_values)
        """
        # Calculate source
        src = self.calculate_source(df, source_type)
        
        # Calculate ATR for this timeframe
        tr = self.calculate_true_range(df)
        if self.change_atr_method:
            atr = tr.ewm(span=atr_period, adjust=False).mean()
        else:
            atr = tr.rolling(window=atr_period).mean()
        
        # Initialize arrays
        n = len(df)
        up = np.zeros(n)
        dn = np.zeros(n)
        trend = np.zeros(n)
        supertrend = np.zeros(n)
        
        # Handle NaN values in early periods
        first_valid_idx = atr.first_valid_index()
        if first_valid_idx is None:
            return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index), atr
        
        start_idx = df.index.get_loc(first_valid_idx)
        
        # Calculate initial values
        if start_idx < n:
            up[start_idx] = src.iloc[start_idx] - (multiplier * atr.iloc[start_idx])
            dn[start_idx] = src.iloc[start_idx] + (multiplier * atr.iloc[start_idx])
            trend[start_idx] = 1
            supertrend[start_idx] = up[start_idx]
        
        # Calculate Supertrend
        for i in range(start_idx + 1, n):
            if pd.isna(atr.iloc[i]) or pd.isna(src.iloc[i]):
                up[i] = up[i-1] if i > 0 else 0
                dn[i] = dn[i-1] if i > 0 else 0
                trend[i] = trend[i-1] if i > 0 else 1
                supertrend[i] = supertrend[i-1] if i > 0 else 0
                continue
            
            # Calculate basic upper and lower bands
            up_basic = src.iloc[i] - (multiplier * atr.iloc[i])
            dn_basic = src.iloc[i] + (multiplier * atr.iloc[i])
            
            # Calculate final upper band
            up[i] = up_basic if df['Close'].iloc[i-1] <= up[i-1] else max(up_basic, up[i-1])
            
            # Calculate final lower band
            dn[i] = dn_basic if df['Close'].iloc[i-1] >= dn[i-1] else min(dn_basic, dn[i-1])
            
            # Determine trend
            if trend[i-1] == -1 and df['Close'].iloc[i] > dn[i-1]:
                trend[i] = 1
            elif trend[i-1] == 1 and df['Close'].iloc[i] < up[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
            
            # Set Supertrend value
            supertrend[i] = up[i] if trend[i] == 1 else dn[i]
        
        # Convert to pandas Series
        supertrend_series = pd.Series(supertrend, index=df.index)
        trend_series = pd.Series(trend, index=df.index)
        
        # Set initial NaN values
        supertrend_series.iloc[:start_idx] = np.nan
        trend_series.iloc[:start_idx] = np.nan
        
        return supertrend_series, trend_series, atr
    
    def calculate_dual_timeframe_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate both daily and weekly Supertrend indicators
        """
        result_df = df.copy()
        
        # Calculate daily Supertrend
        daily_supertrend, daily_trend, daily_atr = self.calculate_supertrend_single(
            df, self.atr_period, self.multiplier, self.source_type
        )
        
        # Calculate weekly Supertrend
        weekly_ohlc = self.resample_to_weekly(df)
        weekly_supertrend, weekly_trend, weekly_atr = self.calculate_supertrend_single(
            weekly_ohlc, self.weekly_atr_period, self.weekly_multiplier, self.weekly_source_type
        )
        
        # Forward fill weekly data to daily frequency
        weekly_trend_daily = weekly_trend.reindex(df.index, method='ffill')
        weekly_supertrend_daily = weekly_supertrend.reindex(df.index, method='ffill')
        
        # Add all calculated values to result DataFrame
        result_df['Daily_ATR'] = daily_atr
        result_df['Daily_Supertrend'] = daily_supertrend
        result_df['Daily_Trend'] = daily_trend
        result_df['Weekly_Supertrend'] = weekly_supertrend_daily
        result_df['Weekly_Trend'] = weekly_trend_daily
        
        # Calculate combined trend (both timeframes must agree)
        if self.use_dual_timeframe:
            result_df['Combined_Trend'] = np.where(
                (daily_trend == 1) & (weekly_trend_daily == 1), 1,
                np.where((daily_trend == -1) & (weekly_trend_daily == -1), -1, 0)
            )
        else:
            result_df['Combined_Trend'] = daily_trend
        
        return result_df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on combined Supertrend
        """
        result_df = df.copy()
        
        if self.use_dual_timeframe:
            # Generate signals only when both timeframes agree
            result_df['Buy_Signal'] = (
                (result_df['Combined_Trend'] == 1) & 
                (result_df['Combined_Trend'].shift(1) != 1)
            )
            result_df['Sell_Signal'] = (
                (result_df['Combined_Trend'] == -1) & 
                (result_df['Combined_Trend'].shift(1) != -1)
            )
            
            # Use daily Supertrend for signal prices
            result_df['Buy_Price'] = np.where(result_df['Buy_Signal'], result_df['Daily_Supertrend'], np.nan)
            result_df['Sell_Price'] = np.where(result_df['Sell_Signal'], result_df['Daily_Supertrend'], np.nan)
        else:
            # Use only daily signals
            result_df['Buy_Signal'] = (result_df['Daily_Trend'] == 1) & (result_df['Daily_Trend'].shift(1) == -1)
            result_df['Sell_Signal'] = (result_df['Daily_Trend'] == -1) & (result_df['Daily_Trend'].shift(1) == 1)
            result_df['Buy_Price'] = np.where(result_df['Buy_Signal'], result_df['Daily_Supertrend'], np.nan)
            result_df['Sell_Price'] = np.where(result_df['Sell_Signal'], result_df['Daily_Supertrend'], np.nan)
        
        return result_df
    
    def run_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete enhanced Supertrend strategy
        """
        # Calculate dual timeframe Supertrend
        df_with_supertrend = self.calculate_dual_timeframe_supertrend(df)
        
        # Generate signals
        df_with_signals = self.generate_signals(df_with_supertrend)
        
        return df_with_signals
    
    def plot_supertrend(self, df: pd.DataFrame, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot enhanced Supertrend with both daily and weekly trends
        """
        # Filter data by date range if provided
        plot_df = df.copy()
        if start_date:
            plot_df = plot_df[plot_df.index >= start_date]
        if end_date:
            plot_df = plot_df[plot_df.index <= end_date]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Main price chart
        ax1.plot(plot_df.index, plot_df['Close'], label='Close Price', linewidth=1, color='black', alpha=0.7)
        
        # Plot Daily Supertrend
        daily_uptrend_mask = plot_df['Daily_Trend'] == 1
        daily_downtrend_mask = plot_df['Daily_Trend'] == -1
        
        ax1.plot(plot_df.index[daily_uptrend_mask], plot_df['Daily_Supertrend'][daily_uptrend_mask], 
                label='Daily Supertrend (Up)', color='lightgreen', linewidth=1.5, alpha=0.8)
        ax1.plot(plot_df.index[daily_downtrend_mask], plot_df['Daily_Supertrend'][daily_downtrend_mask], 
                label='Daily Supertrend (Down)', color='lightcoral', linewidth=1.5, alpha=0.8)
        
        # Plot Weekly Supertrend
        weekly_uptrend_mask = plot_df['Weekly_Trend'] == 1
        weekly_downtrend_mask = plot_df['Weekly_Trend'] == -1
        
        ax1.plot(plot_df.index[weekly_uptrend_mask], plot_df['Weekly_Supertrend'][weekly_uptrend_mask], 
                label='Weekly Supertrend (Up)', color='green', linewidth=3, alpha=0.6)
        ax1.plot(plot_df.index[weekly_downtrend_mask], plot_df['Weekly_Supertrend'][weekly_downtrend_mask], 
                label='Weekly Supertrend (Down)', color='red', linewidth=3, alpha=0.6)
        
        # Plot signals
        buy_signals = plot_df[plot_df['Buy_Signal']]
        sell_signals = plot_df[plot_df['Sell_Signal']]
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['Buy_Price'], 
                      color='green', marker='^', s=150, label='Buy Signal', zorder=5, edgecolors='darkgreen')
        
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['Sell_Price'], 
                      color='red', marker='v', s=150, label='Sell Signal', zorder=5, edgecolors='darkred')
        
        ax1.set_title(f'Enhanced Dual Timeframe Supertrend Strategy\n' +
                     f'Daily: {self.source_type.value.upper()} (Period: {self.atr_period}, Mult: {self.multiplier}) | ' +
                     f'Weekly: {self.weekly_source_type.value.upper()} (Period: {self.weekly_atr_period}, Mult: {self.weekly_multiplier})')
        ax1.set_ylabel('Price')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Trend direction subplot
        if self.use_dual_timeframe:
            combined_colors = ['red' if x == -1 else 'green' if x == 1 else 'gray' for x in plot_df['Combined_Trend']]
            ax2.bar(plot_df.index, plot_df['Combined_Trend'], color=combined_colors, alpha=0.7, width=1)
            ax2.set_title('Combined Trend Direction (Both Timeframes Must Agree)')
            ax2.set_ylim(-1.5, 1.5)
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(['Sell', 'Neutral', 'Buy'])
        else:
            daily_colors = ['red' if x == -1 else 'green' for x in plot_df['Daily_Trend']]
            ax2.bar(plot_df.index, plot_df['Daily_Trend'], color=daily_colors, alpha=0.7, width=1)
            ax2.set_title('Daily Trend Direction')
            ax2.set_ylim(-1.5, 1.5)
            ax2.set_yticks([-1, 1])
            ax2.set_yticklabels(['Sell', 'Buy'])
        
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_strategy_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of the strategy performance"""
        buy_count = df['Buy_Signal'].sum()
        sell_count = df['Sell_Signal'].sum()
        
        # Calculate trend agreement percentage
        if self.use_dual_timeframe:
            total_days = len(df.dropna())
            agreement_days = len(df[df['Combined_Trend'] != 0])
            agreement_pct = (agreement_days / total_days) * 100 if total_days > 0 else 0
        else:
            agreement_pct = 100  # Single timeframe always "agrees" with itself
        
        return {
            'total_buy_signals': int(buy_count),
            'total_sell_signals': int(sell_count),
            'dual_timeframe_enabled': self.use_dual_timeframe,
            'timeframe_agreement_percentage': round(agreement_pct, 2),
            'daily_source': self.source_type.value,
            'weekly_source': self.weekly_source_type.value,
            'daily_atr_period': self.atr_period,
            'weekly_atr_period': self.weekly_atr_period,
            'daily_multiplier': self.multiplier,
            'weekly_multiplier': self.weekly_multiplier
        }

# Example usage and testing
def create_sample_data(days: int = 200):
    """Create sample OHLC data for testing"""
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    np.random.seed(42)
    
    # Generate more realistic trending data
    trend = np.cumsum(np.random.randn(days) * 0.3)
    base_price = 100
    close_prices = base_price + trend + np.random.randn(days) * 2
    
    # Generate OHLC from close prices
    high_prices = close_prices + np.random.uniform(0.5, 3.0, days)
    low_prices = close_prices - np.random.uniform(0.5, 3.0, days)
    open_prices = np.roll(close_prices, 1) + np.random.uniform(-1.0, 1.0, days)
    open_prices[0] = close_prices[0]
    
    # Ensure OHLC relationships are maintained
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices
    }, index=dates)
    
    return df

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = create_sample_data(200)
    
    # Initialize enhanced strategy
    strategy = SupertrendStrategy(
        atr_period=10,
        multiplier=3.0,
        source_type=SourceType.HL2,  # Daily timeframe source
        weekly_atr_period=10,
        weekly_multiplier=3.0,
        weekly_source_type=SourceType.HLCC4,  # Weekly timeframe source
        use_dual_timeframe=True,
        change_atr_method=True
    )
    
    # Run strategy
    results = strategy.run_strategy(sample_data)
    
    # Display results
    print("Enhanced Dual Timeframe Supertrend Strategy Results:")
    print("\nRecent data with all indicators:")
    columns_to_show = ['Close', 'Daily_Supertrend', 'Daily_Trend', 'Weekly_Trend', 
                      'Combined_Trend', 'Buy_Signal', 'Sell_Signal']
    print(results[columns_to_show].tail(10))
    
    # Get strategy summary
    summary = strategy.get_strategy_summary(results)
    print(f"\nStrategy Summary:")
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Plot results
    strategy.plot_supertrend(results)
    
    # Example of different source configurations
    print("\n" + "="*50)
    print("TESTING DIFFERENT SOURCE TYPES")
    print("="*50)
    
    source_types_to_test = [SourceType.CLOSE, SourceType.OHLC4, SourceType.HLC3]
    
    for source_type in source_types_to_test:
        print(f"\nTesting with {source_type.value.upper()} source:")
        test_strategy = SupertrendStrategy(
            source_type=source_type,
            weekly_source_type=source_type,
            use_dual_timeframe=True
        )
        test_results = test_strategy.run_strategy(sample_data)
        test_summary = test_strategy.get_strategy_summary(test_results)
        print(f"Buy Signals: {test_summary['total_buy_signals']}")
        print(f"Sell Signals: {test_summary['total_sell_signals']}")
        print(f"Agreement: {test_summary['timeframe_agreement_percentage']}%")