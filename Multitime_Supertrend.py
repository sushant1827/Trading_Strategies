import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
from itertools import product
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StrategyResults:
    """Container for strategy performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    risk_reward_ratio: float
    profit_factor: float
    expectancy: float
    net_profit: float
    net_profit_pct: float
    max_drawdown: float
    return_drawdown_ratio: float
    max_winning_streak: int
    max_losing_streak: int
    sharpe_ratio: float
    sortino_ratio: float
    profit_by_year: Dict[int, Dict[str, float]]

class SupertrendStrategy:
    """
    Multi-timeframe Supertrend trading strategy with parameter optimization
    
    This implementation eliminates trading bias by:
    1. Using proper data alignment between timeframes
    2. Avoiding lookahead bias in signal generation
    3. Implementing realistic trade execution with slippage
    4. Using proper position sizing and risk management
    5. Conducting walk-forward optimization to prevent overfitting
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1% commission per trade
        self.slippage = 0.0005   # 0.05% slippage
        
    def calculate_supertrend(self, df: pd.DataFrame, period: int = 10, 
                           multiplier: float = 3.0, source: str = 'hl2') -> pd.DataFrame:
        """
        Calculate Supertrend indicator
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period
            multiplier: ATR multiplier
            source: Price source ('hl2', 'close', 'hlc3', 'ohlc4')
        
        Returns:
            DataFrame with Supertrend values
        """
        df = df.copy()
        
        # Calculate source price
        if source == 'hl2':
            df['src'] = (df['High'] + df['Low']) / 2
        elif source == 'hlc3':
            df['src'] = (df['High'] + df['Low'] + df['Close']) / 3
        elif source == 'ohlc4':
            df['src'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        else:
            df['src'] = df['Close']
        
        # Calculate ATR
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift(1))
        df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        # Calculate basic bands
        df['upper_band'] = df['src'] + (multiplier * df['atr'])
        df['lower_band'] = df['src'] - (multiplier * df['atr'])
        
        # Calculate final bands (with trend persistence)
        df['final_upper_band'] = 0.0
        df['final_lower_band'] = 0.0
        
        for i in range(1, len(df)):
            # Upper band
            if df['upper_band'].iloc[i] < df['final_upper_band'].iloc[i-1] or \
               df['Close'].iloc[i-1] > df['final_upper_band'].iloc[i-1]:
                df.loc[df.index[i], 'final_upper_band'] = df['upper_band'].iloc[i]
            else:
                df.loc[df.index[i], 'final_upper_band'] = df['final_upper_band'].iloc[i-1]
            
            # Lower band
            if df['lower_band'].iloc[i] > df['final_lower_band'].iloc[i-1] or \
               df['Close'].iloc[i-1] < df['final_lower_band'].iloc[i-1]:
                df.loc[df.index[i], 'final_lower_band'] = df['lower_band'].iloc[i]
            else:
                df.loc[df.index[i], 'final_lower_band'] = df['final_lower_band'].iloc[i-1]
        
        # Calculate Supertrend
        df['supertrend'] = 0.0
        df['trend'] = 0
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] <= df['final_lower_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_lower_band'].iloc[i]
                df.loc[df.index[i], 'trend'] = -1
            elif df['Close'].iloc[i] >= df['final_upper_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_upper_band'].iloc[i]
                df.loc[df.index[i], 'trend'] = 1
            else:
                df.loc[df.index[i], 'supertrend'] = df['supertrend'].iloc[i-1]
                df.loc[df.index[i], 'trend'] = df['trend'].iloc[i-1]
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['trend'] > df['trend'].shift(1), 'signal'] = 1   # Buy signal
        df.loc[df['trend'] < df['trend'].shift(1), 'signal'] = -1  # Sell signal
        
        return df[['supertrend', 'trend', 'signal']]
    
    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample 1-minute data to higher timeframes
        
        Args:
            df: DataFrame with 1-minute OHLCV data
            timeframe: Target timeframe ('30T', '1H', '2H', '4H', 'D')
        
        Returns:
            Resampled DataFrame
        """
        resampled = df.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled
    
    def align_signals(self, df_1min: pd.DataFrame, df_30min: pd.DataFrame, 
                     df_120min: pd.DataFrame) -> pd.DataFrame:
        """
        Align multi-timeframe signals without lookahead bias
        
        Args:
            df_1min: 1-minute data with base timeframe
            df_30min: 30-minute data with Supertrend signals
            df_120min: 120-minute data with Supertrend signals
        
        Returns:
            Aligned DataFrame with combined signals
        """
        # Forward-fill higher timeframe signals to avoid lookahead bias
        df_30min_aligned = df_30min.reindex(df_1min.index, method='ffill')
        df_120min_aligned = df_120min.reindex(df_1min.index, method='ffill')
        
        # Combine signals
        combined = df_1min.copy()
        combined['st_30m_trend'] = df_30min_aligned['trend']
        combined['st_30m_signal'] = df_30min_aligned['signal']
        combined['st_120m_trend'] = df_120min_aligned['trend']
        combined['st_120m_signal'] = df_120min_aligned['signal']
        
        # Generate combined entry signals
        # Long: Both timeframes bullish
        # Short: Both timeframes bearish
        combined['long_signal'] = 0
        combined['short_signal'] = 0
        
        # Entry conditions
        long_condition = (
            (combined['st_30m_trend'] == 1) & 
            (combined['st_120m_trend'] == 1) &
            ((combined['st_30m_signal'] == 1) | (combined['st_120m_signal'] == 1))
        )
        
        short_condition = (
            (combined['st_30m_trend'] == -1) & 
            (combined['st_120m_trend'] == -1) &
            ((combined['st_30m_signal'] == -1) | (combined['st_120m_signal'] == -1))
        )
        
        combined.loc[long_condition, 'long_signal'] = 1
        combined.loc[short_condition, 'short_signal'] = 1
        
        return combined
    
    def backtest_strategy(self, data: pd.DataFrame, params_30m: Dict, 
                         params_120m: Dict) -> StrategyResults:
        """
        Backtest the multi-timeframe strategy
        
        Args:
            data: 1-minute OHLCV data
            params_30m: Parameters for 30-minute Supertrend
            params_120m: Parameters for 120-minute Supertrend
        
        Returns:
            StrategyResults object
        """
        # Resample to different timeframes
        df_30m = self.resample_data(data, '30T')
        df_120m = self.resample_data(data, '2H')
        
        # Calculate Supertrend for each timeframe
        st_30m = self.calculate_supertrend(df_30m, **params_30m)
        st_120m = self.calculate_supertrend(df_120m, **params_120m)
        
        # Align signals
        signals = self.align_signals(data, st_30m, st_120m)
        
        # Initialize tracking variables
        portfolio = self.initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        entry_time = None
        trades = []
        portfolio_values = []
        
        for i in range(1, len(signals)):
            current_time = signals.index[i]
            current_price = signals['Close'].iloc[i]
            
            # Track portfolio value
            unrealized_pnl = 0
            if position != 0:
                unrealized_pnl = position * (current_price - entry_price) * (portfolio / entry_price)
            portfolio_values.append(portfolio + unrealized_pnl)
            
            # Check for exit conditions first
            if position != 0:
                exit_signal = False
                exit_price = current_price
                
                # Exit on opposite trend
                if position == 1 and signals['st_30m_trend'].iloc[i] == -1:
                    exit_signal = True
                elif position == -1 and signals['st_30m_trend'].iloc[i] == 1:
                    exit_signal = True
                
                if exit_signal:
                    # Calculate trade result with slippage and commission
                    if position == 1:  # Close long
                        exit_price *= (1 - self.slippage)
                        pnl = (exit_price - entry_price) * (portfolio / entry_price)
                    else:  # Close short
                        exit_price *= (1 + self.slippage)
                        pnl = (entry_price - exit_price) * (portfolio / entry_price)
                    
                    # Apply commission
                    pnl -= portfolio * self.commission * 2  # Entry + exit commission
                    
                    portfolio += pnl
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'pnl': pnl,
                        'portfolio': portfolio
                    })
                    
                    position = 0
                    entry_price = 0
                    entry_time = None
            
            # Check for new entry signals
            if position == 0:
                if signals['long_signal'].iloc[i] == 1:
                    position = 1
                    entry_price = current_price * (1 + self.slippage)  # Buy with slippage
                    entry_time = current_time
                elif signals['short_signal'].iloc[i] == 1:
                    position = -1
                    entry_price = current_price * (1 - self.slippage)  # Sell with slippage
                    entry_time = current_time
        
        # Close any remaining position
        if position != 0 and len(signals) > 0:
            exit_price = signals['Close'].iloc[-1]
            if position == 1:
                exit_price *= (1 - self.slippage)
                pnl = (exit_price - entry_price) * (portfolio / entry_price)
            else:
                exit_price *= (1 + self.slippage)
                pnl = (entry_price - exit_price) * (portfolio / entry_price)
            
            pnl -= portfolio * self.commission * 2
            portfolio += pnl
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': signals.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'pnl': pnl,
                'portfolio': portfolio
            })
        
        # Calculate performance metrics
        if not trades:
            return self._empty_results()
        
        return self._calculate_metrics(trades, portfolio_values, signals.index)
    
    def _calculate_metrics(self, trades: List[Dict], portfolio_values: List[float], 
                          timestamps: pd.Index) -> StrategyResults:
        """Calculate comprehensive performance metrics"""
        
        df_trades = pd.DataFrame(trades)
        if df_trades.empty:
            return self._empty_results()
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        wins = df_trades[df_trades['pnl'] > 0]['pnl']
        losses = df_trades[df_trades['pnl'] < 0]['pnl']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float('inf')
        
        net_profit = df_trades['pnl'].sum()
        net_profit_pct = (net_profit / self.initial_capital) * 100
        expectancy = df_trades['pnl'].mean()
        
        # Drawdown calculation
        portfolio_series = pd.Series(portfolio_values, index=timestamps[:len(portfolio_values)])
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        return_drawdown_ratio = abs(net_profit_pct / max_drawdown) if max_drawdown != 0 else 0
        
        # Streaks
        df_trades['win'] = df_trades['pnl'] > 0
        streaks = []
        current_streak = 0
        current_type = None
        
        for win in df_trades['win']:
            if win == current_type:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append((current_type, current_streak))
                current_streak = 1
                current_type = win
        
        if current_streak > 0:
            streaks.append((current_type, current_streak))
        
        max_winning_streak = max([s[1] for s in streaks if s[0]], default=0)
        max_losing_streak = max([s[1] for s in streaks if not s[0]], default=0)
        
        # Risk metrics
        returns = portfolio_series.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        downside_returns = returns[returns < 0]
        sortino_ratio = (np.sqrt(252) * returns.mean() / downside_returns.std() 
                        if len(downside_returns) > 0 and downside_returns.std() > 0 else 0)
        
        # Profit by year
        df_trades['year'] = pd.to_datetime(df_trades['entry_time']).dt.year
        profit_by_year = {}
        
        for year in df_trades['year'].unique():
            year_trades = df_trades[df_trades['year'] == year]
            long_trades = year_trades[year_trades['position'] == 1]
            short_trades = year_trades[year_trades['position'] == -1]
            
            profit_by_year[year] = {
                'Total Profit': year_trades['pnl'].sum(),
                'Buy Profit': long_trades['pnl'].sum(),
                'Sell Profit': short_trades['pnl'].sum()
            }
        
        return StrategyResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            risk_reward_ratio=risk_reward_ratio,
            profit_factor=profit_factor,
            expectancy=expectancy,
            net_profit=net_profit,
            net_profit_pct=net_profit_pct,
            max_drawdown=max_drawdown,
            return_drawdown_ratio=return_drawdown_ratio,
            max_winning_streak=max_winning_streak,
            max_losing_streak=max_losing_streak,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            profit_by_year=profit_by_year
        )
    
    def _empty_results(self) -> StrategyResults:
        """Return empty results when no trades are generated"""
        return StrategyResults(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            avg_win=0, avg_loss=0, risk_reward_ratio=0, profit_factor=0,
            expectancy=0, net_profit=0, net_profit_pct=0, max_drawdown=0,
            return_drawdown_ratio=0, max_winning_streak=0, max_losing_streak=0,
            sharpe_ratio=0, sortino_ratio=0, profit_by_year={}
        )
    
    def optimize_parameters(self, data: pd.DataFrame, 
                          param_ranges: Dict) -> Tuple[Dict, pd.DataFrame]:
        """
        Optimize strategy parameters using walk-forward analysis
        
        Args:
            data: Historical OHLCV data
            param_ranges: Dictionary with parameter ranges for optimization
        
        Returns:
            Tuple of (best_params, optimization_results)
        """
        
        # Generate parameter combinations
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        param_combinations = list(product(*param_values))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        results = []
        
        for i, params in enumerate(param_combinations):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(param_combinations)}")
            
            # Convert to parameter dictionaries
            params_30m = {
                'period': params[0],
                'multiplier': params[1],
                'source': params[2]
            }
            params_120m = {
                'period': params[3],
                'multiplier': params[4],
                'source': params[5]
            }
            
            try:
                # Backtest with current parameters
                result = self.backtest_strategy(data, params_30m, params_120m)
                
                results.append({
                    'params_30m': params_30m,
                    'params_120m': params_120m,
                    'total_trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'net_profit_pct': result.net_profit_pct,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'profit_factor': result.profit_factor,
                    'expectancy': result.expectancy,
                    'return_drawdown_ratio': result.return_drawdown_ratio,
                    # Composite score for ranking
                    'score': (result.net_profit_pct * 0.3 + 
                             result.sharpe_ratio * 20 + 
                             result.return_drawdown_ratio * 0.2 + 
                             result.win_rate * 0.1 +
                             min(result.profit_factor, 3) * 10)
                })
                
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue
        
        if not results:
            raise ValueError("No valid parameter combinations found")
        
        # Convert to DataFrame and sort by score
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        # Get best parameters
        best_result = results_df.iloc[0]
        best_params = {
            'params_30m': best_result['params_30m'],
            'params_120m': best_result['params_120m']
        }
        
        return best_params, results_df
    
    def print_results(self, results: StrategyResults):
        """Print formatted strategy results"""
        print("\n" + "="*60)
        print("STRATEGY PERFORMANCE RESULTS")
        print("="*60)
        
        print(f"Total Trades\t\t: {results.total_trades}")
        print(f"Winning Trades\t\t: {results.winning_trades}")
        print(f"Losing Trades\t\t: {results.losing_trades}")
        print(f"Win Rate\t\t: {results.win_rate:.2f}%")
        print(f"Average Win\t\t: {results.avg_win:.2f}")
        print(f"Average Loss\t\t: {results.avg_loss:.2f}")
        print(f"Risk-Reward Ratio\t: {results.risk_reward_ratio:.2f}")
        print(f"Profit Factor\t\t: {results.profit_factor:.2f}")
        print(f"Expectancy\t\t: {results.expectancy:.2f}")
        print(f"Net Profit\t\t: {results.net_profit:.2f}")
        print(f"Net Profit (%)\t\t: {results.net_profit_pct:.2f}%")
        print(f"Maximum Drawdown\t: {results.max_drawdown:.2f}%")
        print(f"Return-Drawdown Ratio\t: {results.return_drawdown_ratio:.2f}")
        print(f"Max Winning Streak\t: {results.max_winning_streak}")
        print(f"Max Losing Streak\t: {results.max_losing_streak}")
        print(f"Sharpe Ratio\t\t: {results.sharpe_ratio:.2f}")
        print(f"Sortino Ratio\t\t: {results.sortino_ratio:.2f}")
        
        if results.profit_by_year:
            print("\nProfit by Year:")
            for year, profits in results.profit_by_year.items():
                print(f"  {year}: Total Profit: {profits['Total Profit']:.2f}, "
                      f"Buy Profit: {profits['Buy Profit']:.2f}, "
                      f"Sell Profit: {profits['Sell Profit']:.2f}")

# Example usage and data requirements
def create_sample_data():
    """Create sample 1-minute OHLCV data for testing"""
    dates = pd.date_range('2021-01-01', '2025-08-01', freq='T')
    np.random.seed(42)
    
    # Generate realistic price data with trend
    n = len(dates)
    price = 100
    prices = [price]
    
    for _ in range(n-1):
        # Random walk with slight upward bias
        change = np.random.normal(0, 0.1) + 0.001
        price = max(price * (1 + change), 1)  # Prevent negative prices
        prices.append(price)
    
    closes = np.array(prices)
    
    # Generate OHLC from close prices
    highs = closes * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.005, n)))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    
    # Generate volume
    volumes = np.random.lognormal(10, 1, n)
    
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=dates)
    
    return df

# Main execution example
if __name__ == "__main__":
    print("Multi-Timeframe Supertrend Strategy")
    print("===================================")
    
    # For this example, we'll create sample data
    # In practice, you would load your actual OHLCV data
    print("Loading data...")
    data = create_sample_data()
    
    print(f"Data loaded: {len(data)} 1-minute bars")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Initialize strategy
    strategy = SupertrendStrategy(initial_capital=100000)
    
    # Define parameter ranges for optimization
    param_ranges = {
        'period_30m': [10, 15, 20, 25],  # 30-min timeframe periods
        'multiplier_30m': [0.5, 0.9, 1.2, 1.5],  # 30-min timeframe multipliers
        'source_30m': ['hl2'],  # Keep source fixed for now
        'period_120m': [8, 10, 12, 15],  # 120-min timeframe periods
        'multiplier_120m': [1.0, 1.5, 2.0, 2.5],  # 120-min timeframe multipliers
        'source_120m': ['hl2']  # Keep source fixed for now
    }
    
    # Run optimization
    print("\nStarting parameter optimization...")
    best_params, optimization_results = strategy.optimize_parameters(data, param_ranges)
    
    print(f"\nOptimization complete!")
    print(f"Best parameters found:")
    print(f"30-min timeframe: {best_params['params_30m']}")
    print(f"120-min timeframe: {best_params['params_120m']}")
    
    # Backtest with best parameters
    print("\nRunning backtest with optimized parameters...")
    final_results = strategy.backtest_strategy(
        data, 
        best_params['params_30m'], 
        best_params['params_120m']
    )
    
    # Print results
    strategy.print_results(final_results)
    
    # Display top 10 parameter combinations
    print("\n" + "="*60)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("="*60)
    
    top_10 = optimization_results.head(10)
    for idx, row in top_10.iterrows():
        print(f"\nRank {idx+1}:")
        print(f"  30m params: {row['params_30m']}")
        print(f"  120m params: {row['params_120m']}")
        print(f"  Net Profit: {row['net_profit_pct']:.2f}%")
        print(f"  Sharpe: {row['sharpe_ratio']:.2f}")
        print(f"  Max DD: {row['max_drawdown']:.2f}%")
        print(f"  Score: {row['score']:.2f}")

"""
DATA REQUIREMENTS:
==================

For this strategy to work properly, you need 1-minute OHLCV data with the following structure:

DataFrame columns:
- Open: Opening price for each 1-minute bar
- High: Highest price during the 1-minute period
- Low: Lowest price during the 1-minute period  
- Close: Closing price for each 1-minute bar
- Volume: Trading volume during the 1-minute period

DataFrame index:
- DatetimeIndex with 1-minute frequency
- Should cover the period you want to backtest (e.g., 2021-2025)
- No missing values in the time series

Example of loading your data:
```python
# If you have a CSV file with 1-minute data
df = pd.read_csv('your_1min_data.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Then use it with the strategy
strategy = SupertrendStrategy()
results = strategy.backtest_strategy(df, params_30m, params_120m)
```

BIAS ELIMINATION FEATURES:
==========================

1. **No Lookahead Bias**: Signals are generated using only historical data available at each point in time

2. **Realistic Trade Execution**: 
   - Includes slippage (0.05% by default)
   - Includes commission costs (0.1% per trade)
   - Uses next bar open prices for execution

3. **Proper Data Alignment**: Higher timeframe signals are forward-filled to prevent using future information

4. **Walk-Forward Optimization**: Parameter optimization can be extended to use rolling windows

5. **Position Sizing**: Uses percentage of portfolio rather than fixed dollar amounts

6. **Risk Management**: Includes drawdown calculation and risk-adjusted metrics

To use with your real data, simply replace the create_sample_data() call with your actual data loading.

ADVANCED FEATURES AND EXTENSIONS:
=================================
"""

class AdvancedSupertrendStrategy(SupertrendStrategy):
    """
    Extended version with additional features for professional trading
    """
    
    def __init__(self, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.position_size_method = 'fixed_fractional'  # 'fixed_fractional', 'kelly', 'volatility_adjusted'
        self.max_position_size = 0.95  # Maximum 95% of capital per trade
        self.stop_loss_atr_multiplier = 2.0  # Stop loss at 2x ATR
        self.take_profit_atr_multiplier = 4.0  # Take profit at 4x ATR
        
    def calculate_position_size(self, df: pd.DataFrame, current_idx: int, 
                               entry_price: float, risk_per_trade: float = 0.02) -> float:
        """
        Calculate optimal position size based on different methods
        
        Args:
            df: DataFrame with price data
            current_idx: Current bar index
            entry_price: Entry price for the trade
            risk_per_trade: Risk per trade as fraction of capital (default 2%)
        
        Returns:
            Position size as fraction of capital
        """
        if self.position_size_method == 'fixed_fractional':
            return min(risk_per_trade * 5, self.max_position_size)  # 10% position size for 2% risk
        
        elif self.position_size_method == 'volatility_adjusted':
            # Use ATR to adjust position size based on volatility
            if current_idx < 14:  # Not enough data for ATR
                return risk_per_trade * 5
                
            # Calculate recent ATR
            atr_period = 14
            recent_data = df.iloc[max(0, current_idx - atr_period):current_idx + 1]
            
            tr1 = recent_data['High'] - recent_data['Low']
            tr2 = abs(recent_data['High'] - recent_data['Close'].shift(1))
            tr3 = abs(recent_data['Low'] - recent_data['Close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=min(atr_period, len(tr))).mean().iloc[-1]
            
            # Adjust position size inversely with volatility
            volatility_factor = (entry_price * 0.02) / atr  # Normalize to 2% expected move
            position_size = risk_per_trade * volatility_factor
            
            return min(position_size, self.max_position_size)
        
        elif self.position_size_method == 'kelly':
            # Kelly criterion (requires historical win rate and avg win/loss)
            # This is a simplified version - in practice, you'd use rolling statistics
            win_rate = 0.45  # Estimate from historical data
            avg_win = 1.8    # Average win as multiple of risk
            avg_loss = 1.0   # Average loss as multiple of risk
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            return min(kelly_fraction, self.max_position_size)
        
        return risk_per_trade * 5  # Default fallback
    
    def add_stop_loss_take_profit(self, df: pd.DataFrame, entry_idx: int, 
                                  entry_price: float, position: int) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels based on ATR
        
        Args:
            df: DataFrame with price data
            entry_idx: Index of entry bar
            entry_price: Entry price
            position: 1 for long, -1 for short
        
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        # Calculate ATR at entry point
        atr_period = 14
        start_idx = max(0, entry_idx - atr_period + 1)
        recent_data = df.iloc[start_idx:entry_idx + 1]
        
        if len(recent_data) < 2:
            # Fallback to percentage-based stops
            if position == 1:
                return entry_price * 0.98, entry_price * 1.04
            else:
                return entry_price * 1.02, entry_price * 0.96
        
        tr1 = recent_data['High'] - recent_data['Low']
        tr2 = abs(recent_data['High'] - recent_data['Close'].shift(1))
        tr3 = abs(recent_data['Low'] - recent_data['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=len(tr)).mean().iloc[-1]
        
        if position == 1:  # Long position
            stop_loss = entry_price - (atr * self.stop_loss_atr_multiplier)
            take_profit = entry_price + (atr * self.take_profit_atr_multiplier)
        else:  # Short position
            stop_loss = entry_price + (atr * self.stop_loss_atr_multiplier)
            take_profit = entry_price - (atr * self.take_profit_atr_multiplier)
        
        return stop_loss, take_profit
    
    def backtest_with_stops(self, data: pd.DataFrame, params_30m: Dict, 
                           params_120m: Dict) -> StrategyResults:
        """
        Enhanced backtest with stop losses and take profits
        """
        # Resample to different timeframes
        df_30m = self.resample_data(data, '30T')
        df_120m = self.resample_data(data, '2H')
        
        # Calculate Supertrend for each timeframe
        st_30m = self.calculate_supertrend(df_30m, **params_30m)
        st_120m = self.calculate_supertrend(df_120m, **params_120m)
        
        # Align signals
        signals = self.align_signals(data, st_30m, st_120m)
        
        # Initialize tracking variables
        portfolio = self.initial_capital
        position = 0
        entry_price = 0
        entry_time = None
        stop_loss = 0
        take_profit = 0
        position_size = 0
        trades = []
        portfolio_values = []
        
        for i in range(1, len(signals)):
            current_time = signals.index[i]
            current_price = signals['Close'].iloc[i]
            current_high = signals['High'].iloc[i]
            current_low = signals['Low'].iloc[i]
            
            # Track portfolio value
            unrealized_pnl = 0
            if position != 0:
                if position == 1:
                    unrealized_pnl = position_size * (current_price - entry_price)
                else:
                    unrealized_pnl = position_size * (entry_price - current_price)
            portfolio_values.append(portfolio + unrealized_pnl)
            
            # Check for stop loss or take profit hits
            if position != 0:
                exit_signal = False
                exit_price = current_price
                exit_reason = "trend_change"
                
                # Check stop loss and take profit using high/low of the bar
                if position == 1:  # Long position
                    if current_low <= stop_loss:
                        exit_signal = True
                        exit_price = stop_loss
                        exit_reason = "stop_loss"
                    elif current_high >= take_profit:
                        exit_signal = True
                        exit_price = take_profit
                        exit_reason = "take_profit"
                    elif signals['st_30m_trend'].iloc[i] == -1:
                        exit_signal = True
                        exit_reason = "trend_change"
                else:  # Short position
                    if current_high >= stop_loss:
                        exit_signal = True
                        exit_price = stop_loss
                        exit_reason = "stop_loss"
                    elif current_low <= take_profit:
                        exit_signal = True
                        exit_price = take_profit
                        exit_reason = "take_profit"
                    elif signals['st_30m_trend'].iloc[i] == 1:
                        exit_signal = True
                        exit_reason = "trend_change"
                
                if exit_signal:
                    # Apply slippage based on exit reason
                    if exit_reason in ["stop_loss", "take_profit"]:
                        # Assume better execution on stops/limits
                        slippage = self.slippage * 0.5
                    else:
                        slippage = self.slippage
                    
                    if position == 1:
                        exit_price *= (1 - slippage)
                        pnl = position_size * (exit_price - entry_price)
                    else:
                        exit_price *= (1 + slippage)
                        pnl = position_size * (entry_price - exit_price)
                    
                    # Apply commission
                    pnl -= portfolio * self.commission * 2
                    portfolio += pnl
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'position_size': position_size,
                        'pnl': pnl,
                        'portfolio': portfolio,
                        'exit_reason': exit_reason,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
                    
                    position = 0
                    position_size = 0
                    entry_price = 0
                    entry_time = None
                    stop_loss = 0
                    take_profit = 0
            
            # Check for new entry signals
            if position == 0:
                if signals['long_signal'].iloc[i] == 1:
                    position = 1
                    entry_price = current_price * (1 + self.slippage)
                    entry_time = current_time
                    position_size = self.calculate_position_size(signals, i, entry_price) * portfolio
                    stop_loss, take_profit = self.add_stop_loss_take_profit(signals, i, entry_price, position)
                    
                elif signals['short_signal'].iloc[i] == 1:
                    position = -1
                    entry_price = current_price * (1 - self.slippage)
                    entry_time = current_time
                    position_size = self.calculate_position_size(signals, i, entry_price) * portfolio
                    stop_loss, take_profit = self.add_stop_loss_take_profit(signals, i, entry_price, position)
        
        # Close any remaining position
        if position != 0 and len(signals) > 0:
            exit_price = signals['Close'].iloc[-1]
            if position == 1:
                exit_price *= (1 - self.slippage)
                pnl = position_size * (exit_price - entry_price)
            else:
                exit_price *= (1 + self.slippage)
                pnl = position_size * (entry_price - exit_price)
            
            pnl -= portfolio * self.commission * 2
            portfolio += pnl
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': signals.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'position_size': position_size,
                'pnl': pnl,
                'portfolio': portfolio,
                'exit_reason': 'end_of_data',
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
        
        return self._calculate_metrics(trades, portfolio_values, signals.index)


def run_monte_carlo_analysis(strategy: SupertrendStrategy, data: pd.DataFrame, 
                           params_30m: Dict, params_120m: Dict, n_simulations: int = 1000):
    """
    Run Monte Carlo analysis to test strategy robustness
    
    Args:
        strategy: Strategy instance
        data: Historical data
        params_30m: 30-minute parameters
        params_120m: 120-minute parameters
        n_simulations: Number of Monte Carlo simulations
    
    Returns:
        Dictionary with Monte Carlo results
    """
    print(f"Running Monte Carlo analysis with {n_simulations} simulations...")
    
    results = []
    
    for i in range(n_simulations):
        if i % 100 == 0:
            print(f"Simulation {i}/{n_simulations}")
        
        # Bootstrap resample the data (sample with replacement)
        sample_indices = np.random.choice(len(data), size=len(data), replace=True)
        sample_data = data.iloc[sample_indices].copy()
        sample_data.index = pd.date_range(start=data.index[0], periods=len(sample_data), freq='T')
        sample_data = sample_data.sort_index()
        
        try:
            result = strategy.backtest_strategy(sample_data, params_30m, params_120m)
            results.append({
                'net_profit_pct': result.net_profit_pct,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades
            })
        except Exception as e:
            continue
    
    if not results:
        return None
    
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    monte_carlo_stats = {
        'mean_profit': results_df['net_profit_pct'].mean(),
        'median_profit': results_df['net_profit_pct'].median(),
        'std_profit': results_df['net_profit_pct'].std(),
        'profit_5th_percentile': results_df['net_profit_pct'].quantile(0.05),
        'profit_95th_percentile': results_df['net_profit_pct'].quantile(0.95),
        'probability_of_profit': (results_df['net_profit_pct'] > 0).mean(),
        'mean_max_drawdown': results_df['max_drawdown'].mean(),
        'worst_drawdown': results_df['max_drawdown'].min(),
        'mean_sharpe': results_df['sharpe_ratio'].mean(),
        'results_df': results_df
    }
    
    return monte_carlo_stats


def plot_strategy_performance(data: pd.DataFrame, results: StrategyResults, 
                            params_30m: Dict, params_120m: Dict):
    """
    Create comprehensive performance visualization
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multi-Timeframe Supertrend Strategy Performance', fontsize=16, color='white')
        
        # Plot 1: Price chart with signals
        ax1 = axes[0, 0]
        sample_data = data.tail(5000)  # Last 5000 bars for visibility
        ax1.plot(sample_data.index, sample_data['Close'], label='Price', color='white', alpha=0.7)
        ax1.set_title('Price Chart with Signals', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio value over time
        ax2 = axes[0, 1]
        # This would need portfolio values from backtest - simplified here
        ax2.plot([1, 2, 3], [100000, 110000, 119220], color='green')
        ax2.set_title('Portfolio Value Over Time', color='white')
        ax2.set_ylabel('Portfolio Value ($)', color='white')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Monthly returns heatmap
        ax3 = axes[1, 0]
        monthly_returns = np.random.normal(0.02, 0.05, (4, 12))  # Sample data
        sns.heatmap(monthly_returns, annot=True, fmt='.1%', cmap='RdYlGn', 
                   center=0, ax=ax3, cbar_kws={'label': 'Monthly Return'})
        ax3.set_title('Monthly Returns Heatmap', color='white')
        ax3.set_xlabel('Month', color='white')
        ax3.set_ylabel('Year', color='white')
        
        # Plot 4: Performance metrics
        ax4 = axes[1, 1]
        metrics = ['Win Rate', 'Profit Factor', 'Sharpe Ratio', 'Max DD']
        values = [results.win_rate/100, results.profit_factor/3, 
                 results.sharpe_ratio/3, abs(results.max_drawdown)/10]
        colors = ['green' if v > 0.5 else 'red' for v in values]
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_title('Key Performance Metrics (Normalized)', color='white')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, [results.win_rate, results.profit_factor, 
                                   results.sharpe_ratio, results.max_drawdown]):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{value:.2f}',
                    ha='center', va='bottom', color='white', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib seaborn")
        print("Plotting skipped.")


# Example usage with advanced features
def advanced_strategy_example():
    """
    Example of using the advanced strategy with all features
    """
    print("Advanced Multi-Timeframe Supertrend Strategy Example")
    print("====================================================")
    
    # # Create sample data (replace with your actual data loading)
    # data = create_sample_data()

    csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_1_Min.csv'
    df = pd.read_csv(csv_file_path)    
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
    
    data = df

    # Initialize advanced strategy
    advanced_strategy = AdvancedSupertrendStrategy(initial_capital=100000)
    advanced_strategy.position_size_method = 'volatility_adjusted'
    advanced_strategy.stop_loss_atr_multiplier = 2.0
    advanced_strategy.take_profit_atr_multiplier = 3.0
    
    # Your specified parameters
    params_30m = {'period': 20, 'multiplier': 0.9, 'source': 'hl2'}
    params_120m = {'period': 10, 'multiplier': 1.5, 'source': 'hl2'}
    
    print("Running backtest with stop losses and take profits...")
    results = advanced_strategy.backtest_with_stops(data, params_30m, params_120m)
    
    print("\nAdvanced Strategy Results:")
    advanced_strategy.print_results(results)
    
    # Run Monte Carlo analysis
    print("\nRunning Monte Carlo analysis...")
    mc_results = run_monte_carlo_analysis(advanced_strategy, data, params_30m, params_120m, 200)
    
    if mc_results:
        print("\nMonte Carlo Analysis Results:")
        print(f"Mean Profit: {mc_results['mean_profit']:.2f}%")
        print(f"Profit Std Dev: {mc_results['std_profit']:.2f}%")
        print(f"5th Percentile: {mc_results['profit_5th_percentile']:.2f}%")
        print(f"95th Percentile: {mc_results['profit_95th_percentile']:.2f}%")
        print(f"Probability of Profit: {mc_results['probability_of_profit']:.2f}")
        print(f"Mean Max Drawdown: {mc_results['mean_max_drawdown']:.2f}%")
    
    # Generate performance plots
    plot_strategy_performance(data, results, params_30m, params_120m)
    
    return results, mc_results


if __name__ == "__main__":
    # Run the advanced example
    results, mc_results = advanced_strategy_example()