import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class RSICrossoverOptimizer:
    def __init__(self, csv_file_path):
        """
        Initialize the RSI Crossover Strategy Optimizer
        
        Parameters:
        csv_file_path (str): Path to CSV file with OHLC data
        Expected columns: Date, Open, High, Low, Close (and optionally Volume)
        """
        self.data = self.load_data(csv_file_path)
        self.results = []
        
    def load_data(self, csv_file_path):
        """Load and prepare OHLC data"""
        try:
            df = pd.read_csv(csv_file_path)
            
            # Standardize column names
            df.columns = df.columns.str.strip().str.title()
            
            # Try different possible date column names
            date_cols = ['Date', 'Datetime', 'Time', 'Timestamp']
            date_col = None
            for col in date_cols:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            else:
                print("Warning: No date column found. Using index as date.")
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            
            df.sort_index(inplace=True)
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in CSV")
            
            print(f"Data loaded successfully: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def resample_to_weekly(self, df):
        """Convert daily data to weekly data"""
        weekly = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()
        return weekly
    
    def generate_signals(self, daily_rsi_period, weekly_rsi_period, rsi_filter=None, 
                        divergence_filter=False, smoothing=False):
        """
        Generate buy/sell signals based on RSI crossover strategy
        
        Parameters:
        daily_rsi_period (int): Period for daily RSI calculation
        weekly_rsi_period (int): Period for weekly RSI calculation
        rsi_filter (str): Optional filter ('above_50', 'below_50', None)
        divergence_filter (bool): Apply RSI divergence filter
        smoothing (bool): Apply smoothing to RSI values
        """
        df = self.data.copy()
        
        # Calculate daily RSI
        daily_rsi = self.calculate_rsi(df['Close'], daily_rsi_period)
        
        # Calculate weekly RSI from weekly data
        weekly_data = self.resample_to_weekly(df)
        weekly_rsi = self.calculate_rsi(weekly_data['Close'], weekly_rsi_period)
        
        # Align weekly RSI with daily data
        weekly_rsi_daily = weekly_rsi.reindex(df.index, method='ffill')
        
        # Apply smoothing if requested
        if smoothing:
            daily_rsi = daily_rsi.rolling(window=3).mean()
            weekly_rsi_daily = weekly_rsi_daily.rolling(window=3).mean()
        
        # Create signals DataFrame
        signals_df = pd.DataFrame(index=df.index)
        signals_df['Close'] = df['Close']
        signals_df['Daily_RSI'] = daily_rsi
        signals_df['Weekly_RSI'] = weekly_rsi_daily
        
        # Generate crossover signals
        signals_df['RSI_Cross'] = np.where(
            daily_rsi > weekly_rsi_daily, 1,
            np.where(daily_rsi < weekly_rsi_daily, -1, 0)
        )
        
        # Identify actual crossover points
        signals_df['Signal'] = 0
        cross_changes = signals_df['RSI_Cross'].diff()
        signals_df.loc[cross_changes == 2, 'Signal'] = 1  # Buy signal
        signals_df.loc[cross_changes == -2, 'Signal'] = -1  # Sell signal
        
        # Apply RSI level filter
        if rsi_filter == 'above_50':
            signals_df.loc[(signals_df['Signal'] == 1) & (signals_df['Daily_RSI'] <= 50), 'Signal'] = 0
        elif rsi_filter == 'below_50':
            signals_df.loc[(signals_df['Signal'] == -1) & (signals_df['Daily_RSI'] >= 50), 'Signal'] = 0
        
        # Apply divergence filter (simplified version)
        if divergence_filter:
            # Look for price vs RSI divergence over 10-day window
            price_slope = df['Close'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            rsi_slope = daily_rsi.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            
            # Bullish divergence: price falling, RSI rising
            bullish_div = (price_slope < 0) & (rsi_slope > 0)
            # Bearish divergence: price rising, RSI falling
            bearish_div = (price_slope > 0) & (rsi_slope < 0)
            
            # Only keep signals with divergence confirmation
            signals_df.loc[(signals_df['Signal'] == 1) & ~bullish_div, 'Signal'] = 0
            signals_df.loc[(signals_df['Signal'] == -1) & ~bearish_div, 'Signal'] = 0
        
        return signals_df.dropna()
    
    def backtest_strategy(self, signals_df, initial_capital=100000, commission=0.001):
        """
        Backtest the strategy and calculate performance metrics
        
        Parameters:
        signals_df (DataFrame): DataFrame with signals
        initial_capital (float): Starting capital
        commission (float): Commission rate (0.001 = 0.1%)
        """
        df = signals_df.copy()
        df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0)
        df['Position'] = df['Position'].shift(1).fillna(0)  # Lag positions by 1 day
        
        # Calculate returns
        df['Price_Change'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Position'] * df['Price_Change']
        
        # Apply commission costs
        df['Trades'] = df['Position'].diff().abs()
        df['Commission_Cost'] = df['Trades'] * commission
        df['Net_Strategy_Return'] = df['Strategy_Return'] - df['Commission_Cost']
        
        # Calculate cumulative returns
        df['Cumulative_Return'] = (1 + df['Net_Strategy_Return']).cumprod()
        df['Buy_Hold_Return'] = (1 + df['Price_Change']).cumprod()
        
        # Calculate performance metrics
        total_return = (df['Cumulative_Return'].iloc[-1] - 1) * 100
        buy_hold_return = (df['Buy_Hold_Return'].iloc[-1] - 1) * 100
        
        # Calculate number of trades
        num_trades = int(df['Trades'].sum() / 2)  # Divide by 2 since each trade has entry and exit
        
        # Calculate winning trades
        trade_returns = []
        position = 0
        entry_price = 0
        
        for i, row in df.iterrows():
            if row['Position'] != position:
                if position != 0:  # Closing a position
                    trade_return = (row['Close'] - entry_price) / entry_price * position
                    trade_returns.append(trade_return)
                
                if row['Position'] != 0:  # Opening new position
                    entry_price = row['Close']
                
                position = row['Position']
        
        if len(trade_returns) > 0:
            win_rate = (np.array(trade_returns) > 0).mean() * 100
            avg_win = np.mean([r for r in trade_returns if r > 0]) * 100 if any(r > 0 for r in trade_returns) else 0
            avg_loss = np.mean([r for r in trade_returns if r < 0]) * 100 if any(r < 0 for r in trade_returns) else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        # Calculate Sharpe ratio (annualized)
        returns = df['Net_Strategy_Return'].dropna()
        if len(returns) > 0 and returns.std() != 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        cumulative = df['Cumulative_Return']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Calculate Calmar ratio (Total return / Max drawdown)
        calmar_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'Total_Return': total_return,
            'Buy_Hold_Return': buy_hold_return,
            'Excess_Return': total_return - buy_hold_return,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Calmar_Ratio': calmar_ratio,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades,
            'Avg_Win': avg_win,
            'Avg_Loss': avg_loss,
            'Trade_Returns': trade_returns,
            'Equity_Curve': df[['Cumulative_Return', 'Buy_Hold_Return']]
        }
    
    def optimize_parameters(self):
        """
        Run optimization across all parameter combinations
        """
        # Parameter ranges
        daily_rsi_periods = [5, 10, 14]
        weekly_rsi_periods = [10, 14, 21]
        rsi_filters = [None, 'above_50']
        divergence_filters = [False, True]
        smoothing_options = [False, True]
        
        print("Starting parameter optimization...")
        print(f"Total combinations to test: {len(daily_rsi_periods) * len(weekly_rsi_periods) * len(rsi_filters) * len(divergence_filters) * len(smoothing_options)}")
        
        counter = 0
        for daily_rsi, weekly_rsi, rsi_filter, divergence, smoothing in product(
            daily_rsi_periods, weekly_rsi_periods, rsi_filters, divergence_filters, smoothing_options):
            
            counter += 1
            if counter % 10 == 0:
                print(f"Progress: {counter}/{len(daily_rsi_periods) * len(weekly_rsi_periods) * len(rsi_filters) * len(divergence_filters) * len(smoothing_options)}")
            
            try:
                # Generate signals
                signals_df = self.generate_signals(
                    daily_rsi, weekly_rsi, rsi_filter, divergence, smoothing
                )
                
                # Skip if not enough signals generated
                if signals_df['Signal'].abs().sum() < 2:
                    continue
                
                # Backtest strategy
                performance = self.backtest_strategy(signals_df)
                
                # Store results
                result = {
                    'Daily_RSI_Period': daily_rsi,
                    'Weekly_RSI_Period': weekly_rsi,
                    'RSI_Filter': rsi_filter if rsi_filter else 'None',
                    'Divergence_Filter': divergence,
                    'Smoothing': smoothing,
                    **performance
                }
                
                self.results.append(result)
                
            except Exception as e:
                print(f"Error with parameters {daily_rsi}, {weekly_rsi}, {rsi_filter}, {divergence}, {smoothing}: {e}")
                continue
        
        print(f"Optimization completed! Tested {len(self.results)} valid parameter combinations.")
    
    def get_best_parameters(self, sort_by='Calmar_Ratio', min_trades=5):
        """
        Get best performing parameter combinations
        
        Parameters:
        sort_by (str): Metric to sort by ('Total_Return', 'Sharpe_Ratio', 'Calmar_Ratio', 'Win_Rate')
        min_trades (int): Minimum number of trades required
        """
        if not self.results:
            print("No results found. Run optimize_parameters() first.")
            return None
        
        results_df = pd.DataFrame(self.results)
        
        # Filter by minimum trades
        results_df = results_df[results_df['Num_Trades'] >= min_trades]
        
        if len(results_df) == 0:
            print(f"No results with minimum {min_trades} trades found.")
            return None
        
        # Sort by specified metric
        if sort_by in results_df.columns:
            results_df = results_df.sort_values(sort_by, ascending=False)
        else:
            print(f"Sort metric '{sort_by}' not found. Using 'Total_Return'.")
            results_df = results_df.sort_values('Total_Return', ascending=False)
        
        return results_df
    
    def display_results(self, top_n=10, sort_by='Calmar_Ratio'):
        """Display top performing parameter combinations"""
        best_results = self.get_best_parameters(sort_by=sort_by)
        
        if best_results is None:
            return
        
        print(f"\n{'='*80}")
        print(f"TOP {top_n} PARAMETER COMBINATIONS (Sorted by {sort_by})")
        print(f"{'='*80}")
        
        display_cols = [
            'Daily_RSI_Period', 'Weekly_RSI_Period', 'RSI_Filter', 'Divergence_Filter', 'Smoothing',
            'Total_Return', 'Excess_Return', 'Sharpe_Ratio', 'Max_Drawdown', 'Calmar_Ratio',
            'Win_Rate', 'Num_Trades'
        ]
        
        for i, (idx, row) in enumerate(best_results.head(top_n).iterrows()):
            print(f"\nRank #{i+1}:")
            print("-" * 40)
            for col in display_cols:
                if col in row:
                    if col in ['Total_Return', 'Excess_Return', 'Max_Drawdown', 'Win_Rate']:
                        print(f"{col:20}: {row[col]:8.2f}%")
                    elif col in ['Sharpe_Ratio', 'Calmar_Ratio']:
                        print(f"{col:20}: {row[col]:8.3f}")
                    else:
                        print(f"{col:20}: {row[col]}")
        
        # Summary statistics
        print(f"\n{'='*50}")
        print("SUMMARY STATISTICS")
        print(f"{'='*50}")
        print(f"Total combinations tested: {len(best_results)}")
        print(f"Best Total Return: {best_results['Total_Return'].max():.2f}%")
        print(f"Best Sharpe Ratio: {best_results['Sharpe_Ratio'].max():.3f}")
        print(f"Lowest Max Drawdown: {best_results['Max_Drawdown'].max():.2f}%")  # Max because drawdown is negative
        print(f"Best Win Rate: {best_results['Win_Rate'].max():.2f}%")
        print(f"Average number of trades: {best_results['Num_Trades'].mean():.1f}")

# Usage Example
if __name__ == "__main__":
    # Initialize optimizer with your CSV file
    # Replace 'your_data.csv' with the path to your OHLC CSV file
    optimizer = RSICrossoverOptimizer('your_data.csv')
    
    # Run optimization
    optimizer.optimize_parameters()
    
    # Display results sorted by different metrics
    print("Results sorted by Calmar Ratio (Return/Drawdown):")
    optimizer.display_results(top_n=5, sort_by='Calmar_Ratio')
    
    print("\nResults sorted by Total Return:")
    optimizer.display_results(top_n=5, sort_by='Total_Return')
    
    print("\nResults sorted by Win Rate:")
    optimizer.display_results(top_n=5, sort_by='Win_Rate')
    
    # Get the best parameters for further analysis
    best_params = optimizer.get_best_parameters(sort_by='Calmar_Ratio')
    
    if best_params is not None and len(best_params) > 0:
        print(f"\nBest parameter combination:")
        best = best_params.iloc[0]
        print(f"Daily RSI Period: {best['Daily_RSI_Period']}")
        print(f"Weekly RSI Period: {best['Weekly_RSI_Period']}")
        print(f"RSI Filter: {best['RSI_Filter']}")
        print(f"Divergence Filter: {best['Divergence_Filter']}")
        print(f"Smoothing: {best['Smoothing']}")
        print(f"Total Return: {best['Total_Return']:.2f}%")
        print(f"Max Drawdown: {best['Max_Drawdown']:.2f}%")
        print(f"Win Rate: {best['Win_Rate']:.2f}%")
        print(f"Number of Trades: {best['Num_Trades']}")
