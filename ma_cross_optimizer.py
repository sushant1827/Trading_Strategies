import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class MACrossStrategyOptimizer:
    def __init__(self, data_file):
        """
        Initialize the strategy optimizer with OHLC data
        
        Parameters:
        data_file: str - Path to CSV file with OHLC data
        Expected columns: Date, Open, High, Low, Close, Volume (optional)
        """
        self.data = self.load_data(data_file)
        self.results = []
        
    def load_data(self, file_path):
        """Load and prepare OHLC data"""
        try:
            df = pd.read_csv(file_path)
            
            # Try different common date column names
            date_cols = ['Date', 'date', 'DATE', 'Datetime', 'datetime']
            date_col = None
            for col in date_cols:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                # Try lowercase versions
                col_mapping = {}
                for col in missing_cols:
                    if col.lower() in df.columns:
                        col_mapping[col.lower()] = col
                df.rename(columns=col_mapping, inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            print(f"Data loaded successfully: {len(df)} rows")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please ensure your CSV has columns: Date, Open, High, Low, Close")
            return None
    
    def calculate_moving_averages(self, df, daily_period, weekly_period):
        """Calculate daily and weekly moving averages"""
        df_copy = df.copy()
        
        # Daily MA
        df_copy['DMA'] = df_copy['Close'].rolling(window=daily_period).mean()
        
        # Weekly MA (using 5x daily periods to approximate weekly)
        df_copy['WMA'] = df_copy['Close'].rolling(window=weekly_period * 5).mean()
        
        # 200-day SMA filter
        df_copy['SMA_200'] = df_copy['Close'].rolling(window=200).mean()
        
        return df_copy
    
    def generate_signals(self, df):
        """Generate buy/sell signals based on MA crossovers"""
        df_copy = df.copy()
        
        # Initialize signals
        df_copy['Signal'] = 0
        df_copy['Position'] = 0
        
        # Calculate crossovers
        df_copy['DMA_above_WMA'] = df_copy['DMA'] > df_copy['WMA']
        df_copy['DMA_above_WMA_prev'] = df_copy['DMA_above_WMA'].shift(1)
        
        # Buy signal: DMA crosses above WMA and price > 200-day SMA
        buy_condition = (
            (df_copy['DMA_above_WMA'] == True) & 
            (df_copy['DMA_above_WMA_prev'] == False) &
            (df_copy['Close'] > df_copy['SMA_200'])
        )
        
        # Sell signal: DMA crosses below WMA
        sell_condition = (
            (df_copy['DMA_above_WMA'] == False) & 
            (df_copy['DMA_above_WMA_prev'] == True)
        )
        
        df_copy.loc[buy_condition, 'Signal'] = 1
        df_copy.loc[sell_condition, 'Signal'] = -1
        
        # Calculate positions
        df_copy['Position'] = df_copy['Signal'].replace(to_replace=0, method='ffill').fillna(0)
        
        return df_copy
    
    def calculate_returns(self, df):
        """Calculate strategy returns and metrics"""
        df_copy = df.copy()
        
        # Calculate daily returns
        df_copy['Returns'] = df_copy['Close'].pct_change()
        df_copy['Strategy_Returns'] = df_copy['Returns'] * df_copy['Position'].shift(1)
        
        # Calculate cumulative returns
        df_copy['Cumulative_Returns'] = (1 + df_copy['Returns']).cumprod()
        df_copy['Cumulative_Strategy_Returns'] = (1 + df_copy['Strategy_Returns']).cumprod()
        
        return df_copy
    
    def calculate_metrics(self, df):
        """Calculate performance metrics"""
        strategy_returns = df['Strategy_Returns'].dropna()
        
        if len(strategy_returns) == 0:
            return None
        
        # Basic metrics
        total_return = df['Cumulative_Strategy_Returns'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = df['Cumulative_Strategy_Returns']
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = len(strategy_returns[strategy_returns != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Number of trades (buy signals)
        num_trades = (df['Signal'] == 1).sum()
        
        return {
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades
        }
    
    def backtest_strategy(self, daily_period, weekly_period):
        """Backtest strategy with given parameters"""
        if self.data is None:
            return None
        
        # Calculate moving averages
        df = self.calculate_moving_averages(self.data, daily_period, weekly_period)
        
        # Generate signals
        df = self.generate_signals(df)
        
        # Calculate returns
        df = self.calculate_returns(df)
        
        # Calculate metrics
        metrics = self.calculate_metrics(df)
        
        if metrics:
            metrics['Daily_Period'] = daily_period
            metrics['Weekly_Period'] = weekly_period
            
        return metrics, df
    
    def optimize_parameters(self, daily_periods=[5, 10, 20, 50], 
                          weekly_periods=[10, 20, 50, 100]):
        """Optimize strategy parameters"""
        print("Starting parameter optimization...")
        
        self.results = []
        total_combinations = len(daily_periods) * len(weekly_periods)
        current = 0
        
        for daily_p, weekly_p in product(daily_periods, weekly_periods):
            current += 1
            print(f"Testing combination {current}/{total_combinations}: DMA={daily_p}, WMA={weekly_p}")
            
            try:
                metrics, _ = self.backtest_strategy(daily_p, weekly_p)
                if metrics:
                    self.results.append(metrics)
            except Exception as e:
                print(f"Error with parameters DMA={daily_p}, WMA={weekly_p}: {e}")
        
        if self.results:
            self.results_df = pd.DataFrame(self.results)
            print(f"Optimization complete! Tested {len(self.results)} combinations.")
            return self.results_df
        else:
            print("No valid results found!")
            return None
    
    def get_best_parameters(self, metric='Sharpe_Ratio'):
        """Get best parameters based on specified metric"""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results available. Run optimize_parameters first.")
            return None
        
        if metric not in self.results_df.columns:
            print(f"Metric '{metric}' not found. Available metrics: {list(self.results_df.columns)}")
            return None
        
        # Sort by metric (descending for positive metrics, ascending for drawdown)
        ascending = True if 'Drawdown' in metric else False
        best_result = self.results_df.sort_values(metric, ascending=ascending).iloc[0]
        
        return {
            'Daily_Period': int(best_result['Daily_Period']),
            'Weekly_Period': int(best_result['Weekly_Period']),
            'Metric_Value': best_result[metric],
            'All_Metrics': best_result.to_dict()
        }
    
    def display_top_results(self, top_n=10):
        """Display top performing parameter combinations"""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results available. Run optimize_parameters first.")
            return
        
        print(f"\nTop {top_n} Parameter Combinations by Sharpe Ratio:")
        print("="*80)
        
        top_results = self.results_df.sort_values('Sharpe_Ratio', ascending=False).head(top_n)
        
        for i, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"\n{i}. DMA: {int(row['Daily_Period'])}, WMA: {int(row['Weekly_Period'])}")
            print(f"   Sharpe Ratio: {row['Sharpe_Ratio']:.3f}")
            print(f"   Annual Return: {row['Annual_Return']:.2%}")
            print(f"   Max Drawdown: {row['Max_Drawdown']:.2%}")
            print(f"   Win Rate: {row['Win_Rate']:.2%}")
            print(f"   Number of Trades: {int(row['Num_Trades'])}")
    
    def plot_optimization_results(self):
        """Plot heatmaps of optimization results"""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results available. Run optimize_parameters first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strategy Optimization Results', fontsize=16)
        
        metrics = ['Sharpe_Ratio', 'Annual_Return', 'Max_Drawdown', 'Win_Rate']
        titles = ['Sharpe Ratio', 'Annual Return', 'Max Drawdown', 'Win Rate']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = i // 2, i % 2
            
            # Create pivot table for heatmap
            pivot_data = self.results_df.pivot(
                index='Weekly_Period', 
                columns='Daily_Period', 
                values=metric
            )
            
            # Create heatmap
            cmap = 'RdYlGn' if metric != 'Max_Drawdown' else 'RdYlGn_r'
            sns.heatmap(
                pivot_data, 
                annot=True, 
                fmt='.3f', 
                cmap=cmap,
                ax=axes[row, col],
                cbar_kws={'label': title}
            )
            
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Daily MA Period')
            axes[row, col].set_ylabel('Weekly MA Period')
        
        plt.tight_layout()
        plt.show()
    
    def plot_best_strategy_performance(self, metric='Sharpe_Ratio'):
        """Plot performance of best strategy"""
        best_params = self.get_best_parameters(metric)
        if not best_params:
            return
        
        print(f"\nPlotting best strategy performance:")
        print(f"Best parameters: DMA={best_params['Daily_Period']}, WMA={best_params['Weekly_Period']}")
        
        # Run backtest with best parameters
        _, df = self.backtest_strategy(
            best_params['Daily_Period'], 
            best_params['Weekly_Period']
        )
        
        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Price and Moving Averages
        axes[0].plot(df.index, df['Close'], label='Close Price', alpha=0.7)
        axes[0].plot(df.index, df['DMA'], label=f"DMA ({best_params['Daily_Period']})", linewidth=2)
        axes[0].plot(df.index, df['WMA'], label=f"WMA ({best_params['Weekly_Period']})", linewidth=2)
        axes[0].plot(df.index, df['SMA_200'], label='SMA (200)', alpha=0.5)
        
        # Mark buy/sell signals
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        
        axes[0].scatter(buy_signals.index, buy_signals['Close'], 
                       color='green', marker='^', s=100, label='Buy', zorder=5)
        axes[0].scatter(sell_signals.index, sell_signals['Close'], 
                       color='red', marker='v', s=100, label='Sell', zorder=5)
        
        axes[0].set_title('Price Chart with Moving Averages and Signals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Returns
        axes[1].plot(df.index, (df['Cumulative_Returns'] - 1) * 100, 
                    label='Buy & Hold', linewidth=2)
        axes[1].plot(df.index, (df['Cumulative_Strategy_Returns'] - 1) * 100, 
                    label='Strategy', linewidth=2)
        axes[1].set_title('Cumulative Returns Comparison')
        axes[1].set_ylabel('Returns (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        cumulative = df['Cumulative_Strategy_Returns']
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max * 100
        
        axes[2].fill_between(df.index, drawdown, 0, color='red', alpha=0.3)
        axes[2].plot(df.index, drawdown, color='red', linewidth=1)
        axes[2].set_title('Strategy Drawdown')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize optimizer with your CSV file
    # optimizer = MACrossStrategyOptimizer('your_data.csv')
    
    # Define parameter ranges to test
    daily_periods = [5, 10, 15, 20, 25, 30, 50]
    weekly_periods = [10, 15, 20, 25, 30, 40, 50, 75, 100]
    
    # Example of how to use the optimizer:
    print("MA Cross Strategy Optimizer")
    print("=" * 50)
    print("To use this optimizer:")
    print("1. optimizer = MACrossStrategyOptimizer('your_data.csv')")
    print("2. results = optimizer.optimize_parameters(daily_periods, weekly_periods)")
    print("3. optimizer.display_top_results()")
    print("4. best_params = optimizer.get_best_parameters('Sharpe_Ratio')")
    print("5. optimizer.plot_optimization_results()")
    print("6. optimizer.plot_best_strategy_performance()")
    
    # Uncomment these lines when you have your data file:
    # optimizer = MACrossStrategyOptimizer('your_data.csv')
    # results = optimizer.optimize_parameters(daily_periods, weekly_periods)
    # optimizer.display_top_results()
    # best_params = optimizer.get_best_parameters('Sharpe_Ratio')
    # print(f"\nBest parameters: {best_params}")
    # optimizer.plot_optimization_results()
    # optimizer.plot_best_strategy_performance()
