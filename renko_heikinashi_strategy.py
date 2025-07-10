import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframeStrategy:
    """
    Multi-timeframe trading strategy for NIFTY index
    Higher TF: Daily with Heikin Ashi + EMAs for trend
    Lower TF: 60min with Renko + EMAs for entries
    """
    
    def __init__(self):
        # EMA periods
        self.fast_emas = [3, 5, 7, 9, 12, 14]
        self.slow_emas = [20, 25, 30, 35, 40, 50]
        self.all_emas = self.fast_emas + self.slow_emas
        
        # Renko parameters
        self.atr_period = 14
        self.atr_multiplier = 0.5
        
        # Exit conditions
        self.exit_consecutive_bricks = 3
        
        # Strategy state
        self.current_position = None  # 'long', 'short', or None
        self.entry_price = None
        self.stop_loss = None
        self.consecutive_counter = 0
        self.last_brick_color = None
    
    def calculate_emas(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate EMAs for given periods"""
        df = data.copy()
        for period in periods:
            df[f'EMA_{period}'] = df['close'].ewm(span=period).mean()
        return df
    
    def calculate_heikin_ashi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Heikin Ashi candles"""
        df = data.copy()
        
        # Initialize first Heikin Ashi candle
        df['HA_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['HA_open'] = np.nan
        df['HA_high'] = np.nan
        df['HA_low'] = np.nan
        
        # Calculate Heikin Ashi values
        for i in range(len(df)):
            if i == 0:
                df.loc[df.index[i], 'HA_open'] = (df.loc[df.index[i], 'open'] + df.loc[df.index[i], 'close']) / 2
            else:
                df.loc[df.index[i], 'HA_open'] = (df.loc[df.index[i-1], 'HA_open'] + df.loc[df.index[i-1], 'HA_close']) / 2
            
            df.loc[df.index[i], 'HA_high'] = max(df.loc[df.index[i], 'high'], 
                                                df.loc[df.index[i], 'HA_open'], 
                                                df.loc[df.index[i], 'HA_close'])
            df.loc[df.index[i], 'HA_low'] = min(df.loc[df.index[i], 'low'], 
                                               df.loc[df.index[i], 'HA_open'], 
                                               df.loc[df.index[i], 'HA_close'])
        
        # Determine Heikin Ashi candle color
        df['HA_bullish'] = df['HA_close'] > df['HA_open']
        
        return df
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        df = data.copy()
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        return df['tr'].rolling(window=period).mean()
    
    def create_renko_bricks(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ATR-based Renko bricks"""
        df = data.copy()
        
        # Calculate ATR for brick size
        atr = self.calculate_atr(df, self.atr_period)
        brick_size = atr * self.atr_multiplier
        
        renko_data = []
        current_brick_high = None
        current_brick_low = None
        
        for i, (idx, row) in enumerate(df.iterrows()):
            if i == 0:
                # Initialize first brick
                current_brick_high = row['close']
                current_brick_low = row['close']
                renko_data.append({
                    'timestamp': idx,
                    'open': row['close'],
                    'close': row['close'],
                    'high': row['close'],
                    'low': row['close'],
                    'brick_size': brick_size.iloc[i],
                    'color': 'neutral'
                })
                continue
            
            current_brick_size = brick_size.iloc[i]
            price = row['close']
            
            # Check for new brick formation
            if price >= current_brick_high + current_brick_size:
                # Green brick
                new_brick_low = current_brick_high
                new_brick_high = current_brick_high + current_brick_size
                
                # Handle multiple bricks if price moved significantly
                while price >= new_brick_high + current_brick_size:
                    renko_data.append({
                        'timestamp': idx,
                        'open': new_brick_low,
                        'close': new_brick_high,
                        'high': new_brick_high,
                        'low': new_brick_low,
                        'brick_size': current_brick_size,
                        'color': 'green'
                    })
                    new_brick_low = new_brick_high
                    new_brick_high = new_brick_high + current_brick_size
                
                renko_data.append({
                    'timestamp': idx,
                    'open': new_brick_low,
                    'close': new_brick_high,
                    'high': new_brick_high,
                    'low': new_brick_low,
                    'brick_size': current_brick_size,
                    'color': 'green'
                })
                
                current_brick_high = new_brick_high
                current_brick_low = new_brick_low
                
            elif price <= current_brick_low - current_brick_size:
                # Red brick
                new_brick_high = current_brick_low
                new_brick_low = current_brick_low - current_brick_size
                
                # Handle multiple bricks if price moved significantly
                while price <= new_brick_low - current_brick_size:
                    renko_data.append({
                        'timestamp': idx,
                        'open': new_brick_high,
                        'close': new_brick_low,
                        'high': new_brick_high,
                        'low': new_brick_low,
                        'brick_size': current_brick_size,
                        'color': 'red'
                    })
                    new_brick_high = new_brick_low
                    new_brick_low = new_brick_low - current_brick_size
                
                renko_data.append({
                    'timestamp': idx,
                    'open': new_brick_high,
                    'close': new_brick_low,
                    'high': new_brick_high,
                    'low': new_brick_low,
                    'brick_size': current_brick_size,
                    'color': 'red'
                })
                
                current_brick_high = new_brick_high
                current_brick_low = new_brick_low
        
        return pd.DataFrame(renko_data)
    
    def check_ema_alignment(self, data: pd.DataFrame, bullish: bool = True) -> bool:
        """Check if EMAs are properly aligned"""
        latest_row = data.iloc[-1]
        
        if bullish:
            # Check if EMAs are in ascending order (3>5>7>9>12>14>20>25>30>35>40>50)
            for i in range(len(self.all_emas) - 1):
                ema_current = latest_row[f'EMA_{self.all_emas[i]}']
                ema_next = latest_row[f'EMA_{self.all_emas[i+1]}']
                if ema_current <= ema_next:
                    return False
            return True
        else:
            # Check if EMAs are in descending order (3<5<7<9<12<14<20<25<30<35<40<50)
            for i in range(len(self.all_emas) - 1):
                ema_current = latest_row[f'EMA_{self.all_emas[i]}']
                ema_next = latest_row[f'EMA_{self.all_emas[i+1]}']
                if ema_current >= ema_next:
                    return False
            return True
    
    def analyze_higher_timeframe(self, daily_data: pd.DataFrame) -> str:
        """Analyze daily timeframe for trend direction"""
        # Calculate Heikin Ashi
        ha_data = self.calculate_heikin_ashi(daily_data)
        
        # Calculate EMAs
        ha_data = self.calculate_emas(ha_data, self.all_emas)
        
        # Check latest Heikin Ashi candle
        latest_candle = ha_data.iloc[-1]
        is_ha_bullish = latest_candle['HA_bullish']
        
        # Check EMA alignment
        is_ema_bullish = self.check_ema_alignment(ha_data, bullish=True)
        is_ema_bearish = self.check_ema_alignment(ha_data, bullish=False)
        
        if is_ha_bullish and is_ema_bullish:
            return 'bullish'
        elif not is_ha_bullish and is_ema_bearish:
            return 'bearish'
        else:
            return 'neutral'
    
    def generate_signals(self, daily_data: pd.DataFrame, hourly_data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on multi-timeframe analysis"""
        # Analyze higher timeframe trend
        htf_trend = self.analyze_higher_timeframe(daily_data)
        
        # Calculate EMAs for lower timeframe
        ltf_data = self.calculate_emas(hourly_data, self.all_emas)
        
        # Create Renko bricks
        renko_data = self.create_renko_bricks(ltf_data)
        
        # Merge Renko with EMA data
        ltf_data['renko_color'] = None
        ltf_data['signal'] = None
        ltf_data['entry_price'] = None
        ltf_data['stop_loss'] = None
        ltf_data['exit_signal'] = None
        
        signals = []
        
        for i, (idx, row) in enumerate(ltf_data.iterrows()):
            signal_data = {
                'timestamp': idx,
                'htf_trend': htf_trend,
                'signal': None,
                'entry_price': None,
                'stop_loss': None,
                'exit_signal': None,
                'renko_color': None
            }
            
            # Find corresponding Renko brick
            renko_brick = None
            for _, brick in renko_data.iterrows():
                if brick['timestamp'] <= idx:
                    renko_brick = brick
                else:
                    break
            
            if renko_brick is not None:
                signal_data['renko_color'] = renko_brick['color']
                
                # Check for entry conditions
                if self.current_position is None:
                    ltf_ema_bullish = self.check_ema_alignment(ltf_data.iloc[:i+1], bullish=True)
                    ltf_ema_bearish = self.check_ema_alignment(ltf_data.iloc[:i+1], bullish=False)
                    
                    # BUY conditions
                    if (htf_trend == 'bullish' and 
                        ltf_ema_bullish and 
                        renko_brick['color'] == 'green'):
                        
                        signal_data['signal'] = 'BUY'
                        signal_data['entry_price'] = renko_brick['close']  # Next brick open
                        signal_data['stop_loss'] = self.get_previous_brick_bottom(renko_data, renko_brick)
                        
                        self.current_position = 'long'
                        self.entry_price = signal_data['entry_price']
                        self.stop_loss = signal_data['stop_loss']
                        self.consecutive_counter = 0
                    
                    # SELL conditions
                    elif (htf_trend == 'bearish' and 
                          ltf_ema_bearish and 
                          renko_brick['color'] == 'red'):
                        
                        signal_data['signal'] = 'SELL'
                        signal_data['entry_price'] = renko_brick['close']  # Next brick open
                        signal_data['stop_loss'] = self.get_previous_brick_top(renko_data, renko_brick)
                        
                        self.current_position = 'short'
                        self.entry_price = signal_data['entry_price']
                        self.stop_loss = signal_data['stop_loss']
                        self.consecutive_counter = 0
                
                # Check for exit conditions
                elif self.current_position is not None:
                    # Count consecutive opposite colored bricks
                    if self.current_position == 'long' and renko_brick['color'] == 'red':
                        if self.last_brick_color == 'red':
                            self.consecutive_counter += 1
                        else:
                            self.consecutive_counter = 1
                    elif self.current_position == 'short' and renko_brick['color'] == 'green':
                        if self.last_brick_color == 'green':
                            self.consecutive_counter += 1
                        else:
                            self.consecutive_counter = 1
                    else:
                        self.consecutive_counter = 0
                    
                    # Exit on consecutive bricks
                    if self.consecutive_counter >= self.exit_consecutive_bricks:
                        signal_data['exit_signal'] = 'EXIT'
                        signal_data['entry_price'] = renko_brick['close']
                        
                        self.current_position = None
                        self.entry_price = None
                        self.stop_loss = None
                        self.consecutive_counter = 0
                
                self.last_brick_color = renko_brick['color']
            
            signals.append(signal_data)
        
        return pd.DataFrame(signals)
    
    def get_previous_brick_bottom(self, renko_data: pd.DataFrame, current_brick: pd.Series) -> float:
        """Get the bottom of the previous brick for stop loss"""
        current_idx = None
        for i, (_, brick) in enumerate(renko_data.iterrows()):
            if brick['timestamp'] == current_brick['timestamp']:
                current_idx = i
                break
        
        if current_idx is not None and current_idx > 0:
            previous_brick = renko_data.iloc[current_idx - 1]
            return previous_brick['low']
        else:
            return current_brick['low'] * 0.99  # 1% below as fallback
    
    def get_previous_brick_top(self, renko_data: pd.DataFrame, current_brick: pd.Series) -> float:
        """Get the top of the previous brick for stop loss"""
        current_idx = None
        for i, (_, brick) in enumerate(renko_data.iterrows()):
            if brick['timestamp'] == current_brick['timestamp']:
                current_idx = i
                break
        
        if current_idx is not None and current_idx > 0:
            previous_brick = renko_data.iloc[current_idx - 1]
            return previous_brick['high']
        else:
            return current_brick['high'] * 1.01  # 1% above as fallback
    
    def backtest_strategy(self, daily_data: pd.DataFrame, hourly_data: pd.DataFrame) -> Dict:
        """Backtest the strategy and return performance metrics"""
        signals = self.generate_signals(daily_data, hourly_data)
        
        trades = []
        current_trade = None
        
        for _, signal in signals.iterrows():
            if signal['signal'] in ['BUY', 'SELL']:
                current_trade = {
                    'entry_time': signal['timestamp'],
                    'entry_type': signal['signal'],
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': None,
                    'status': 'open'
                }
            
            elif signal['exit_signal'] == 'EXIT' and current_trade is not None:
                current_trade['exit_time'] = signal['timestamp']
                current_trade['exit_price'] = signal['entry_price']
                
                if current_trade['entry_type'] == 'BUY':
                    current_trade['pnl'] = current_trade['exit_price'] - current_trade['entry_price']
                else:
                    current_trade['pnl'] = current_trade['entry_price'] - current_trade['exit_price']
                
                current_trade['status'] = 'closed'
                trades.append(current_trade)
                current_trade = None
        
        # Calculate performance metrics
        if trades:
            pnls = [trade['pnl'] for trade in trades if trade['pnl'] is not None]
            winning_trades = [pnl for pnl in pnls if pnl > 0]
            losing_trades = [pnl for pnl in pnls if pnl < 0]
            
            metrics = {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
                'total_pnl': sum(pnls),
                'avg_win': np.mean(winning_trades) if winning_trades else 0,
                'avg_loss': np.mean(losing_trades) if losing_trades else 0,
                'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf'),
                'max_drawdown': self.calculate_max_drawdown(pnls),
                'trades': trades
            }
        else:
            metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'trades': []
            }
        
        return metrics
    
    def calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not pnls:
            return 0
        
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return np.max(drawdowns) if len(drawdowns) > 0 else 0

def load_nifty_data(daily_csv_path: str, hourly_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare NIFTY data from CSV files
    
    Args:
        daily_csv_path: Path to daily data CSV (NIFTY50_INDEX_D_Min.csv)
        hourly_csv_path: Path to 60-min data CSV (NIFTY50_INDEX_60_Min.csv)
    
    Returns:
        Tuple of (daily_data, hourly_data) DataFrames
    """
    
    # Load daily data
    daily_df = pd.read_csv(daily_csv_path)
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df.set_index('Date', inplace=True)
    daily_df.columns = daily_df.columns.str.lower()  # Convert to lowercase
    
    # Load hourly data
    hourly_df = pd.read_csv(hourly_csv_path)
    hourly_df['Date'] = pd.to_datetime(hourly_df['Date'])
    hourly_df.set_index('Date', inplace=True)
    hourly_df.columns = hourly_df.columns.str.lower()  # Convert to lowercase
    
    # Remove any rows with missing data
    daily_df = daily_df.dropna()
    hourly_df = hourly_df.dropna()
    
    # Sort by date to ensure proper order
    daily_df = daily_df.sort_index()
    hourly_df = hourly_df.sort_index()
    
    print(f"Loaded daily data: {len(daily_df)} rows from {daily_df.index[0]} to {daily_df.index[-1]}")
    print(f"Loaded hourly data: {len(hourly_df)} rows from {hourly_df.index[0]} to {hourly_df.index[-1]}")
    
    return daily_df, hourly_df

def load_and_prepare_data():
    """
    Example function showing how to load your NIFTY CSV files
    """
    # Replace these paths with your actual file paths
    daily_csv_path = "NIFTY50_INDEX_D_Min.csv"
    hourly_csv_path = "NIFTY50_INDEX_60_Min.csv"
    
    try:
        daily_data, hourly_data = load_nifty_data(daily_csv_path, hourly_csv_path)
        return daily_data, hourly_data
    except FileNotFoundError as e:
        print(f"CSV file not found: {e}")
        print("Please ensure the CSV files are in the correct directory")
        print("Expected files: NIFTY50_INDEX_D_Min.csv, NIFTY50_INDEX_60_Min.csv")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def run_strategy_analysis():
    """
    Complete strategy analysis with detailed reporting
    """
    # Initialize strategy
    strategy = MultiTimeframeStrategy()
    
    # Load your data
    daily_data, hourly_data = load_and_prepare_data()
    
    if daily_data is None or hourly_data is None:
        print("Failed to load data. Please check your CSV files.")
        return
    
    print("\n" + "="*60)
    print("NIFTY MULTI-TIMEFRAME TRADING STRATEGY ANALYSIS")
    print("="*60)
    
    # Display data summary
    print(f"\nData Summary:")
    print(f"Daily data period: {daily_data.index[0].date()} to {daily_data.index[-1].date()}")
    print(f"Hourly data period: {hourly_data.index[0].date()} to {hourly_data.index[-1].date()}")
    print(f"Daily records: {len(daily_data)}")
    print(f"Hourly records: {len(hourly_data)}")
    
    # Run backtest
    print(f"\nRunning backtest...")
    results = strategy.backtest_strategy(daily_data, hourly_data)
    
    # Print detailed results
    print("\n" + "="*50)
    print("STRATEGY PERFORMANCE RESULTS")
    print("="*50)
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Losing Trades: {results['losing_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Total P&L: {results['total_pnl']:.2f} points")
    print(f"Average Win: {results['avg_win']:.2f} points")
    print(f"Average Loss: {results['avg_loss']:.2f} points")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f} points")
    
    # Show recent trades
    if results['trades']:
        print(f"\n" + "="*50)
        print("RECENT TRADES (Last 5)")
        print("="*50)
        recent_trades = results['trades'][-5:] if len(results['trades']) >= 5 else results['trades']
        
        for i, trade in enumerate(recent_trades, 1):
            print(f"\nTrade {i}:")
            print(f"  Type: {trade['entry_type']}")
            print(f"  Entry: {trade['entry_time']} at {trade['entry_price']:.2f}")
            if trade['exit_time']:
                print(f"  Exit: {trade['exit_time']} at {trade['exit_price']:.2f}")
                print(f"  P&L: {trade['pnl']:.2f} points")
                print(f"  Status: {trade['status']}")
            else:
                print(f"  Status: {trade['status']}")
    
    return results

def optimize_parameters():
    """
    Example function for parameter optimization
    You can expand this to test different parameter combinations
    """
    print("\n" + "="*50)
    print("PARAMETER OPTIMIZATION EXAMPLE")
    print("="*50)
    
    # Load data once
    daily_data, hourly_data = load_and_prepare_data()
    if daily_data is None or hourly_data is None:
        return
    
    # Test different ATR multipliers
    atr_multipliers = [0.3, 0.4, 0.5, 0.6, 0.7]
    exit_brick_counts = [2, 3, 4, 5]
    
    best_profit_factor = 0
    best_params = {}
    results_summary = []
    
    print("Testing different parameter combinations...")
    
    for atr_mult in atr_multipliers:
        for exit_count in exit_brick_counts:
            # Create strategy with modified parameters
            strategy = MultiTimeframeStrategy()
            strategy.atr_multiplier = atr_mult
            strategy.exit_consecutive_bricks = exit_count
            
            # Run backtest
            results = strategy.backtest_strategy(daily_data, hourly_data)
            
            # Store results
            result_summary = {
                'atr_multiplier': atr_mult,
                'exit_bricks': exit_count,
                'total_trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'profit_factor': results['profit_factor'],
                'total_pnl': results['total_pnl']
            }
            results_summary.append(result_summary)
            
            # Track best performing parameters
            if results['profit_factor'] > best_profit_factor and results['total_trades'] > 10:
                best_profit_factor = results['profit_factor']
                best_params = {
                    'atr_multiplier': atr_mult,
                    'exit_bricks': exit_count,
                    'performance': results
                }
            
            print(f"ATR: {atr_mult}, Exit: {exit_count} bricks -> "
                  f"Trades: {results['total_trades']}, "
                  f"Win Rate: {results['win_rate']:.1f}%, "
                  f"PF: {results['profit_factor']:.2f}")
    
    # Display best parameters
    if best_params:
        print(f"\n" + "="*40)
        print("BEST PARAMETERS FOUND")
        print("="*40)
        print(f"ATR Multiplier: {best_params['atr_multiplier']}")
        print(f"Exit Brick Count: {best_params['exit_bricks']}")
        print(f"Profit Factor: {best_params['performance']['profit_factor']:.2f}")
        print(f"Win Rate: {best_params['performance']['win_rate']:.2f}%")
        print(f"Total P&L: {best_params['performance']['total_pnl']:.2f}")

# Example usage
if __name__ == "__main__":
    # Run main strategy analysis
    results = run_strategy_analysis()
    
    # Optional: Run parameter optimization
    # Uncomment the line below to run optimization
    # optimize_parameters()
