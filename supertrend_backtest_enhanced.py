import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from numba import jit
import os

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Strategy Configuration
INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.02  # Risk exactly 2% of current capital per trade
TRANSACTION_COSTS = 0.001  # 0.1% transaction costs (bid-ask spread + commissions)
MIN_POSITION_SIZE = 0.01  # Minimum position size in lots/contracts
MAX_POSITION_SIZE = 1.0   # Maximum position size in lots/contracts

# Supertrend Parameters
DEFAULT_ST_PERIOD = 10
DEFAULT_ST_MULTIPLIER = 3.0

# Data Configuration
CSV_FILE_PATH = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'

# Output Configuration
RESULTS_DIR = 'backtest_results'
TRADE_LOG_FILE = 'supertrend_trade_log.csv'
PERFORMANCE_REPORT_FILE = 'supertrend_performance_report.txt'

# =============================================================================
# SUPER TREND CALCULATION FUNCTIONS
# =============================================================================

@jit(nopython=True)
def calculate_atr_numba(high, low, close, period):
    """Calculate ATR using Numba for performance"""
    tr = np.zeros_like(high)
    atr = np.zeros_like(high)

    # Calculate True Range
    for i in range(len(high)):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

    # Calculate ATR using Wilder's smoothing
    if len(high) > 0:
        atr[0] = tr[0]
        alpha = 1.0 / period

        for i in range(1, len(high)):
            atr[i] = (tr[i] * alpha) + (atr[i-1] * (1 - alpha))

    return atr

@jit(nopython=True)
def calculate_supertrend_numba(high, low, close, atr, period, multiplier):
    """Calculate SuperTrend using Numba for performance"""
    upper_band = np.zeros_like(high)
    lower_band = np.zeros_like(high)
    supertrend = np.zeros_like(high)
    direction = np.zeros_like(high, dtype=np.int32)

    # Calculate bands
    for i in range(len(high)):
        hl_avg = (high[i] + low[i]) / 2
        upper_band[i] = hl_avg + (multiplier * atr[i])
        lower_band[i] = hl_avg - (multiplier * atr[i])

    # Initialize SuperTrend
    if len(high) > period:
        # Find first valid ATR index
        first_valid_idx = 0
        for i in range(len(atr)):
            if not np.isnan(atr[i]):
                first_valid_idx = i
                break

        if first_valid_idx < len(high):
            if close[first_valid_idx] <= upper_band[first_valid_idx]:
                direction[first_valid_idx] = 1  # Bullish
                supertrend[first_valid_idx] = upper_band[first_valid_idx]
            else:
                direction[first_valid_idx] = -1  # Bearish
                supertrend[first_valid_idx] = lower_band[first_valid_idx]

            # Calculate subsequent values
            for i in range(first_valid_idx + 1, len(high)):
                prev_direction = direction[i-1]

                if prev_direction == 1:  # Previous bullish
                    if close[i] < supertrend[i-1]:
                        direction[i] = -1
                        supertrend[i] = lower_band[i]
                    else:
                        direction[i] = 1
                        supertrend[i] = min(upper_band[i], supertrend[i-1])
                else:  # Previous bearish
                    if close[i] > supertrend[i-1]:
                        direction[i] = 1
                        supertrend[i] = upper_band[i]
                    else:
                        direction[i] = -1
                        supertrend[i] = max(lower_band[i], supertrend[i-1])

    return upper_band, lower_band, supertrend, direction

def calculate_supertrend(df, period=DEFAULT_ST_PERIOD, multiplier=DEFAULT_ST_MULTIPLIER):
    """
    Calculate SuperTrend indicator

    Parameters:
    df (pd.DataFrame): DataFrame with OHLC columns
    period (int): ATR period
    multiplier (float): ATR multiplier

    Returns:
    pd.DataFrame: DataFrame with SuperTrend columns
    """
    if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain OHLC columns")

    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values

    # Calculate ATR
    atr = calculate_atr_numba(high, low, close, period)

    # Calculate SuperTrend
    upper_band, lower_band, supertrend, direction = calculate_supertrend_numba(
        high, low, close, atr, period, multiplier
    )

    # Create result DataFrame
    result_df = pd.DataFrame({
        'SuperTrend': supertrend,
        'ST_Direction': direction,
        'ST_Upper': upper_band,
        'ST_Lower': lower_band,
        'ATR': atr
    }, index=df.index)

    return result_df

# =============================================================================
# POSITION SIZING AND RISK MANAGEMENT
# =============================================================================

class PositionSizer:
    """Handle position sizing based on risk management"""

    def __init__(self, initial_capital, risk_per_trade=0.02):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.current_capital = initial_capital

    def calculate_position_size(self, entry_price, stop_loss_price, current_price=None):
        """
        Calculate position size based on risk management

        Parameters:
        entry_price (float): Entry price for the trade
        stop_loss_price (float): Stop loss price
        current_price (float): Current market price (for market orders)

        Returns:
        dict: Position sizing information
        """
        if current_price is None:
            current_price = entry_price

        # Calculate risk amount (2% of current capital)
        risk_amount = self.current_capital * self.risk_per_trade

        # Calculate stop loss distance in price terms
        if entry_price > stop_loss_price:  # Long position
            stop_distance = entry_price - stop_loss_price
            direction = 1
        else:  # Short position
            stop_distance = stop_loss_price - entry_price
            direction = -1

        if stop_distance <= 0:
            return {
                'position_size': 0,
                'risk_amount': 0,
                'stop_distance': 0,
                'direction': 0,
                'reason': 'Invalid stop loss distance'
            }

        # Calculate position size based on risk
        # For futures/options, this would be in lots/contracts
        # For stocks, this would be in shares
        position_size = risk_amount / stop_distance

        # Apply position size limits
        position_size = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, position_size))

        # Adjust risk amount based on actual position size
        actual_risk = position_size * stop_distance

        return {
            'position_size': position_size,
            'risk_amount': actual_risk,
            'stop_distance': stop_distance,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'current_capital': self.current_capital
        }

    def update_capital(self, pnl):
        """Update current capital after trade closure"""
        self.current_capital += pnl

    def get_current_capital(self):
        """Get current capital"""
        return self.current_capital

# =============================================================================
# TRADING STRATEGY ENGINE
# =============================================================================

class SupertrendStrategy:
    """SuperTrend trading strategy with risk management"""

    def __init__(self, initial_capital=INITIAL_CAPITAL, risk_per_trade=RISK_PER_TRADE,
                 st_period=DEFAULT_ST_PERIOD, st_multiplier=DEFAULT_ST_MULTIPLIER):
        self.position_sizer = PositionSizer(initial_capital, risk_per_trade)
        self.st_period = st_period
        self.st_multiplier = st_multiplier

        # Strategy state
        self.current_position = 0  # 0 = no position, 1 = long, -1 = short
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.trades = []

    def generate_signals(self, df):
        """
        Generate trading signals based on SuperTrend

        Parameters:
        df (pd.DataFrame): Price data with SuperTrend indicators

        Returns:
        pd.DataFrame: DataFrame with trading signals
        """
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
        signals['supertrend'] = df['SuperTrend']
        signals['direction'] = df['ST_Direction']

        # Generate signals based on SuperTrend direction changes
        for i in range(1, len(df)):
            current_direction = df['ST_Direction'].iloc[i]
            previous_direction = df['ST_Direction'].iloc[i-1]

            # Bullish signal: direction changes from bearish to bullish
            if previous_direction == -1 and current_direction == 1:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1

            # Bearish signal: direction changes from bullish to bearish
            elif previous_direction == 1 and current_direction == -1:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1

        return signals

    def execute_trade(self, date, price, signal, current_capital):
        """
        Execute a trade based on signal

        Parameters:
        date: Trade date
        price: Current price
        signal: Trading signal (1 = buy, -1 = sell, 0 = hold)
        current_capital: Current available capital

        Returns:
        dict: Trade information or None if no trade executed
        """
        trade_executed = False
        trade_info = None

        if signal == 1 and self.current_position != 1:  # Buy signal and not already long
            # Calculate stop loss for long position (below SuperTrend)
            stop_loss = price * 0.98  # 2% stop loss for long positions

            # Calculate position size
            position_info = self.position_sizer.calculate_position_size(price, stop_loss)

            if position_info['position_size'] > 0:
                # Check if we have enough capital
                capital_required = position_info['position_size'] * price

                if capital_required <= current_capital:
                    # Execute long position
                    self.current_position = 1
                    self.position_size = position_info['position_size']
                    self.entry_price = price
                    self.stop_loss = stop_loss

                    trade_info = {
                        'date': date,
                        'type': 'BUY',
                        'price': price,
                        'quantity': self.position_size,
                        'amount': capital_required,
                        'stop_loss': stop_loss,
                        'capital_before': current_capital,
                        'capital_after': current_capital - capital_required
                    }
                    trade_executed = True

        elif signal == -1 and self.current_position != -1:  # Sell signal and not already short
            # For short positions, we need to handle differently
            # This is a simplified version - in practice, short selling requires margin
            if self.current_position == 1:  # Close long position first
                # Close long position at current price
                pnl = (price - self.entry_price) * self.position_size
                capital_return = self.position_size * price

                trade_info = {
                    'date': date,
                    'type': 'SELL_CLOSE',
                    'price': price,
                    'quantity': self.position_size,
                    'amount': capital_return,
                    'pnl': pnl,
                    'capital_before': current_capital,
                    'capital_after': current_capital + capital_return + pnl
                }

                # Update position sizer capital
                self.position_sizer.update_capital(pnl)

                # Reset position
                self.current_position = 0
                self.position_size = 0
                self.entry_price = 0
                self.stop_loss = 0

                trade_executed = True

        return trade_info

    def check_stop_loss(self, date, price, current_capital):
        """
        Check if stop loss is hit

        Parameters:
        date: Current date
        price: Current price
        current_capital: Current available capital

        Returns:
        dict: Stop loss trade information or None
        """
        if self.current_position == 1 and price <= self.stop_loss:
            # Stop loss hit for long position
            pnl = (price - self.entry_price) * self.position_size
            capital_return = self.position_size * price

            trade_info = {
                'date': date,
                'type': 'STOP_LOSS',
                'price': price,
                'quantity': self.position_size,
                'amount': capital_return,
                'pnl': pnl,
                'stop_loss': self.stop_loss,
                'capital_before': current_capital,
                'capital_after': current_capital + capital_return + pnl
            }

            # Update position sizer capital
            self.position_sizer.update_capital(pnl)

            # Reset position
            self.current_position = 0
            self.position_size = 0
            self.entry_price = 0
            self.stop_loss = 0

            return trade_info

        return None

# =============================================================================
# PERFORMANCE CALCULATION
# =============================================================================

def calculate_performance_metrics(trades_df, initial_capital=INITIAL_CAPITAL):
    """
    Calculate comprehensive performance metrics

    Parameters:
    trades_df (pd.DataFrame): DataFrame with trade information
    initial_capital (float): Initial capital

    Returns:
    dict: Performance metrics
    """
    if trades_df.empty:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'annualized_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0
        }

    # Basic metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])

    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    # P&L metrics
    total_pnl = trades_df['pnl'].sum()
    total_return = (total_pnl / initial_capital) * 100

    winning_pnl = trades_df[trades_df['pnl'] > 0]['pnl']
    losing_pnl = trades_df[trades_df['pnl'] < 0]['pnl']

    avg_win = winning_pnl.mean() if len(winning_pnl) > 0 else 0
    avg_loss = abs(losing_pnl.mean()) if len(losing_pnl) > 0 else 0
    largest_win = winning_pnl.max() if len(winning_pnl) > 0 else 0
    largest_loss = abs(losing_pnl.min()) if len(losing_pnl) > 0 else 0

    # Profit factor
    gross_profit = winning_pnl.sum() if len(winning_pnl) > 0 else 0
    gross_loss = abs(losing_pnl.sum()) if len(losing_pnl) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Time-based metrics
    if not trades_df.empty:
        start_date = trades_df['date'].min()
        end_date = trades_df['date'].max()

        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        days_held = (end_date - start_date).days

        if days_held > 0:
            # Annualized return
            years_held = days_held / 365.25
            annualized_return = ((1 + total_return / 100) ** (1 / years_held) - 1) * 100

            # Daily returns for Sharpe ratio
            daily_returns = trades_df.set_index('date')['pnl'].resample('D').sum().fillna(0)
            daily_returns_pct = daily_returns / initial_capital

            if len(daily_returns_pct) > 1:
                avg_daily_return = daily_returns_pct.mean()
                std_daily_return = daily_returns_pct.std()

                # Sharpe ratio (assuming 0% risk-free rate)
                sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
            else:
                sharpe_ratio = 0
                annualized_return = 0
        else:
            annualized_return = 0
            sharpe_ratio = 0
    else:
        annualized_return = 0
        sharpe_ratio = 0

    # Maximum drawdown
    if not trades_df.empty:
        cumulative_returns = trades_df['pnl'].cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / initial_capital * 100
        max_drawdown = drawdowns.min() if not drawdowns.empty else 0
    else:
        max_drawdown = 0

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': round(win_rate, 2),
        'total_return': round(total_return, 2),
        'annualized_return': round(annualized_return, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown': round(max_drawdown, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'largest_win': round(largest_win, 2),
        'largest_loss': round(largest_loss, 2)
    }

# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def run_backtest(df, st_period=DEFAULT_ST_PERIOD, st_multiplier=DEFAULT_ST_MULTIPLIER,
                initial_capital=INITIAL_CAPITAL, risk_per_trade=RISK_PER_TRADE):
    """
    Run SuperTrend strategy backtest

    Parameters:
    df (pd.DataFrame): OHLC price data
    st_period (int): SuperTrend period
    st_multiplier (float): SuperTrend multiplier
    initial_capital (float): Initial capital
    risk_per_trade (float): Risk per trade (as decimal)

    Returns:
    tuple: (trades_df, performance_metrics, equity_curve)
    """
    print(f"Running SuperTrend backtest with period={st_period}, multiplier={st_multiplier}")

    # Calculate SuperTrend
    st_data = calculate_supertrend(df, st_period, st_multiplier)

    # Combine with original data
    df_combined = pd.concat([df, st_data], axis=1)

    # Initialize strategy
    strategy = SupertrendStrategy(initial_capital, risk_per_trade, st_period, st_multiplier)

    # Generate signals
    signals = strategy.generate_signals(df_combined)

    # Run backtest
    trades = []
    current_capital = initial_capital

    for i in range(len(df_combined)):
        date = df_combined.index[i]
        price = df_combined['Close'].iloc[i]
        signal = signals['signal'].iloc[i]

        # Check for stop loss first
        stop_loss_trade = strategy.check_stop_loss(date, price, current_capital)
        if stop_loss_trade:
            trades.append(stop_loss_trade)
            current_capital = stop_loss_trade['capital_after']

        # Execute signal if no stop loss triggered
        if not stop_loss_trade:
            trade_info = strategy.execute_trade(date, price, signal, current_capital)
            if trade_info:
                trades.append(trade_info)
                current_capital = trade_info['capital_after']

    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        # Calculate performance metrics
        metrics = calculate_performance_metrics(trades_df, initial_capital)

        # Calculate equity curve
        equity_curve = pd.DataFrame(index=trades_df['date'].unique())
        equity_curve['capital'] = initial_capital

        running_capital = initial_capital
        for _, trade in trades_df.iterrows():
            running_capital = trade['capital_after']
            equity_curve.loc[trade['date'], 'capital'] = running_capital

        # Forward fill equity curve
        equity_curve = equity_curve.fillna(method='ffill')
        equity_curve['returns'] = equity_curve['capital'].pct_change()
    else:
        metrics = calculate_performance_metrics(trades_df, initial_capital)
        equity_curve = pd.DataFrame({'capital': [initial_capital], 'returns': [0]})

    return trades_df, metrics, equity_curve

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_data(csv_path=None, use_sample_data=False):
    """
    Load OHLC data from CSV or create sample data

    Parameters:
    csv_path (str): Path to CSV file
    use_sample_data (bool): Whether to use sample data

    Returns:
    pd.DataFrame: OHLC data
    """
    if use_sample_data:
        print("Using sample data for backtesting...")
        return create_sample_data()

    try:
        if csv_path is None:
            csv_path = CSV_FILE_PATH

        df = pd.read_csv(csv_path)
        df = df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
        df = df.rename(columns={'Date': 'DateTime'})

        # Parse datetime
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
        df = df.set_index('DateTime')
        df = df[~df.index.duplicated(keep='first')]

        # Sort by date
        df = df.sort_index()

        # Round price columns
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = df[col].round(2)

        print(f"Loaded data from {csv_path}")
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    except FileNotFoundError:
        print(f"File '{csv_path}' not found. Using sample data instead.")
        return create_sample_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using sample data instead.")
        return create_sample_data()

def create_sample_data():
    """Create sample OHLC data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    # Generate trending price data
    np.random.seed(42)
    n = len(dates)

    # Create a trending series with some volatility
    trend = np.linspace(100, 150, n)
    noise = np.random.normal(0, 2, n)
    close_prices = trend + noise

    # Generate OHLC from close prices
    high_prices = close_prices + np.abs(np.random.normal(0, 1, n))
    low_prices = close_prices - np.abs(np.random.normal(0, 1, n))
    open_prices = np.random.uniform(low_prices, high_prices)

    sample_df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices
    }, index=dates)

    return sample_df

# =============================================================================
# OUTPUT AND REPORTING
# =============================================================================

def save_results(trades_df, metrics, equity_curve, st_period, st_multiplier, output_dir=RESULTS_DIR):
    """
    Save backtest results to files

    Parameters:
    trades_df (pd.DataFrame): Trade log
    metrics (dict): Performance metrics
    equity_curve (pd.DataFrame): Equity curve data
    st_period (int): SuperTrend period used
    st_multiplier (float): SuperTrend multiplier used
    output_dir (str): Output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save trade log
    if not trades_df.empty:
        trade_log_path = os.path.join(output_dir, TRADE_LOG_FILE)
        trades_df.to_csv(trade_log_path, index=False)
        print(f"Trade log saved to: {trade_log_path}")

    # Save performance report
    report_path = os.path.join(output_dir, PERFORMANCE_REPORT_FILE)
    with open(report_path, 'w') as f:
        f.write("SUPER TREND STRATEGY BACKTEST REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Strategy Parameters:\n")
        f.write(f"SuperTrend Period: {st_period}\n")
        f.write(f"SuperTrend Multiplier: {st_multiplier}\n")
        f.write(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}\n")
        f.write(f"Risk per Trade: {RISK_PER_TRADE:.1%}\n")
        f.write(f"Transaction Costs: {TRANSACTION_COSTS:.2%}\n\n")

        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")

        for key, value in metrics.items():
            if isinstance(value, float):
                if 'rate' in key.lower() or 'return' in key.lower() or 'drawdown' in key.lower():
                    f.write(f"{key.replace('_', ' ').title()}: {value:.2f}%\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: ${value:,.2f}\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")

        f.write(f"\nTotal Trades: {metrics.get('total_trades', 0)}\n")
        f.write(f"Winning Trades: {metrics.get('winning_trades', 0)}\n")
        f.write(f"Losing Trades: {metrics.get('losing_trades', 0)}\n")

    print(f"Performance report saved to: {report_path}")

    # Save equity curve
    if not equity_curve.empty:
        equity_path = os.path.join(output_dir, f'equity_curve_st_{st_period}_{st_multiplier}.csv')
        equity_curve.to_csv(equity_path)
        print(f"Equity curve saved to: {equity_path}")

def print_results(trades_df, metrics):
    """Print backtest results to console"""
    print("\n" + "="*60)
    print("SUPER TREND STRATEGY BACKTEST RESULTS")
    print("="*60)

    print("\nPERFORMANCE METRICS:")
    print("-" * 30)

    for key, value in metrics.items():
        if isinstance(value, float):
            if 'rate' in key.lower() or 'return' in key.lower() or 'drawdown' in key.lower():
                print(f"{key.replace('_', ' ').title()}: {value:>.2f}%")
            else:
                print(f"{key.replace('_', ' ').title()}: ${value:>,.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")

    print(f"\nTotal Trades: {metrics.get('total_trades', 0)}")
    print(f"Winning Trades: {metrics.get('winning_trades', 0)}")
    print(f"Losing Trades: {metrics.get('losing_trades', 0)}")

    if not trades_df.empty:
        print("\nTRADE SUMMARY:")
        print("-" * 30)
        print(f"Average Win: ${metrics.get('avg_win', 0):.2f}")
        print(f"Average Loss: ${metrics.get('avg_loss', 0):.2f}")
        print(f"Largest Win: ${metrics.get('largest_win', 0):.2f}")
        print(f"Largest Loss: ${metrics.get('largest_loss', 0):.2f}")
    print("="*60)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("SuperTrend Backtest with Risk Management")
    print("=" * 50)

    # Load data
    df = load_data(use_sample_data=False)

    if df.empty:
        print("No data available for backtesting")
        return

    # Run backtest with default parameters
    trades_df, metrics, equity_curve = run_backtest(
        df,
        st_period=DEFAULT_ST_PERIOD,
        st_multiplier=DEFAULT_ST_MULTIPLIER,
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE
    )

    # Display results
    print_results(trades_df, metrics)

    # Save results
    save_results(trades_df, metrics, equity_curve, DEFAULT_ST_PERIOD, DEFAULT_ST_MULTIPLIER)

    # Test different parameter combinations
    print("\nTesting different SuperTrend parameters...")

    parameter_tests = [
        (7, 2.5),
        (10, 3.0),
        (14, 3.5),
        (20, 4.0)
    ]

    results_summary = []

    for period, multiplier in parameter_tests:
        print(f"\nTesting Period={period}, Multiplier={multiplier}")

        test_trades, test_metrics, _ = run_backtest(
            df, period, multiplier, INITIAL_CAPITAL, RISK_PER_TRADE
        )

        results_summary.append({
            'period': period,
            'multiplier': multiplier,
            **test_metrics
        })

        print(f"  Total Return: {test_metrics.get('total_return', 0):.2f}%")
        print(f"  Win Rate: {test_metrics.get('win_rate', 0):.2f}%")
        print(f"  Max Drawdown: {test_metrics.get('max_drawdown', 0):.2f}%")
        print(f"  Sharpe Ratio: {test_metrics.get('sharpe_ratio', 0):.2f}")
    # Save parameter test results
    if results_summary:
        param_results_df = pd.DataFrame(results_summary)
        param_results_path = os.path.join(RESULTS_DIR, 'parameter_test_results.csv')
        param_results_df.to_csv(param_results_path, index=False)
        print(f"\nParameter test results saved to: {param_results_path}")

        # Find best parameters
        best_params = param_results_df.loc[param_results_df['total_return'].idxmax()]
        print("
BEST PARAMETERS:"        print(f"Period: {best_params['period']}")
        print(f"Multiplier: {best_params['multiplier']}")
        print(f"Total Return: {best_params['total_return']:.2f}%")
        print(f"Win Rate: {best_params['win_rate']:.2f}%")
        print(f"Max Drawdown: {best_params['max_drawdown']:.2f}%")
    print("
Backtest completed!")

if __name__ == "__main__":
    main()