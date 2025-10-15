import pandas as pd
import numpy as np
from numba import jit

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Data Configuration
CSV_FILE_PATH = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'

# Supertrend Parameters
ST_PERIOD = 25
ST_MULTIPLIER = 1.2

# Strategy Configuration
INITIAL_CAPITAL = 100000
RISK_PER_TRADE_PERCENT = 2.0  # Risk 2% of capital per trade
START_YEAR = 2019

# Position Sizing Configuration
STOP_LOSS_ATR_MULTIPLIER = 2.0  # Use 1.5 * ATR as stop loss
COMMISSION_PER_TRADE = 40  # Commission per trade (entry + exit)

# Partial Profit Booking Configuration
PROFIT_BOOKING_LEVELS = [
    {'capital_percent': 1.0, 'profit_percent': 40},  # Book 40% at 1% capital gain
    {'capital_percent': 2.0, 'profit_percent': 40},  # Book another 40% at 2% capital gain
]
REMAINING_PERCENT = 20  # Keep 20% until exit signal

# Data Processing
COLUMNS_TO_ROUND = ['Open', 'High', 'Low', 'Close']
DECIMAL_PLACES = 2

# Output Configuration
RESULTS_CSV_PATH = 'ohlc_supertrend_position_sizing_with_partial_profit_results.csv'

@jit(nopython=True)
def _calculate_supertrend_numba(high, low, close, tr, atr, period, multiplier, basic_upper_band, basic_lower_band, first_valid_atr_idx):
    """
    Calculate SuperTrend using OHLC data instead of Heikin-Ashi
    """
    trending_up = np.empty_like(close)
    trending_down = np.empty_like(close)
    direction = np.empty_like(close, dtype=np.int32)
    supertrend = np.empty_like(close)

    if first_valid_atr_idx is None or first_valid_atr_idx >= len(close):
        return trending_up, trending_down, direction, supertrend

    trending_up[first_valid_atr_idx] = basic_lower_band[first_valid_atr_idx]
    trending_down[first_valid_atr_idx] = basic_upper_band[first_valid_atr_idx]

    if close[first_valid_atr_idx] > trending_down[first_valid_atr_idx]:
        direction[first_valid_atr_idx] = 1
    elif close[first_valid_atr_idx] < trending_up[first_valid_atr_idx]:
        direction[first_valid_atr_idx] = -1
    else:
        direction[first_valid_atr_idx] = 1

    supertrend[first_valid_atr_idx] = trending_up[first_valid_atr_idx] if direction[first_valid_atr_idx] == 1 else trending_down[first_valid_atr_idx]

    for i in range(first_valid_atr_idx + 1, len(close)):
        prev_close = close[i - 1]
        prev_trending_up = trending_up[i - 1]
        prev_trending_down = trending_down[i - 1]
        prev_direction = direction[i - 1]

        current_basic_upper_band = basic_upper_band[i]
        current_basic_lower_band = basic_lower_band[i]
        current_close = close[i]

        if prev_close > prev_trending_up:
            trending_up[i] = max(current_basic_lower_band, prev_trending_up)
        else:
            trending_up[i] = current_basic_lower_band

        if prev_close < prev_trending_down:
            trending_down[i] = min(current_basic_upper_band, prev_trending_down)
        else:
            trending_down[i] = current_basic_upper_band

        if current_close > trending_down[i - 1]:
            direction[i] = 1
        elif current_close < trending_up[i - 1]:
            direction[i] = -1
        else:
            direction[i] = prev_direction

        supertrend[i] = trending_up[i] if direction[i] == 1 else trending_down[i]

    return trending_up, trending_down, direction, supertrend

def calculate_supertrend(df, period=10, multiplier=3.0, suffix=""):
    """
    Calculate standard SuperTrend indicator using OHLC data
    """
    original_index = df.index
    df_reset = df.reset_index(drop=True)

    if not all(col in df_reset.columns for col in ["Open", "High", "Low", "Close"]):
        raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns for Supertrend calculation.")

    high = df_reset["High"].to_numpy()
    low = df_reset["Low"].to_numpy()
    close = df_reset["Close"].to_numpy()

    # Calculate True Range based on OHLC candles
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # Correct TR for the first element

    # ATR using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]  # Initialize first ATR
    alpha = 1 / period
    for i in range(1, len(tr)):
        atr[i] = (tr[i] * alpha) + (atr[i-1] * (1 - alpha))

    # Calculate SuperTrend bands using OHLC high/low
    basic_upper_band = ((high + low) / 2) + (multiplier * atr)
    basic_lower_band = ((high + low) / 2) - (multiplier * atr)

    first_valid_atr_idx = np.where(~np.isnan(atr))[0][0] if np.any(~np.isnan(atr)) else -1

    trending_up_arr, trending_down_arr, direction_arr, supertrend_arr = _calculate_supertrend_numba(
        high, low, close, tr, atr, period, multiplier, basic_upper_band, basic_lower_band, first_valid_atr_idx
    )

    st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    trending_up_col_name = f"trendingUp_OHLC_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    trending_down_col_name = f"trendingDown_OHLC_{period}_{str(multiplier).replace('.', '_')}{suffix}"

    result_df = pd.DataFrame({
        st_col_name: supertrend_arr,
        st_dir_col_name: direction_arr,
        trending_up_col_name: trending_up_arr,
        trending_down_col_name: trending_down_arr
    }, index=original_index)

    return result_df

@jit(nopython=True)
def _run_strategy_numba(st_dir, close_prices):
    """
    Execute SuperTrend trading strategy
    """
    buy_entry_prices = np.full_like(close_prices, np.nan)
    buy_exit_prices = np.full_like(close_prices, np.nan)
    sell_entry_prices = np.full_like(close_prices, np.nan)
    sell_exit_prices = np.full_like(close_prices, np.nan)

    in_buy_position = False
    in_sell_position = False

    for i in range(1, len(close_prices)):
        # Check for Buy_Entry condition (Supertrend direction changes to up)
        if not in_buy_position and not in_sell_position and st_dir[i] == 1:
            buy_entry_prices[i] = close_prices[i]
            in_buy_position = True
        # Check for Buy_Exit condition (Supertrend direction changes to down)
        elif in_buy_position and st_dir[i] == -1:
            buy_exit_prices[i] = close_prices[i]
            in_buy_position = False

        # Check for Sell_Entry condition (Supertrend direction changes to down)
        if not in_sell_position and not in_buy_position and st_dir[i] == -1:
            sell_entry_prices[i] = close_prices[i]
            in_sell_position = True
        # Check for Sell_Exit condition (Supertrend direction changes to up)
        elif in_sell_position and st_dir[i] == 1:
            sell_exit_prices[i] = close_prices[i]
            in_sell_position = False

    return buy_entry_prices, buy_exit_prices, sell_entry_prices, sell_exit_prices

def calculate_position_size(entry_price, stop_loss_price, current_capital, risk_percent):
    """
    Calculate position size based on risk management

    Parameters:
    entry_price: Entry price for the trade
    stop_loss_price: Stop loss price
    current_capital: Current available capital
    risk_percent: Risk percentage per trade

    Returns:
    position_size: Number of shares/contracts to trade
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0

    # Calculate risk amount (2% of current capital)
    risk_amount = current_capital * (risk_percent / 100)

    # Calculate stop loss distance in price terms
    if entry_price > stop_loss_price:  # Long position
        stop_loss_distance = entry_price - stop_loss_price
    else:  # Short position
        stop_loss_distance = stop_loss_price - entry_price

    if stop_loss_distance <= 0:
        return 0

    # Calculate position size (risk amount / stop loss distance per share)
    position_size = risk_amount / stop_loss_distance

    return int(position_size)  # Round down to nearest integer

def run_strategy_with_position_sizing(df_original, supertrend_params_list):
    """
    Run SuperTrend strategy with proper position sizing and risk management
    """
    df = df_original.copy()

    # Use the first (and only) Supertrend parameter
    period, multiplier = supertrend_params_list[0]
    suffix = "_1"

    # Calculate OHLC Supertrend directly
    st_result_ohlc = calculate_supertrend(
        df.copy(),
        period=period,
        multiplier=multiplier,
        suffix=suffix
    )
    st_col_name = f"ST_{period}_{str(multiplier).replace('.', '_')}{suffix}"
    st_dir_col_name = f"ST_Dir_{period}_{str(multiplier).replace('.', '_')}{suffix}"

    df = pd.concat([df, st_result_ohlc], axis=1)

    # Convert direction column to numpy array for Numba
    st_dir = df[st_dir_col_name].to_numpy()
    close_prices = df['Close'].to_numpy()
    high_prices = df['High'].to_numpy()
    low_prices = df['Low'].to_numpy()

    buy_entry_prices, buy_exit_prices, sell_entry_prices, sell_exit_prices = _run_strategy_numba(
        st_dir, close_prices
    )

    df['Buy_Entry'] = buy_entry_prices
    df['Buy_Exit'] = buy_exit_prices
    df['Sell_Entry'] = sell_entry_prices
    df['Sell_Exit'] = sell_exit_prices

    # Initialize portfolio tracking
    current_capital = INITIAL_CAPITAL
    portfolio_values = []
    trades_log = []
    position_sizes = []

    current_position = None  # None, 'long', or 'short'
    entry_price = None
    position_size = 0
    original_position_size = 0  # Track original position size for profit calculations
    trade_start_date = None
    profit_booking_status = {}  # Track which profit levels have been booked

    for i in range(len(df)):
        current_date = df.index[i]
        current_close = close_prices[i]

        # Track portfolio value (cash + position value)
        if current_position == 'long':
            position_value = position_size * current_close
            total_value = current_capital + position_value
        elif current_position == 'short':
            position_value = position_size * (2 * entry_price - current_close)  # Short position value
            total_value = current_capital + position_value
        else:
            total_value = current_capital

        portfolio_values.append(total_value)
        position_sizes.append(position_size)

        # Check for trade entry
        if current_position is None:
            if not pd.isna(df['Buy_Entry'].iloc[i]):
                # Long entry
                entry_price = df['Buy_Entry'].iloc[i]

                # Calculate ATR for stop loss
                atr_period = min(period, i+1)  # Use available data
                if i >= atr_period:
                    recent_high = high_prices[max(0, i-atr_period+1):i+1]
                    recent_low = low_prices[max(0, i-atr_period+1):i+1]
                    recent_close = close_prices[max(0, i-atr_period+1):i+1]

                    tr = np.maximum(recent_high[1:] - recent_low[1:],
                                   np.abs(recent_high[1:] - np.roll(recent_close, 1)[1:]))
                    tr = np.maximum(tr, np.abs(recent_low[1:] - np.roll(recent_close, 1)[1:]))
                    atr = np.mean(tr) if len(tr) > 0 else (entry_price * 0.02)  # Fallback to 2% of price
                else:
                    atr = entry_price * 0.02  # Fallback for insufficient data

                stop_loss_price = entry_price - (STOP_LOSS_ATR_MULTIPLIER * atr)

                # Calculate position size
                position_size = calculate_position_size(
                    entry_price, stop_loss_price, current_capital, RISK_PER_TRADE_PERCENT
                )

                if position_size > 0:
                    current_position = 'long'
                    original_position_size = position_size
                    trade_start_date = current_date

                    # Initialize profit booking status for this position
                    profit_booking_status = {level['capital_percent']: False for level in PROFIT_BOOKING_LEVELS}

                    # Calculate cost and update capital
                    trade_cost = position_size * entry_price + COMMISSION_PER_TRADE
                    current_capital -= trade_cost

                    trades_log.append({
                        'type': 'long_entry',
                        'date': current_date,
                        'price': entry_price,
                        'size': position_size,
                        'original_size': original_position_size,
                        'capital_before': current_capital + trade_cost,
                        'capital_after': current_capital,
                        'stop_loss': stop_loss_price
                    })

            elif not pd.isna(df['Sell_Entry'].iloc[i]):
                # Short entry
                entry_price = df['Sell_Entry'].iloc[i]

                # Calculate ATR for stop loss
                atr_period = min(period, i+1)
                if i >= atr_period:
                    recent_high = high_prices[max(0, i-atr_period+1):i+1]
                    recent_low = low_prices[max(0, i-atr_period+1):i+1]
                    recent_close = close_prices[max(0, i-atr_period+1):i+1]

                    tr = np.maximum(recent_high[1:] - recent_low[1:],
                                   np.abs(recent_high[1:] - np.roll(recent_close, 1)[1:]))
                    tr = np.maximum(tr, np.abs(recent_low[1:] - np.roll(recent_close, 1)[1:]))
                    atr = np.mean(tr) if len(tr) > 0 else (entry_price * 0.02)
                else:
                    atr = entry_price * 0.02

                stop_loss_price = entry_price + (STOP_LOSS_ATR_MULTIPLIER * atr)

                # Calculate position size for short
                position_size = calculate_position_size(
                    entry_price, stop_loss_price, current_capital, RISK_PER_TRADE_PERCENT
                )

                if position_size > 0:
                    current_position = 'short'
                    original_position_size = position_size
                    trade_start_date = current_date

                    # Initialize profit booking status for this position
                    profit_booking_status = {level['capital_percent']: False for level in PROFIT_BOOKING_LEVELS}

                    # For short selling, we need to consider margin requirements
                    # Simplified: assume we can short with similar capital requirement
                    trade_cost = position_size * entry_price + COMMISSION_PER_TRADE
                    current_capital -= trade_cost

                    trades_log.append({
                        'type': 'short_entry',
                        'date': current_date,
                        'price': entry_price,
                        'size': position_size,
                        'original_size': original_position_size,
                        'capital_before': current_capital + trade_cost,
                        'capital_after': current_capital,
                        'stop_loss': stop_loss_price
                    })

        # Check for partial profit booking and trade exit (Long position)
        elif current_position == 'long':
            # Check for partial profit booking first
            if position_size > 0:
                # Calculate current unrealized P&L
                current_profit = position_size * (current_close - entry_price)
                profit_percent = (current_profit / current_capital) * 100

                # Check each profit booking level
                for level in PROFIT_BOOKING_LEVELS:
                    capital_target = level['capital_percent']
                    profit_target = level['profit_percent']

                    if profit_percent >= capital_target and not profit_booking_status[capital_target]:
                        # Calculate portion to book (e.g., 40% of current position)
                        portion_to_book = int(position_size * (profit_target / 100))
                        remaining_size = position_size - portion_to_book

                        if portion_to_book > 0:
                            # Book partial profit
                            gross_profit_booked = portion_to_book * (current_close - entry_price)
                            net_profit_booked = gross_profit_booked - (COMMISSION_PER_TRADE * 0.5)  # Half commission for partial exit
                            current_capital += (portion_to_book * current_close) + net_profit_booked

                            # Update position size
                            position_size = remaining_size

                            # Mark this level as booked
                            profit_booking_status[capital_target] = True

                            trades_log.append({
                                'type': 'long_partial_exit',
                                'date': current_date,
                                'price': current_close,
                                'size': portion_to_book,
                                'remaining_size': remaining_size,
                                'profit': net_profit_booked,
                                'profit_level': f"{profit_target}% at {capital_target}% capital gain",
                                'capital_before': current_capital - (portion_to_book * current_close),
                                'capital_after': current_capital
                            })

                            print(f"Long Partial Exit: Booked {profit_target}% at {capital_target}% capital gain")

            # Check for final exit (signal or trailing stop)
            if not pd.isna(df['Buy_Exit'].iloc[i]) or current_close <= entry_price * 0.95 or position_size == 0:
                if position_size > 0:  # Only if there's remaining position
                    exit_price = df['Buy_Exit'].iloc[i] if not pd.isna(df['Buy_Exit'].iloc[i]) else current_close

                    # Calculate P&L for remaining position
                    gross_profit = position_size * (exit_price - entry_price)
                    net_profit = gross_profit - (COMMISSION_PER_TRADE * 0.5)  # Half commission for final exit
                    current_capital += (position_size * exit_price) + net_profit

                    trades_log.append({
                        'type': 'long_exit',
                        'date': current_date,
                        'price': exit_price,
                        'size': position_size,
                        'profit': net_profit,
                        'capital_before': current_capital - (position_size * exit_price),
                        'capital_after': current_capital
                    })

                current_position = None
                position_size = 0
                original_position_size = 0
                profit_booking_status = {}

        # Check for partial profit booking and trade exit (Short position)
        elif current_position == 'short':
            # Check for partial profit booking first
            if position_size > 0:
                # Calculate current unrealized P&L for short position
                current_profit = position_size * (entry_price - current_close)  # Profit when price decreases
                profit_percent = (current_profit / current_capital) * 100

                # Check each profit booking level
                for level in PROFIT_BOOKING_LEVELS:
                    capital_target = level['capital_percent']
                    profit_target = level['profit_percent']

                    if profit_percent >= capital_target and not profit_booking_status[capital_target]:
                        # Calculate portion to book (e.g., 40% of current position)
                        portion_to_book = int(position_size * (profit_target / 100))
                        remaining_size = position_size - portion_to_book

                        if portion_to_book > 0:
                            # Book partial profit for short
                            gross_profit_booked = portion_to_book * (entry_price - current_close)
                            net_profit_booked = gross_profit_booked - (COMMISSION_PER_TRADE * 0.5)  # Half commission for partial exit
                            current_capital += (portion_to_book * entry_price) + net_profit_booked

                            # Update position size
                            position_size = remaining_size

                            # Mark this level as booked
                            profit_booking_status[capital_target] = True

                            trades_log.append({
                                'type': 'short_partial_exit',
                                'date': current_date,
                                'price': current_close,
                                'size': portion_to_book,
                                'remaining_size': remaining_size,
                                'profit': net_profit_booked,
                                'profit_level': f"{profit_target}% at {capital_target}% capital gain",
                                'capital_before': current_capital - (portion_to_book * entry_price),
                                'capital_after': current_capital
                            })

                            print(f"Short Partial Exit: Booked {profit_target}% at {capital_target}% capital gain")

            # Check for final exit (signal or trailing stop)
            if not pd.isna(df['Sell_Exit'].iloc[i]) or current_close >= entry_price * 1.05 or position_size == 0:
                if position_size > 0:  # Only if there's remaining position
                    exit_price = df['Sell_Exit'].iloc[i] if not pd.isna(df['Sell_Exit'].iloc[i]) else current_close

                    # Calculate P&L for remaining position (short)
                    gross_profit = position_size * (entry_price - exit_price)
                    net_profit = gross_profit - (COMMISSION_PER_TRADE * 0.5)  # Half commission for final exit
                    current_capital += (position_size * entry_price) + net_profit

                    trades_log.append({
                        'type': 'short_exit',
                        'date': current_date,
                        'price': exit_price,
                        'size': position_size,
                        'profit': net_profit,
                        'capital_before': current_capital - (position_size * entry_price),
                        'capital_after': current_capital
                    })

                current_position = None
                position_size = 0
                original_position_size = 0
                profit_booking_status = {}

    # Add portfolio tracking columns to dataframe
    df['Portfolio_Value'] = portfolio_values
    df['Position_Size'] = position_sizes
    df['Available_Capital'] = [INITIAL_CAPITAL] * len(df)  # This will be updated in the loop above

    # Update available capital column
    capital_values = [INITIAL_CAPITAL]
    for i in range(1, len(df)):
        if current_position == 'long':
            position_value = position_sizes[i] * close_prices[i]
            capital_values.append(current_capital + position_value)
        elif current_position == 'short':
            position_value = position_sizes[i] * (2 * entry_price - close_prices[i])
            capital_values.append(current_capital + position_value)
        else:
            capital_values.append(current_capital)

    df['Available_Capital'] = capital_values

    return df, trades_log

def calculate_comprehensive_metrics(trades_log, initial_capital):
    """
    Calculate comprehensive performance metrics
    """
    if not trades_log:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0,
            'total_profit': 0, 'avg_win': 0, 'avg_loss': 0, 'max_profit': 0, 'max_loss': 0,
            'profit_factor': 0, 'max_drawdown': 0, 'sharpe_ratio': 0, 'final_capital': initial_capital
        }

    # Convert trades log to DataFrame for easier analysis (include partial exits for analysis)
    trades_df = pd.DataFrame([trade for trade in trades_log if trade['type'] in ['long_exit', 'short_exit', 'long_partial_exit', 'short_partial_exit']])

    if trades_df.empty:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0,
            'total_profit': 0, 'avg_win': 0, 'avg_loss': 0, 'max_profit': 0, 'max_loss': 0,
            'profit_factor': 0, 'max_drawdown': 0, 'sharpe_ratio': 0, 'final_capital': initial_capital
        }

    # Calculate metrics
    winning_trades = trades_df[trades_df['profit'] > 0]
    losing_trades = trades_df[trades_df['profit'] < 0]

    total_trades = len(trades_df)
    winning_count = len(winning_trades)
    losing_count = len(losing_trades)

    win_rate = (winning_count / total_trades) * 100 if total_trades > 0 else 0

    total_profit = trades_df['profit'].sum()
    avg_win = winning_trades['profit'].mean() if winning_count > 0 else 0
    avg_loss = losing_trades['profit'].mean() if losing_count > 0 else 0

    max_profit = trades_df['profit'].max() if total_trades > 0 else 0
    max_loss = trades_df['profit'].min() if total_trades > 0 else 0

    gross_profit = winning_trades['profit'].sum() if winning_count > 0 else 0
    gross_loss = abs(losing_trades['profit'].sum()) if losing_count > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

    # Calculate drawdown
    capital_over_time = [initial_capital]
    running_capital = initial_capital

    for _, trade in trades_df.iterrows():
        running_capital += trade['profit']
        capital_over_time.append(running_capital)

    capital_series = pd.Series(capital_over_time)
    peak = capital_series.expanding(min_periods=1).max()
    drawdown = (capital_series - peak) / peak * 100
    max_drawdown = drawdown.min() if not drawdown.empty else 0

    # Calculate Sharpe ratio (simplified)
    if total_trades > 1:
        returns = trades_df['profit'].diff()
        sharpe_ratio = (trades_df['profit'].mean() / trades_df['profit'].std()) * np.sqrt(252) if trades_df['profit'].std() != 0 else 0
    else:
        sharpe_ratio = 0

    final_capital = initial_capital + total_profit

    return {
        'total_trades': total_trades,
        'winning_trades': winning_count,
        'losing_trades': losing_count,
        'win_rate': round(win_rate, 2),
        'total_profit': round(total_profit, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'max_profit': round(max_profit, 2),
        'max_loss': round(max_loss, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
        'max_drawdown': round(max_drawdown, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'final_capital': round(final_capital, 2)
    }

def load_data(csv_path=None, use_sample_data=False):
    """
    Load data from CSV file or create sample data
    """
    if use_sample_data:
        print("Using sample data for demonstration...")
        # Create sample OHLC data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        n = len(dates)

        trend = np.linspace(100, 150, n)
        noise = np.random.normal(0, 2, n)
        close_prices = trend + noise

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

    try:
        if csv_path is None:
            csv_path = CSV_FILE_PATH

        df_original = pd.read_csv(csv_path)
        df_original = df_original.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
        df_original = df_original.rename(columns={'Date': 'DateTime'})
        df_original = df_original.set_index('DateTime')
        df_original.index = pd.to_datetime(df_original.index)
        df_original = df_original[~df_original.index.duplicated(keep='first')]

        print(f"Loaded data from {csv_path}")
        print(f"Data shape: {df_original.shape}")
        print(f"Date range: {df_original.index.min()} to {df_original.index.max()}")

        return df_original

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        print("Using sample data instead...")
        return load_data(use_sample_data=True)
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        print("Using sample data instead...")
        return load_data(use_sample_data=True)

def main():
    print("=== OHLC Supertrend Strategy with Position Sizing ===")

    # Load data
    df_original = load_data(use_sample_data=False)

    if df_original.empty:
        print("No data available")
        return

    # Filter for start year
    if df_original.index.year.min() < START_YEAR:
        df_original = df_original[df_original.index.year >= START_YEAR]
        print(f"Using data from {START_YEAR} onwards: {df_original.shape}")

    # Round price columns
    for col in COLUMNS_TO_ROUND:
        if col in df_original.columns:
            df_original[col] = df_original[col].round(DECIMAL_PLACES)

    # Run strategy with position sizing
    print(f"\n=== Running Strategy with Position Sizing ===")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Risk per Trade: {RISK_PER_TRADE_PERCENT}%")
    print(f"Stop Loss ATR Multiplier: {STOP_LOSS_ATR_MULTIPLIER}")
    print(f"Commission per Trade: ${COMMISSION_PER_TRADE}")

    print(f"\n=== Partial Profit Booking Configuration ===")
    for level in PROFIT_BOOKING_LEVELS:
        print(f"Book {level['profit_percent']}% profit at {level['capital_percent']}% capital gain")
    print(f"Keep {REMAINING_PERCENT}% until exit signal")

    supertrend_params = [(ST_PERIOD, ST_MULTIPLIER)]
    df_with_results, trades_log = run_strategy_with_position_sizing(df_original.copy(), supertrend_params)

    # Calculate metrics
    metrics = calculate_comprehensive_metrics(trades_log, INITIAL_CAPITAL)

    # Display results
    print("=== Strategy Results ===")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Win Rate: {metrics['win_rate']}%")
    print(f"Total Profit: ${metrics['total_profit']:,.2f}")
    print(f"Final Capital: ${metrics['final_capital']:,.2f}")
    print(f"Return: {((metrics['final_capital'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100):.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']}")

    if metrics['winning_trades'] > 0:
        print(f"Average Win: ${metrics['avg_win']:,.2f}")
    if metrics['losing_trades'] > 0:
        print(f"Average Loss: ${metrics['avg_loss']:,.2f}")
    if metrics['total_trades'] > 0:
        print(f"Max Profit: ${metrics['max_profit']:,.2f}")
        print(f"Max Loss: ${metrics['max_loss']:,.2f}")

    # Partial profit booking statistics
    if trades_log:
        partial_exits = [trade for trade in trades_log if trade['type'] in ['long_partial_exit', 'short_partial_exit']]
        if partial_exits:
            partial_df = pd.DataFrame(partial_exits)
            total_partial_profit = partial_df['profit'].sum()
            print(f"\n=== Partial Profit Booking Statistics ===")
            print(f"Total Partial Exits: {len(partial_exits)}")
            print(f"Total Profit from Partial Exits: ${total_partial_profit:,.2f}")
            print(f"Average Partial Exit Profit: ${partial_df['profit'].mean():,.2f}")

    # Save results
    if trades_log:
        trades_df = pd.DataFrame(trades_log)
        trades_df.to_csv(RESULTS_CSV_PATH, index=False)
        print(f"\nTrades log saved to '{RESULTS_CSV_PATH}'")

    # Save portfolio values for analysis
    portfolio_df = df_with_results[['Close', 'Portfolio_Value', 'Available_Capital', 'Position_Size']].copy()
    portfolio_csv_path = 'ohlc_supertrend_portfolio_with_partial_profit_values.csv'
    portfolio_df.to_csv(portfolio_csv_path)
    print(f"Portfolio values saved to '{portfolio_csv_path}'")

    return metrics

if __name__ == "__main__":
    main()