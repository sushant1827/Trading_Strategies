import pandas as pd
import numpy as np

def backtest_investment_strategy(ohlc_data, initial_capital=50000,
                                  weekly_investment=500,
                                  reference_for_dip='previous_close'):
    """
    Backtest the investment strategy with comprehensive metrics

    Rules:
    1. Invest on every Monday at closing price:
       - Max amount ₹500, quantity = max(1, int(500 / close_price)) if close_price <=500 else 1
    2. On any day if price drops >=1% from previous close (measured at close), invest ₹1000
       - For each additional 0.5% drop beyond the initial 1% (measured from low to reference close), invest additional ₹500
       - Reference close may be adjusted based on qualifying candle logic (previous candles that are red with open-to-low drop <1%)
    3. Buy-only strategy with unlimited funds (can go negative cash)

    Parameters:
    -----------
    ohlc_data : DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    initial_capital : float, starting capital
    weekly_investment : float, max investment on Thursday
    reference_for_dip : str, 'previous_close' or 'day_open' (currently not used, uses previous_close with qualifying logic)
    """
    
    # Prepare data
    df = ohlc_data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['DayOfWeek'] = df['Date'].dt.day_name()
    
    # Initialize tracking variables
    cash = initial_capital
    shares = 0
    trades = []
    portfolio_values = []
    trade_dates = set()  # Track dates where trades occurred

    previous_close = None
    previous_open = None
    previous_low = None
    
    for idx, row in df.iterrows():
        date = row['Date']
        open_price = row['Open']
        close_price = row['Close']
        high_price = row['High']
        low_price = row['Low']
        day_of_week = row['DayOfWeek']

        daily_investment = 0
        daily_shares_bought = 0

        # Rule 1: Monday Investment - every Monday at close
        if day_of_week == 'Monday':
            if close_price <= weekly_investment:
                quantity = max(1, int(weekly_investment / close_price))
            else:
                quantity = 1

            amount = quantity * close_price
            shares += quantity
            cash -= amount  # Allow negative cash for unlimited funds
            daily_investment += amount
            daily_shares_bought += quantity

            trades.append({
                'Date': date,
                'Type': 'Monday SIP',
                'Price': close_price,
                'Quantity': quantity,
                'Amount': amount,
                'Cash_Remaining': cash,
                'Total_Shares': shares
            })
            trade_dates.add(date)

        # Rule 2: Dip Buying - check every day
        if previous_close is not None:
            reference_price = previous_close
            # Check up to 3 previous candles for qualifying conditions
            for back in range(1, 4):  # Check previous 1, 2, 3 candles
                if idx - back < 0:
                    break
                prev_row = df.iloc[idx - back]
                prev_date = prev_row['Date']
                if prev_date in trade_dates:
                    continue  # Skip candles where trade was taken
                p_close = prev_row['Close']
                p_open = prev_row['Open']
                p_low = prev_row['Low']
                if p_close >= p_open:  # Not red, break chain
                    break
                p_drop_percent = ((p_open - p_low) / p_open) * 100
                if p_drop_percent < 1:
                    reference_price = p_open  # Set to this qualifying open
                    break  # Break chain, use this reference
            dip_percent = ((reference_price - close_price) / reference_price) * 100

            if dip_percent >= 1:
                # Invest ₹1000 for initial 1%, plus ₹500 for each additional 0.5% based on low price
                full_drop_percent = ((reference_price - low_price) / reference_price) * 100
                additional_drops = full_drop_percent - 1
                num_additional_increments = int(additional_drops // 0.5)
                dip_investment = 1000 + num_additional_increments * 500

                # Buy at close price
                buy_price = close_price
                quantity = max(1, int(dip_investment / buy_price))

                amount = quantity * buy_price
                shares += quantity
                cash -= amount  # Allow negative cash for unlimited funds
                daily_investment += amount
                daily_shares_bought += quantity

                actual_dip_from_prev_close = ((previous_close - close_price) / previous_close) * 100
                trades.append({
                    'Date': date,
                    'Type': f'Dip Buy ({actual_dip_from_prev_close:.1f}% from prev close - ref: {reference_price:.2f})',
                    'Price': buy_price,
                    'Quantity': quantity,
                    'Amount': amount,
                    'Cash_Remaining': cash,
                    'Total_Shares': shares
                })
                trade_dates.add(date)

        # Calculate portfolio value at close
        portfolio_value = cash + (shares * close_price)
        portfolio_values.append({
            'Date': date,
            'Cash': cash,
            'Shares': shares,
            'Close_Price': close_price,
            'Holdings_Value': shares * close_price,
            'Total_Portfolio_Value': portfolio_value,
            'Total_Invested': initial_capital - cash + daily_investment,
            'Unrealized_PnL': portfolio_value - initial_capital,
            'Unrealized_PnL_Percent': ((portfolio_value - initial_capital) / initial_capital) * 100
        })

        previous_close = close_price
        previous_open = open_price
        previous_low = low_price
    
    # Create DataFrames
    trades_df = pd.DataFrame(trades)
    portfolio_df = pd.DataFrame(portfolio_values)
    
    # Calculate Trading Metrics (TradingView style)
    final_portfolio_value = portfolio_df.iloc[-1]['Total_Portfolio_Value']
    total_invested = initial_capital
    total_return = final_portfolio_value - total_invested
    total_return_percent = (total_return / total_invested) * 100
    
    # Calculate daily returns
    portfolio_df['Daily_Return'] = portfolio_df['Total_Portfolio_Value'].pct_change()
    
    # Calculate metrics
    num_trades = len(trades_df)
    total_amount_invested = trades_df['Amount'].sum() if len(trades_df) > 0 else 0
    avg_buy_price = trades_df['Price'].mean() if len(trades_df) > 0 else 0
    current_price = df.iloc[-1]['Close']
    
    # Drawdown calculation
    portfolio_df['Cumulative_Max'] = portfolio_df['Total_Portfolio_Value'].cummax()
    portfolio_df['Drawdown'] = (portfolio_df['Total_Portfolio_Value'] - portfolio_df['Cumulative_Max']) / portfolio_df['Cumulative_Max'] * 100
    max_drawdown = portfolio_df['Drawdown'].min()
    
    # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
    returns = portfolio_df['Daily_Return'].dropna()
    if len(returns) > 0 and returns.std() != 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0
    
    # Win rate (days with positive returns)
    positive_days = (returns > 0).sum()
    total_days = len(returns)
    win_rate = (positive_days / total_days * 100) if total_days > 0 else 0
    
    # Trading period
    start_date = df.iloc[0]['Date']
    end_date = df.iloc[-1]['Date']
    days_in_market = (end_date - start_date).days
    
    # CAGR (Compound Annual Growth Rate)
    years = days_in_market / 365.25
    cagr = (((final_portfolio_value / initial_capital) ** (1 / years)) - 1) * 100 if years > 0 else 0
    
    metrics = {
        'Initial Capital': f'₹{initial_capital:,.2f}',
        'Final Portfolio Value': f'₹{final_portfolio_value:,.2f}',
        'Total Return': f'₹{total_return:,.2f}',
        'Total Return %': f'{total_return_percent:.2f}%',
        'CAGR': f'{cagr:.2f}%',
        'Number of Trades': num_trades,
        'Total Amount Invested in Buying': f'₹{total_amount_invested:,.2f}',
        'Total Shares Owned': shares,
        'Average Buy Price': f'₹{avg_buy_price:.2f}',
        'Current Price': f'₹{current_price:.2f}',
        'Unrealized Gain/Loss per Share': f'₹{current_price - avg_buy_price:.2f}',
        'Cash Remaining': f'₹{cash:,.2f}',
        'Max Drawdown': f'{max_drawdown:.2f}%',
        'Sharpe Ratio': f'{sharpe_ratio:.2f}',
        'Win Rate (Positive Days)': f'{win_rate:.2f}%',
        'Trading Period': f'{days_in_market} days ({start_date.date()} to {end_date.date()})',
        'Avg Investment per Trade': f'₹{total_amount_invested/num_trades:,.2f}' if num_trades > 0 else '₹0.00',
    }
    
    return {
        'metrics': metrics,
        'trades': trades_df,
        'portfolio': portfolio_df,
        'ohlc': df
    }


def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print("\n" + "="*60)
    print(" "*15 + "TRADING METRICS REPORT")
    print("="*60 + "\n")

    for key, value in metrics.items():
        print(f"{key:.<40} {value:>15}")
    print("\n" + "="*60)


# Example usage:
if __name__ == "__main__":
    # Define the path to the CSV file
    csv_file_path = 'NIFTYBEES\\NIFTYBEES_EQ_D_Min.csv'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Remove 'Unnamed: 0' and 'Volume' columns
    df = df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')

    # Set 'Date' column as index
    df = df.set_index('Date')

    # Convert index to datetime objects
    df.index = pd.to_datetime(df.index)

    # Reset index to make Date a column again for compatibility with backtest function
    df = df.reset_index()

    # Drop duplicate Date entries, keeping the first one
    df = df[~df['Date'].duplicated(keep='first')]

    # Filter data from 2021 onwards
    df = df[df['Date'].dt.year >= 2021]

    # Run backtest
    results = backtest_investment_strategy(
        ohlc_data=df,
        initial_capital=50000,
        weekly_investment=500,
        reference_for_dip='previous_close'  # or 'day_open'
    )

    # Display results
    print_metrics(results['metrics'])


# ============================================================

## ETF SIP Investment Strategy Rules

# This is a long-term buy-only investment strategy that combines weekly systematic investment ("SIP") with additional opportunistic purchases during market dips. The strategy uses unlimited borrowing capacity (can go into negative cash) to maintain position sizing during weak markets.

# ### Core Investment Rules:

# **1. Weekly Monday Investment (SIP Component):**
# - Every Monday at market close, automatically invest ₹500
# - If the closing price is ≤ ₹500, calculate shares as: maximum of 1 or (₹500 ÷ closing price, rounded down to nearest whole number)
# - If the closing price is > ₹500, buy exactly 1 share
# - This creates a systematic weekly buying pattern regardless of market conditions

# **2. Opportunistic Dip Buying (Extra Investment Component):**
# - Check every trading day (not just Mondays) for potential dip opportunities
# - Look for price drops of 1% or more from a reference price
# - The reference price starts as the previous day's closing price, but can be adjusted based on qualifying candle patterns

# **Reference Price Adjustment Logic:**
# - Examine up to 3 previous candles (going backwards from current day)
# - Look for "qualifying candles": red candles (close < open) where the open-to-low drop is less than 1%
# - If such a candle is found moving backwards, use its opening price as the new reference price
# - Stop checking further back once a qualifying candle is found or a non-qualifying candle is encountered

# **Dip Investment Calculation:**
# - If the current close is ≥1% below the reference price:
#   - Invest ₹1,000 base amount for the initial 1% drop
#   - For each additional 0.5% drop below the reference price (measured against the day's low price), add ₹500 more investment
#   - Use the full drop from reference to low price to calculate extra increments
#   - Buy the calculated amount at the current closing price, minimum 1 share
#   - Trade was taken on any of the previous 3 candles, those candles are not considered
#     for the dip percent reference calculation, and the next qualifying candle (if available) is used instead.

# **Portfolio Management:**
# - No selling component - this is purely a buy-and-hold strategy
# - Unlimited borrowing allowed to maintain investment amounts during drawdowns
# - Tracks comprehensive portfolio metrics including returns, drawdowns, Sharpe ratio, and CAGR

# **Key Characteristics:**
# - Combines systematic weekly investing with momentum-based dip buying
# - Uses technical analysis to identify and quantify market weakness
# - Assumes unlimited liquidity/funding access for uninterrupted buying
# - Designed for volatile markets where dips create buying opportunities

# ============================================================
#                TRADING METRICS REPORT
# ============================================================

# Initial Capital.........................      ₹50,000.00
# Final Portfolio Value...................     ₹230,468.68
# Total Return............................     ₹180,468.68
# Total Return %..........................         360.94%
# CAGR....................................          37.39%
# Number of Trades........................             417
# Total Amount Invested in Buying.........     ₹445,267.32
# Total Shares Owned......................            2150
# Average Buy Price.......................         ₹218.49
# Current Price...........................         ₹291.04
# Unrealized Gain/Loss per Share..........          ₹72.55
# Cash Remaining..........................    ₹-395,267.32
# Max Drawdown............................         -36.15%
# Sharpe Ratio............................            1.21
# Win Rate (Positive Days)................          53.81%
# Trading Period.......................... 1757 days (2021-01-01 to 2025-10-24)
# Avg Investment per Trade................       ₹1,067.79

# ============================================================
