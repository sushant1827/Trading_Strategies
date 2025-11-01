import pandas as pd
import numpy as np

def backtest_investment_strategy(ohlc_data, initial_capital=50000,
                                  weekly_investment=500,
                                  reference_for_dip='previous_close',
                                  profit_booking_threshold=0.08):
    """
    Backtest the investment strategy with profit booking and comprehensive metrics

    Rules:
    1. Invest on every Monday at closing price using booked profit + additional if needed:
       - Base amount ₹500, but use booked profit first, then additional cash
       - If the closing price is ≤ ₹500, calculate shares as: maximum of 1 or (₹500 ÷ closing price, rounded down to nearest whole number)
       - If the closing price is > ₹500, buy exactly 1 share
    2. On any day if price drops >=1% from previous close (measured at close), invest ₹1000 + additional using booked profit first:
       - For each additional 0.5% drop beyond the initial 1% (measured from low to reference close), invest additional ₹500
       - Reference close may be adjusted based on qualifying candle logic (previous candles that are red with open-to-low drop <1%)
       - Investments funded by booked profit first, then additional cash (can go negative)
    3. Book profit when average profit reaches profit_booking_threshold (default 8%):
       - Sell enough shares to bring unrealized P&L back to the threshold %, add excess profit to booked profit
       - Booked profit is used for subsequent Monday SIP and dip investments
    4. Unlimited funds for additional investment (can go negative cash)

    Parameters:
    -----------
    ohlc_data : DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    initial_capital : float, starting capital
    weekly_investment : float, base investment on Monday
    reference_for_dip : str, 'previous_close' or 'day_open' (currently not used, uses previous_close with qualifying logic)
    profit_booking_threshold : float, percentage threshold for booking profit (e.g., 0.08 for 8%)
    """

    # Prepare data
    df = ohlc_data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['DayOfWeek'] = df['Date'].dt.day_name()

    # Initialize tracking variables
    cash = initial_capital
    shares = 0
    total_cost = 0  # total amount invested in shares
    booked_profit = 0  # accumulated booked profit from sales
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

        # Rule 3: Profit Booking - check every day before investments
        avg_buy_price = total_cost / shares if shares > 0 else 0
        if avg_buy_price > 0:
            unrealized_profit_pct = ((close_price - avg_buy_price) / avg_buy_price) * 100
            if unrealized_profit_pct >= profit_booking_threshold * 100 and shares > 0:
                # Book the profit above the threshold
                threshold_pct = profit_booking_threshold * 100
                excess_pct = unrealized_profit_pct - threshold_pct
                sell_fraction = excess_pct / unrealized_profit_pct
                sell_quantity = int(shares * sell_fraction)
                if sell_quantity > 0:
                    sell_quantity = min(sell_quantity, shares - 1)  # Leave at least 1 share
                    if sell_quantity > 0:
                        sell_amount = sell_quantity * close_price
                        gain = sell_amount - sell_quantity * avg_buy_price
                        cash += sell_amount
                        booked_profit += gain
                        total_cost -= sell_quantity * avg_buy_price
                        shares -= sell_quantity

                        trades.append({
                            'Date': date,
                            'Type': 'Book Profit',
                            'Price': close_price,
                            'Quantity': -sell_quantity,
                            'Amount': -sell_amount,
                            'Cash_Remaining': cash,
                            'Total_Shares': shares
                        })
                        trade_dates.add(date)

        daily_investment = 0
        daily_shares_bought = 0

        # Rule 1: Monday Investment - every Monday at close
        if day_of_week == 'Monday':
            required = weekly_investment
            used_from_booked = min(required, booked_profit)
            booked_profit -= used_from_booked
            used_from_cash = required - used_from_booked
            cash -= used_from_cash  # Allow negative cash for unlimited additional funds

            if close_price <= required:
                quantity = max(1, int(required / close_price))
            else:
                quantity = 1

            amount = quantity * close_price
            shares += quantity
            total_cost += amount  # Amount is close_price * quantity, approximately required
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
                required = 1000 + num_additional_increments * 500

                # Fund investment using booked profit first, then additional cash
                used_from_booked = min(required, booked_profit)
                booked_profit -= used_from_booked
                used_from_cash = required - used_from_booked
                cash -= used_from_cash  # Allow negative cash for unlimited additional funds

                # Buy at close price
                buy_price = close_price
                quantity = max(1, int(required / buy_price))

                amount = quantity * buy_price
                shares += quantity
                total_cost += amount
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
        avg_buy_price = total_cost / shares if shares > 0 else 0
        portfolio_values.append({
            'Date': date,
            'Cash': cash,
            'Shares': shares,
            'Close_Price': close_price,
            'Holdings_Value': shares * close_price,
            'Total_Portfolio_Value': portfolio_value,
            'Total_Invested': initial_capital - cash + daily_investment,
            'Unrealized_PnL': portfolio_value - initial_capital,
            'Unrealized_PnL_Percent': ((portfolio_value - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0
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
    total_amount_invested = trades_df['Amount'].sum() if len(trades_df) > 0 else 0  # This is net now, but we need gross buying
    buy_trades_df = trades_df[trades_df['Amount'] > 0]
    total_amount_invested_buying = buy_trades_df['Amount'].sum() if len(buy_trades_df) > 0 else 0
    avg_buy_price = total_cost / shares if shares > 0 else 0
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
        'Total Amount Invested in Buying': f'₹{total_amount_invested_buying:,.2f}',
        'Total Amount Profited (Booked)': f'₹{initial_capital - cash + total_amount_invested_buying:,.2f}',  # Approximate net investment
        'Total Shares Owned': shares,
        'Average Buy Price': f'₹{avg_buy_price:.2f}',
        'Current Price': f'₹{current_price:.2f}',
        'Unrealized Gain/Loss per Share': f'₹{current_price - avg_buy_price:.2f}',
        'Cash Remaining': f'₹{cash:,.2f}',
        'Booked Profit Remaining': f'₹{booked_profit:,.2f}',
        'Max Drawdown': f'{max_drawdown:.2f}%',
        'Sharpe Ratio': f'{sharpe_ratio:.2f}',
        'Win Rate (Positive Days)': f'{win_rate:.2f}%',
        'Trading Period': f'{days_in_market} days ({start_date.date()} to {end_date.date()})',
        'Avg Investment per Trade': f'₹{total_amount_invested_buying/num_trades:,.2f}' if num_trades > 0 else '₹0.00',
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
    import os

    # Define the path to the CSV file relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, '..', 'NIFTYBEES', 'NIFTYBEES_EQ_D_Min.csv')

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
        reference_for_dip='previous_close',
        profit_booking_threshold=0.25  # 8%
    )

    # Display results
    print_metrics(results['metrics'])

    # Save trades to CSV
    results['trades'].to_csv('ETF_SIP_Trades.csv', index=False)

    # Plot portfolio value
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(results['portfolio']['Date'], results['portfolio']['Total_Portfolio_Value'], linewidth=2, color='blue')
    plt.title('ETF SIP with Profit Booking - Portfolio Value Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Portfolio Value (INR)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ETF_SIP_Portfolio_Plot.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close to prevent display issues


# ============================================================

## ETF SIP Investment Strategy with Profit Booking Rules

# Based on the provided Python script, here are the rules of the ETF SIP (Systematic Investment Plan) strategy with profit booking, explained in plain English step by step. This strategy combines regular weekly investments, opportunistic dip buying during market declines, and systematic profit-taking to lock in gains. It assumes unlimited additional funds can be borrowed if needed (cash can go negative).

# ## 1. __Initial Setup__

# - Start with an initial amount of money (default ₹50,000).

# - Track three types of funds:

#   - __Cash balance__: Available money for investments.
#   - __Shares owned__: ETFs held in the portfolio.
#   - __Booked profit__: Profitable gains that have been "locked in" by selling shares at a profit and set aside for future investments.

# ## 2. __Weekly Monday Regular Investments (SIP Component)__

# Every Monday at market close:

# - Attempt to invest ₹500 (base amount).

# - __Funding priority__: Use booked profit first. If there's not enough booked profit, use remaining cash (which can go negative if needed for unlimited borrowing).

# - __Share calculation__:

#   - If the ETF closing price is ₹500 or less: Buy as many shares as possible with ₹500 (minimum 1 share), calculated as ₹500 ÷ closing price rounded down.
#   - If the ETF closing price is more than ₹500: Buy exactly 1 share.

# - This systematic weekly buying creates regular entry points regardless of market conditions.

# ## 3. __Opportunistic Dip Buying During Declines__

# Every trading day (not just Mondays), check for significant price drops:

# - __Detect dips__: Compare current day's closing price to a reference price (starts with previous day's close, but can adjust based on candle patterns - see below).

# - __Trigger condition__: If the drop is 1% or more from the reference price.

# - __Investment calculation__:

#   - Base investment of ₹1,000 for the initial 1% drop.
#   - Add ₹500 for each additional 0.5% the price dropped below the reference point (measured using the day's low price).
#   - __Example__: If price drops 2.5% below reference (initial 1% + 3×0.5%), invest ₹1,000 + (3×₹500) = ₹2,500.

# - __Reference price adjustment__:

#   - Examine up to 3 previous days' candles moving backwards.
#   - Look for "qualifying" red candles (where close < open and the drop from open to low is less than 1%).
#   - If found, use that qualifying candle's opening price as the new reference point instead of the previous close.

# - __Funding priority__: Use booked profit first, then cash (can go negative).

# - __Purchase__: Buy at the current day's closing price, minimum 1 share.

# ## 4. __Profit Booking (Taking Gains)__

# Check every day at market close, before any investments:

# - Calculate the average buy price of all owned shares.

# - Calculate unrealized percentage profit = [(current price - average buy price) ÷ average buy price] × 100%.

# - __If unrealized profit ≥ profit booking threshold (default 8%)__:

#   - Sell enough shares to bring the unrealized profit back to the threshold %.
#   - Add the excess profit to booked profit (this profit is now locked in and saved for future investments).
#   - The number of shares sold is calculated to reset the portfolio's unrealized gains to the threshold level.

# ## 5. __Portfolio Management__

# - __Total cost tracking__: Track cumulative money invested in shares.
# - __Unlimited liquidity__: Cash can go negative (borrowing allowed) to ensure no investment opportunities are missed during deep dips.
# - __Reinvestment__: Booked profits are prioritized for both regular Monday SIP investments and dip buying opportunities.
# - __Daily tracking__: Record portfolio value, cash balance, shares owned, and various performance metrics.

# ## 6. __Key Strategy Characteristics__

# - Combines regular income-based investing (SIP) with momentum/mean-reversion (dip buying).
# - Uses profit-taking to lock in gains and fuel future growth.
# - Assumes volatile markets where dips represent buying opportunities.
# - Designed for long-term holding with systematic entry and disciplined exit points.

# ============================================================


# ============================================================
#                TRADING METRICS REPORT
# ============================================================

# Initial Capital.........................      ₹50,000.00
# Final Portfolio Value...................     ₹179,236.00
# Total Return............................     ₹129,236.00
# Total Return %..........................         258.47%
# CAGR....................................          30.40%
# Number of Trades........................             417
# Total Amount Invested in Buying.........     ₹445,267.32
# Total Amount Profited (Booked)..........     ₹941,767.32
# Total Shares Owned......................            2150
# Average Buy Price.......................         ₹207.10
# Current Price...........................         ₹291.04
# Unrealized Gain/Loss per Share..........          ₹83.94
# Cash Remaining..........................    ₹-446,500.00
# Booked Profit Remaining.................           ₹0.00
# Max Drawdown............................         -47.85%
# Sharpe Ratio............................            0.88
# Win Rate (Positive Days)................          52.98%
# Trading Period.......................... 1757 days (2021-01-01 to 2025-10-24)
# Avg Investment per Trade................       ₹1,067.79

# ============================================================
