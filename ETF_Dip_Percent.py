import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def backtest_dip_only_strategy(ohlc_data, initial_capital=50000,
                                reference_for_dip='previous_close', pe_data=None):
    """
    Backtest the dip-only investment strategy with comprehensive metrics

    Rules:
    1. On any day if price drops >=1% from previous close (measured at close), invest ₹1000
    2. For each additional 0.5% drop beyond the initial 1% (measured from low to previous close), invest additional ₹500
        (e.g., if close drops 1.4% but low was 1.9%, investment = ₹1000 for 1% + ₹500 for 1.5% = ₹1500)
    3. Buy at the close price (in real trading, use limit orders based on how low price goes)
    4. Buy-only strategy with unlimited funds (can go negative cash)
    5. Apply PE ratio-based investment multiplier based on previous day's PE:
        - PE < 18: 1.5x (more aggressive investing when valuations are low)
        - PE 18-22: 1x (normal investing)
        - PE > 22: 0.5x (less aggressive when valuations are high)

    Parameters:
    -----------
    ohlc_data : DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    initial_capital : float, starting capital
    reference_for_dip : str, 'previous_close' or 'day_open'
    pe_data : DataFrame with columns ['Date', 'P/E'], optional PE ratio data
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

        if previous_close is None:
            # First day, no previous candle for logic, just set portfolio
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
            continue

        # Rule 1: Dip Buying - check every day
        # Default reference is previous close
        reference_price = previous_close

        # Check up to 3 previous candles for qualifying conditions
        # A qualifying candle is red (close < open) with open-to-low drop < 1%
        for back in range(1, 4):  # Check previous 1, 2, 3 candles
            if idx - back < 0:
                break
            prev_row = df.iloc[idx - back]
            p_close = prev_row['Close']
            p_open = prev_row['Open']
            p_low = prev_row['Low']

            if p_close >= p_open:  # Not red, break chain
                break
            p_drop_percent = ((p_open - p_low) / p_open) * 100
            if p_drop_percent < 1:
                reference_price = p_open  # Set to this qualifying open
            else:
                break  # Does not qualify, break chain

        dip_percent = ((reference_price - close_price) / reference_price) * 100

        if dip_percent >= 1:
            # Invest ₹1000 for the initial 1% drop, plus ₹500 for each additional 0.5% drop based on low price
            full_drop_percent = ((reference_price - low_price) / reference_price) * 100
            additional_drops = full_drop_percent - 1.0
            num_additional_increments = int(additional_drops // 0.5)
            base_dip_investment = 1000 + num_additional_increments * 500

            # Apply PE-based investment multiplier based on previous day's PE ratio
            pe_multiplier = 1.0  # Default multiplier
            if pe_data is not None:
                # Find previous day's PE ratio
                prev_date = date - pd.Timedelta(days=1)
                prev_pe_row = pe_data[pe_data['Date'] == prev_date]
                if not prev_pe_row.empty:
                    prev_pe = prev_pe_row['PE'].iloc[0]
                    if pd.notna(prev_pe):
                        if prev_pe < 18:
                            pe_multiplier = 1.5
                        elif prev_pe <= 22:
                            pe_multiplier = 1.0
                        else:  # prev_pe > 22
                            pe_multiplier = 0.5

            # Apply the multiplier to the base investment
            dip_investment = base_dip_investment * pe_multiplier

            # Buy at the close price
            buy_price = close_price
            quantity = max(1, int(dip_investment / buy_price))

            amount = quantity * buy_price
            shares += quantity
            cash -= amount  # Allow negative cash for unlimited funds
            daily_investment += amount
            daily_shares_bought += quantity

            # Use the actual dip percent for display, but calculation was based on reference_price
            actual_dip_from_prev_close = ((previous_close - close_price) / previous_close) * 100
            pe_info = f" (PE: {prev_pe:.1f}, x{pe_multiplier:.1f})" if pe_data is not None and 'prev_pe' in locals() and pd.notna(prev_pe) else ""
            trades.append({
                'Date': date,
                'Type': f'Dip Buy ({actual_dip_from_prev_close:.1f}% from prev close - ref: {reference_price:.2f}){pe_info}',
                'Price': buy_price,
                'Quantity': quantity,
                'Amount': amount,
                'Cash_Remaining': cash,
                'Total_Shares': shares
            })

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

    # Calculate Yearly Metrics
    portfolio_df['Year'] = portfolio_df['Date'].dt.year
    trades_df['Year'] = trades_df['Date'].dt.year

    yearly_metrics = []
    for year in sorted(portfolio_df['Year'].unique()):
        port_year = portfolio_df[portfolio_df['Year'] == year]
        trades_year = trades_df[trades_df['Year'] == year]

        initial_cap_year = port_year.iloc[0]['Total_Portfolio_Value']
        final_pv_year = port_year.iloc[-1]['Total_Portfolio_Value']
        total_return_year = final_pv_year - initial_cap_year
        num_trades_year = len(trades_year)
        total_invested_year = trades_year['Amount'].sum() if not trades_year.empty else 0
        total_return_pct_year = (total_return_year / total_invested_year) * 100 if total_invested_year != 0 else 0
        avg_buy_price_year = trades_year['Price'].mean() if not trades_year.empty else 0
        year_end_price = port_year.iloc[-1]['Close_Price']

        yearly_metrics.append({
            'Year': year,
            'Final Portfolio Value': f'₹{final_pv_year:,.2f}',
            'Total Return': f'₹{final_pv_year:,.2f}',
            'Total Return %': f'{final_pv_year / total_invested_year * 100:.2f}%',
            'Number of Trades': num_trades_year,
            'Total Amount Invested in Buying': f'₹{total_invested_year:,.2f}',
            'Average Buy Price': f'₹{avg_buy_price_year:.2f}',
            'Year End Price': f'₹{year_end_price:.2f}'
        })

    return {
        'metrics': metrics,
        'trades': trades_df,
        'portfolio': portfolio_df,
        'ohlc': df,
        'yearly_metrics': yearly_metrics
    }


def print_yearly_metrics(yearly_metrics):
    """Print yearly metrics in a formatted way"""
    if not yearly_metrics:
        return
    print("\n" + "="*130)
    print(" "*45 + "ANNUAL BREAKDOWN")
    print("="*130 + "\n")

    # Print header
    headers = ['Year', 'Final Portfolio Value', 'Total Return', 'Total Return %', 'Number of Trades', 'Total Inv Invested', 'Avg Buy Price', 'Year End Price']
    print("{:<6} {:>22} {:>14} {:>15} {:>17} {:>19} {:>15} {:>17}".format(*headers))
    print("="*130)

    for ym in yearly_metrics:
        row = [
            ym['Year'],
            ym['Final Portfolio Value'].replace('₹', ''),
            ym['Total Return'].replace('₹', ''),
            ym['Total Return %'],
            ym['Number of Trades'],
            ym['Total Amount Invested in Buying'].replace('₹', ''),
            ym['Average Buy Price'].replace('₹', ''),
            ym['Year End Price'].replace('₹', '')
        ]
        print("{:<6} {:>22} {:>14} {:>15} {:>17} {:>19} {:>15} {:>17}".format(*row))

    print("\n" + "="*130)


def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print("\n" + "="*60)
    print(" "*15 + "TRADING METRICS REPORT")
    print("="*60 + "\n")

    for key, value in metrics.items():
        print(f"{key:.<40} {value:>15}")
    print("\n" + "="*60)


def plot_results(results):
    """Create visualization of the strategy performance"""
    portfolio_df = results['portfolio']
    trades_df = results['trades']
    ohlc_df = results['ohlc']

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Portfolio Value Over Time
    ax1 = axes[0]
    ax1.plot(portfolio_df['Date'], portfolio_df['Total_Portfolio_Value'],
             label='Portfolio Value', linewidth=2, color='blue')
    ax1.axhline(y=50000, color='red', linestyle='--', label='Initial Capital', alpha=0.7)
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value (₹)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Stock Price with Buy Points
    ax2 = axes[1]
    ax2.plot(ohlc_df['Date'], ohlc_df['Close'], label='Close Price',
             linewidth=1.5, color='black', alpha=0.7)

    if len(trades_df) > 0:
        dip_trades = trades_df[trades_df['Type'].str.contains('Dip Buy')]

        if len(dip_trades) > 0:
            ax2.scatter(dip_trades['Date'], dip_trades['Price'],
                       color='orange', marker='v', s=100, label='Dip Buy', zorder=5)

    ax2.set_title('Stock Price with Entry Points', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price (₹)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Drawdown
    ax3 = axes[2]
    ax3.fill_between(portfolio_df['Date'], portfolio_df['Drawdown'], 0,
                     color='red', alpha=0.3)
    ax3.plot(portfolio_df['Date'], portfolio_df['Drawdown'],
             color='darkred', linewidth=1.5)
    ax3.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Sample data - Replace this with your actual OHLC data
    # Your DataFrame should have columns: Date, Open, High, Low, Close, Volume

    # Example: Load from CSV
    # df = pd.read_csv('your_stock_data.csv')

    # Define the path to the CSV file
    csv_file_path = 'D:\\Sushant\\Fyers_AlgoTrade\\Fyers_Data\\NIFTYBEES\\NIFTYBEES_EQ_D_Min.csv'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Remove 'Unnamed: 0' and 'Volume' columns
    df = df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')

    # Rename 'Date' column to 'DateTime'
    # df = df.rename(columns={'Date': 'DateTime'})

    # Set 'DateTime' column as index
    df = df.set_index('Date')

    # Convert index to datetime objects
    df.index = pd.to_datetime(df.index)

    # Reset index to make Date a column again for compatibility with backtest function
    df = df.reset_index()

    # Drop duplicate Date entries, keeping the first one
    df = df[~df['Date'].duplicated(keep='first')]
    print(f"After dropping duplicates, df Date column is unique: {df['Date'].is_unique}")

    # Filter data from 2021 onwards
    df = df[df['Date'].dt.year >= 2021]
    print(f"After filtering for 2021 onwards, df shape: {df.shape}")

    # Load PE data
    print("Loading PE ratio data...")
    pe_df = pd.read_csv('NIFTY50_Historical_PE_PB_DIV_Merged.csv')
    pe_df['Date'] = pd.to_datetime(pe_df['Date'])
    pe_df = pe_df.sort_values('Date').reset_index(drop=True)
    pe_df = pe_df[['Date', 'P/E']].rename(columns={'P/E': 'PE'})

    # Remove duplicate dates, keeping the last one
    pe_df = pe_df[~pe_df['Date'].duplicated(keep='last')]
    print(f"PE data loaded: {pe_df.shape} rows")

    # Example: Create sample data
    # sample_data = {
    #     'Date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
    #     'Open': np.random.uniform(450, 550, 100),
    #     'High': np.random.uniform(460, 560, 100),
    #     'Low': np.random.uniform(440, 540, 100),
    #     'Close': np.random.uniform(450, 550, 100),
    #     'Volume': np.random.randint(100000, 1000000, 100)
    # }
    # df = pd.DataFrame(sample_data)

    # Run backtest
    results = backtest_dip_only_strategy(
        ohlc_data=df,
        initial_capital=50000,
        reference_for_dip='previous_close',  # or 'day_open'
        pe_data=pe_df
    )

    # Display results
    print_metrics(results['metrics'])

    print_yearly_metrics(results['yearly_metrics'])

    # Save trades to CSV
    results['trades'].to_csv('ETF_Dip_Trades.csv', index=False)
    print("\nTrades saved to ETF_Dip_Trades.csv")

    print("\n" + "="*60)
    print("TRADE HISTORY (First 10 trades)")
    print("="*60)
    print(results['trades'].head(10).to_string(index=False))

    print("\n" + "="*60)
    print("PORTFOLIO SUMMARY (Last 5 days)")
    print("="*60)
    print(results['portfolio'][['Date', 'Shares', 'Close_Price', 'Total_Portfolio_Value',
                                 'Unrealized_PnL_Percent']].tail().to_string(index=False))

    # Plot results
    plot_results(results)


# ============================================================
#                TRADING METRICS REPORT
# ============================================================

# Initial Capital.........................      ₹50,000.00
# Final Portfolio Value...................     ₹156,694.49
# Total Return............................     ₹106,694.49
# Total Return %..........................         213.39%
# CAGR....................................          26.80%
# Number of Trades........................             122
# Total Amount Invested in Buying.........     ₹232,367.11
# Total Shares Owned......................            1165
# Average Buy Price.......................         ₹212.93
# Current Price...........................         ₹291.04
# Unrealized Gain/Loss per Share..........          ₹78.11
# Cash Remaining..........................    ₹-182,367.11
# Max Drawdown............................         -29.20%
# Sharpe Ratio............................            1.17
# Win Rate (Positive Days)................          53.31%
# Trading Period.......................... 1757 days (2021-01-01 to 2025-10-24)
# Avg Investment per Trade................       ₹1,904.65

# ============================================================
