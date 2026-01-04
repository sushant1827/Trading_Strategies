import pandas as pd
import numpy as np
from datetime import datetime

def calculate_metrics(portfolio_history, trades, initial_capital, end_date, start_date):
    """
    Calculates key performance metrics for the backtest.
    """
    if not portfolio_history:
        return {"Error": "No portfolio history to calculate metrics."}

    # Create DataFrame from portfolio history
    portfolio_df = pd.DataFrame(portfolio_history).set_index('date')
    portfolio_df['value'] = portfolio_df['value'].astype(float)

    # --- Basic Performance ---
    final_total_value = portfolio_df['value'].iloc[-1]
    net_profit = final_total_value - initial_capital
    total_return_pct = (net_profit / initial_capital) * 100

    # --- Time-Based Metrics ---
    num_days = (end_date - start_date).days
    annualized_return_pct = 0
    if num_days > 0:
        annualized_return_pct = ((1 + (total_return_pct / 100)) ** (365.25 / num_days) - 1) * 100

    # --- Drawdown ---
    portfolio_df['peak'] = portfolio_df['value'].cummax()
    portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['peak']) / portfolio_df['peak']
    max_drawdown_pct = portfolio_df['drawdown'].min() * 100

    # --- Trade Statistics ---
    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        # Ensure dates are datetime
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

        trades_df['pnl'] = trades_df['exit_price'] - trades_df['entry_price']
        trades_df['pnl_pct'] = (trades_df['pnl'] / trades_df['entry_price']) * 100

        # Calculate holding time
        trades_df['holding_days'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days

        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]

        num_trades = len(trades_df)
        num_wins = len(wins)
        num_losses = len(losses)

        win_rate_pct = (num_wins / num_trades) * 100 if num_trades > 0 else 0

        gross_profit = wins['pnl'].sum() * wins.get('quantity', 1).mean() # Approximate if quantity varies
        gross_loss = abs(losses['pnl'].sum() * losses.get('quantity', 1).mean()) # Approximate

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        avg_trade_pnl = trades_df['pnl'].mean()
        avg_win_pnl = wins['pnl'].mean() if num_wins > 0 else 0
        avg_loss_pnl = losses['pnl'].mean() if num_losses > 0 else 0

        avg_win_pct = wins['pnl_pct'].mean() if num_wins > 0 else 0
        avg_loss_pct = losses['pnl_pct'].mean() if num_losses > 0 else 0

        avg_holding_days = trades_df['holding_days'].mean()
        avg_win_holding_days = wins['holding_days'].mean() if num_wins > 0 else 0
        avg_loss_holding_days = losses['holding_days'].mean() if num_losses > 0 else 0

    else:
        num_trades = num_wins = num_losses = 0
        win_rate_pct = 0
        gross_profit = gross_loss = profit_factor = 0
        avg_trade_pnl = avg_win_pnl = avg_loss_pnl = 0
        avg_win_pct = avg_loss_pct = 0
        avg_holding_days = avg_win_holding_days = avg_loss_holding_days = 0

    # --- Risk/Return Ratios (Annualized) ---
    portfolio_df['daily_return'] = portfolio_df['value'].pct_change()

    if portfolio_df['daily_return'].std() > 0:
        # Assuming 252 trading days for annualization
        trading_days = len(portfolio_df)
        days_in_year = 365.25
        trading_days_per_year = (trading_days / num_days) * days_in_year if num_days > 0 else 252 # Estimate trading days per year

        avg_daily_return = portfolio_df['daily_return'].mean()
        std_daily_return = portfolio_df['daily_return'].std()

        annualized_volatility_pct = std_daily_return * np.sqrt(trading_days_per_year) * 100

        # Assuming 0% risk-free rate for simplicity
        rf_rate_daily = 0

        sharpe_ratio = (avg_daily_return - rf_rate_daily) / std_daily_return * np.sqrt(trading_days_per_year)

        # Sortino Ratio
        downside_returns = portfolio_df['daily_return'].copy()
        downside_returns[downside_returns > 0] = 0
        downside_std = downside_returns.std()

        if downside_std > 0:
            sortino_ratio = (avg_daily_return - rf_rate_daily) / downside_std * np.sqrt(trading_days_per_year)
        else:
            sortino_ratio = np.inf
    else:
        annualized_volatility_pct = 0
        sharpe_ratio = np.inf
        sortino_ratio = np.inf

    return {
        "--- Overview ---": "",
        "Start Date": start_date.strftime('%Y-%m-%d'),
        "End Date": end_date.strftime('%Y-%m-%d'),
        "Duration (Days)": num_days,
        "Initial Capital": f"₹{initial_capital:,.2f}",
        "Final Portfolio Value": f"₹{final_total_value:,.2f}",
        "Net Profit": f"₹{net_profit:,.2f}",
        "Total Return": f"{total_return_pct:.2f}%",
        "Annualized Return": f"{annualized_return_pct:.2f}%",
        "Max Drawdown": f"{max_drawdown_pct:.2f}%",

        "--- Risk Ratios ---": "",
        "Annualized Volatility": f"{annualized_volatility_pct:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.3f}",
        "Sortino Ratio": f"{sortino_ratio:.3f}",

        "--- Trade Stats ---": "",
        "Total Trades": num_trades,
        "Winning Trades": num_wins,
        "Losing Trades": num_losses,
        "Win Rate": f"{win_rate_pct:.2f}%",
        "Profit Factor": f"{profit_factor:.2f}" if gross_loss > 0 else "inf",
        "Avg. Trade P&L": f"₹{avg_trade_pnl:.2f}" if num_trades > 0 else "N/A",
        "Avg. Win": f"₹{avg_win_pnl:.2f}" if num_wins > 0 else "N/A",
        "Avg. Loss": f"₹{avg_loss_pnl:.2f}" if num_losses > 0 else "N/A",
        "Avg. Win %": f"{avg_win_pct:.2f}%" if num_wins > 0 else "N/A",
        "Avg. Loss %": f"{avg_loss_pct:.2f}%" if num_losses > 0 else "N/A",

        "--- Holding Period ---": "",
        "Avg. Holding Days": f"{avg_holding_days:.1f}" if num_trades > 0 else "N/A",
        "Avg. Win Holding Days": f"{avg_win_holding_days:.1f}" if num_wins > 0 else "N/A",
        "Avg. Loss Holding Days": f"{avg_loss_holding_days:.1f}" if num_losses > 0 else "N/A",
    }


def backtest_strategy(data_dict, initial_capital, chunk_percentage, investment_frequency, profit_target, initial_stop_loss_pct, trailing_stop_loss_pct, start_date, end_date):
    """
    Runs the swing strategy backtest with uptrend filter, initial stop-loss, and trailing stop-loss.

    :param data_dict: Dictionary of {symbol: pd.DataFrame} with OHLC data.
    :param initial_capital: Total starting capital.
    :param chunk_percentage: Pct of *initial* capital for each investment lot.
    :param investment_frequency: Pandas offset string (e.g., 'W-MON' for weekly Monday).
    :param profit_target: Pct profit to activate trailing stop (e.g., 0.05 for 5%).
    :param initial_stop_loss_pct: Pct below entry price for initial stop-loss (e.g., 0.03 for 3%).
    :param trailing_stop_loss_pct: Pct below highest price for trailing stop (e.g., 0.03 for 3%).
    :param start_date: Backtest start date (str or datetime).
    :param end_date: Backtest end date (str or datetime).
    """

    # --- 1. Initialization ---
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    open_lots = []
    trades = []
    portfolio_history = []

    available_capital = initial_capital
    booked_profit = 0
    chunk_capital = initial_capital * chunk_percentage

    symbols = list(data_dict.keys())
    symbol_index = 0

    # --- 2. Data Preparation ---
    # Create a master date range for the entire backtest period
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B') # 'B' for business days
    master_df = pd.DataFrame(index=all_dates)

    # Align all data to the master date range and forward-fill missing values
    processed_data = {}
    for symbol, df in data_dict.items():
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        # Use 'Close' price as it's the standard for NAV
        df = df[['Close']].reindex(all_dates, method='ffill')
        df = df.ffill().bfill() # Fill any remaining NaNs

        # Calculate 50-day SMA for trend determination
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        # Trend: 'up' if Close > SMA_50, else 'down'
        df['trend'] = np.where(df['Close'] > df['SMA_50'], 'up', 'down')

        processed_data[symbol] = df

    # Get scheduled investment dates
    investment_schedule = pd.date_range(start=start_date, end=end_date, freq=investment_frequency)

    print(f"Backtest running from {start_date.date()} to {end_date.date()}...")
    print(f"Initial Capital: ₹{initial_capital:,.2f}")
    print(f"Investment Chunk: ₹{chunk_capital:,.2f} (deployed every {investment_frequency})")
    print(f"Profit Target: {profit_target * 100:.1f}% (to activate trailing stop)")
    print(f"Initial Stop Loss: {initial_stop_loss_pct * 100:.1f}% below entry price")
    print(f"Trailing Stop Loss: {trailing_stop_loss_pct * 100:.1f}% below highest price")
    print("Uptrend Filter: Invest only when Close > 50-day SMA")
    print("-" * 30)

    # --- 3. Main Backtest Loop ---
    for date in all_dates:
        if date > end_date:
            break

        current_portfolio_value = available_capital + booked_profit

        # --- 3a. Check Exits (FIFO) ---
        # Iterate over a copy as we modify the list
        for lot in open_lots[:]:
            symbol = lot['symbol']
            current_price = processed_data[symbol].loc[date, 'Close']

            # Update highest price
            lot['highest_price'] = max(lot['highest_price'], current_price)

            # Check initial stop-loss first
            initial_stop_price = lot['entry_price'] * (1 - initial_stop_loss_pct)
            if current_price <= initial_stop_price:
                # --- INITIAL STOP-LOSS HIT ---
                exit_price = current_price
                profit = (exit_price - lot['entry_price']) * lot['quantity']

                booked_profit += profit
                available_capital += lot['lot_capital']

                trades.append({
                    'symbol': symbol,
                    'entry_date': lot['entry_date'],
                    'exit_date': date,
                    'entry_price': lot['entry_price'],
                    'exit_price': exit_price,
                    'quantity': lot['quantity'],
                    'status': 'LOSS (Initial SL)'
                })

                open_lots.remove(lot)
                continue  # Skip to next lot

            # Activate trailing stop if profit target reached and not already activated
            if current_price >= lot['entry_price'] * (1 + profit_target) and lot.get('trailing_stop') is None:
                lot['trailing_stop'] = lot['highest_price'] * (1 - trailing_stop_loss_pct)

            # Update trailing stop if activated
            if lot.get('trailing_stop') is not None:
                lot['trailing_stop'] = lot['highest_price'] * (1 - trailing_stop_loss_pct)

            # Check for exit via trailing stop
            if lot.get('trailing_stop') is not None and current_price <= lot['trailing_stop']:
                # --- TRAILING STOP HIT ---
                exit_price = current_price
                profit = (exit_price - lot['entry_price']) * lot['quantity']

                booked_profit += profit
                available_capital += lot['lot_capital']

                trades.append({
                    'symbol': symbol,
                    'entry_date': lot['entry_date'],
                    'exit_date': date,
                    'entry_price': lot['entry_price'],
                    'exit_price': exit_price,
                    'quantity': lot['quantity'],
                    'status': 'WIN (Trailing Stop)' if profit > 0 else 'LOSS (Trailing Stop)'
                })

                open_lots.remove(lot)
                continue  # Skip to next lot

            # Update current portfolio value with open lots
            current_portfolio_value += lot['quantity'] * current_price

        # --- 3b. Check Entries ---
        if date in investment_schedule:
            if available_capital >= chunk_capital:
                # Check if in uptrend
                symbol_to_check = symbols[symbol_index % len(symbols)]
                current_trend = processed_data[symbol_to_check].loc[date, 'trend']

                if current_trend == 'up':
                    # --- INVEST A NEW CHUNK ---
                    symbol_to_invest = symbol_to_check
                    symbol_index += 1

                    entry_price = processed_data[symbol_to_invest].loc[date, 'Close']

                    if pd.notna(entry_price) and entry_price > 0:
                        quantity = chunk_capital / entry_price

                        new_lot = {
                            'symbol': symbol_to_invest,
                            'entry_date': date,
                            'entry_price': entry_price,
                            'quantity': quantity,
                            'lot_capital': chunk_capital,
                            'highest_price': entry_price,
                            'trailing_stop': None,
                        }

                        open_lots.append(new_lot)
                        available_capital -= chunk_capital

                        # Add this lot's value to portfolio *after* purchase
                        current_portfolio_value += new_lot['quantity'] * entry_price

                        print(f"Invested ₹{chunk_capital:,.2f} in {symbol_to_invest} on {date.date()} (Uptrend)")
                else:
                    print(f"Skipped investment on {date.date()} - Not in uptrend (Trend: {current_trend})")
                    # Still increment symbol_index to cycle through symbols
                    symbol_index += 1

            else:
                # print(f"Warning: Not enough capital to invest on {date.date()}")
                pass

        # --- 3c. Record Daily Portfolio Value ---
        # Re-calculate value one last time for non-investment, non-exit days
        if not (date in investment_schedule):
            current_portfolio_value = available_capital + booked_profit
            for lot in open_lots:
                current_price = processed_data[lot['symbol']].loc[date, 'Close']
                current_portfolio_value += lot['quantity'] * current_price

        portfolio_history.append({'date': date, 'value': current_portfolio_value})

    # --- 4. End of Backtest: Liquidate Open Positions ---
    print(f"\nBacktest complete. Liquidating {len(open_lots)} open positions on {end_date.date()}...")

    final_price_date = all_dates[-1]

    for lot in open_lots:
        symbol = lot['symbol']
        exit_price = processed_data[symbol].loc[final_price_date, 'Close']

        pnl = (exit_price - lot['entry_price']) * lot['quantity']

        if pnl > 0:
            booked_profit += pnl
            status = 'WIN (Forced)'
        else:
            # Realize the loss from booked_profit
            booked_profit += pnl # pnl is negative, so this is subtraction
            status = 'LOSS (Forced)'

        available_capital += lot['lot_capital'] # Return original capital

        trades.append({
            'symbol': symbol,
            'entry_date': lot['entry_date'],
            'exit_date': final_price_date,
            'entry_price': lot['entry_price'],
            'exit_price': exit_price,
            'quantity': lot['quantity'],
            'status': status
        })

    open_lots = []
    final_total_value = available_capital + booked_profit

    # Add final value to history
    portfolio_history.append({'date': final_price_date, 'value': final_total_value})

    print("Liquidation complete.")

    # --- 5. Calculate Metrics ---
    metrics = calculate_metrics(portfolio_history, trades, initial_capital, end_date, start_date)

    return metrics, pd.DataFrame(portfolio_history), pd.DataFrame(trades)


# --- EXAMPLE USAGE ---
if __name__ == "__main__":

    # --- 1. Load Your Real Data (REPLACING MOCK DATA) ---

    # Define file path - using 'r' prefix for Windows path
    file_path = r"D:\\Sushant\\Fyers_AlgoTrade\\Fyers_Data\\NIFTYBEES\\NIFTYBEES_EQ_D_Min.csv"

    # Define column names as specified
    column_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    try:
        # Load the CSV
        # We assume the file does *not* have a header row based on your prompt.
        # If your file *does* have a header row, change 'header=None' to 'header=0'
        # and remove 'names=column_names'.
        df_niftybees = pd.read_csv(
            file_path,
            header=0,
            parse_dates=['Date']
        )

        # --- 1b. Prepare Data for Backtest ---

        # Create the data dictionary required by the backtester
        data_to_backtest = {
            'NIFTYBEES': df_niftybees
        }

        # Get start and end dates dynamically from your data
        backtest_start = df_niftybees['Date'].min()
        backtest_end = df_niftybees['Date'].max()

        print(f"Successfully loaded data for NIFTYBEES.")
        print(f"Backtest range set: {backtest_start.date()} to {backtest_end.date()}")

    except FileNotFoundError:
        print(f"--- ERROR ---")
        print(f"File not found at path: {file_path}")
        print("Please check the file path and try again.")
        print("Exiting.")
        exit()
    except Exception as e:
        print(f"--- ERROR ---")
        print(f"An error occurred while loading the data: {e}")
        print("Please ensure pandas is installed (`pip install pandas`) and the file is a valid CSV.")
        print("Exiting.")
        exit()


    # --- 2. Set Strategy Parameters ---
    params = {
        'data_dict': data_to_backtest,       # Use your loaded data
        'initial_capital': 1000000,          # ₹10 Lakhs
        'chunk_percentage': 0.05,            # 5% of total capital per chunk (₹50,000)
        'investment_frequency': 'W-MON',     # Invest every Monday
        'profit_target': 0.05,               # 5% profit target to activate trailing stop
        'initial_stop_loss_pct': 0.03,       # 3% initial stop-loss below entry price
        'trailing_stop_loss_pct': 0.03,      # 3% trailing stop below highest price
        'start_date': backtest_start,      # Use dynamic start date
        'end_date': backtest_end          # Use dynamic end date
    }

    # --- 3. Run the Backtest ---
    metrics, portfolio_df, trades_df = backtest_strategy(**params)

    # --- 4. Print Results ---
    print("\n" + "=" * 30)
    print("      BACKTEST RESULTS")
    print("=" * 30)

    for key, value in metrics.items():
        if "---" in key:
            print(f"\n{key}")
        else:
            print(f"{key:<25}: {value}")

    # Optional: You can inspect the detailed DataFrames
    # print("\n--- Portfolio History (Sample) ---")
    # print(portfolio_df.tail())
    # print("\n--- Trades History (Sample) ---")
    # print(trades_df.tail())
