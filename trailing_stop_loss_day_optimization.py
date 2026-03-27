#!/usr/bin/env python3
"""
Day of Week Optimization for Trailing Stop-Loss Strategy

This module tests different days of the week for investment timing to find the optimal day
for the trailing stop-loss strategy. Tests Monday through Friday weekly investments.

Uses the optimized parameters from grid search optimization.
"""

import pandas as pd
import numpy as np
from trailing_stop_loss import TrailingStopLoss, calculate_ema, calculate_trading_metrics


class DayOptimizedTrailingStopStrategy:
    """
    Trailing stop-loss strategy optimized for different days of the week.
    """

    def __init__(self, investment_frequency='W-MON'):
        # Optimized parameters from grid search
        self.initial_target_pct = 0.05      # 5% initial profit target
        self.trail_pct = 0.02               # 2% trailing percentage
        self.chunk_percentage = 0.1        # 10% of capital per investment
        self.investment_frequency = investment_frequency
        self.ema_short_period = 15          # 15-day EMA
        self.ema_long_period = 40           # 40-day EMA

    def prepare_data(self, df):
        """
        Prepare data with technical indicators.

        Args:
            df (pd.DataFrame): Raw price data with 'Date' and 'Close' columns

        Returns:
            pd.DataFrame: Data with added technical indicators
        """
        data_df = df.copy()

        # Create master date range
        start_date = data_df['Date'].min()
        end_date = data_df['Date'].max()
        all_dates = pd.date_range(start=start_date, end=end_date, freq='B')

        data_df = data_df.set_index('Date').reindex(all_dates, method='ffill').ffill().bfill()

        # Add optimized EMA indicators
        data_df['EMA_short'] = calculate_ema(data_df['Close'], self.ema_short_period)
        data_df['EMA_long'] = calculate_ema(data_df['Close'], self.ema_long_period)
        data_df['Uptrend'] = data_df['EMA_short'] > data_df['EMA_long']

        return data_df

    def backtest(self, df, initial_capital=100000):
        """
        Run backtest with optimized parameters.

        Args:
            df (pd.DataFrame): Historical price data
            initial_capital (float): Starting capital

        Returns:
            dict: Performance metrics and trades
        """
        data_df = self.prepare_data(df)

        start_date = data_df.index.min()
        end_date = data_df.index.max()

        open_lots = []
        trades = []
        portfolio_history = []

        available_capital = initial_capital
        booked_profit = 0
        chunk_capital = initial_capital * self.chunk_percentage

        investment_schedule = pd.date_range(start=start_date, end=end_date, freq=self.investment_frequency)

        for date in data_df.index:
            current_portfolio_value = available_capital + booked_profit

            # Check exits with trailing stops
            for lot in open_lots[:]:
                current_price = data_df.loc[date, 'Close']
                tsl = lot['trailing_stop']
                result = tsl.update_price(current_price)

                if result['action'] == 'sell':
                    exit_price = current_price
                    profit = (exit_price - lot['entry_price']) * lot['quantity']

                    booked_profit += profit
                    available_capital += lot['lot_capital']

                    trades.append({
                        'entry_date': lot['entry_date'],
                        'exit_date': date,
                        'entry_price': lot['entry_price'],
                        'exit_price': exit_price,
                        'quantity': lot['quantity'],
                        'pnl': profit,
                        'status': 'WIN' if profit > 0 else 'LOSS'
                    })

                    open_lots.remove(lot)
                else:
                    current_portfolio_value += lot['quantity'] * current_price

            # Check entries (only in uptrend)
            if date in investment_schedule and available_capital >= chunk_capital:
                entry_price = data_df.loc[date, 'Close']
                is_uptrend = data_df.loc[date, 'Uptrend']

                if pd.notna(entry_price) and entry_price > 0 and is_uptrend:
                    quantity = chunk_capital / entry_price
                    tsl = TrailingStopLoss(entry_price, quantity, self.initial_target_pct, self.trail_pct)

                    new_lot = {
                        'entry_date': date,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'lot_capital': chunk_capital,
                        'trailing_stop': tsl
                    }

                    open_lots.append(new_lot)
                    available_capital -= chunk_capital
                    current_portfolio_value += new_lot['quantity'] * entry_price

            portfolio_history.append({'date': date, 'value': current_portfolio_value})

        # Liquidate remaining positions
        final_price_date = data_df.index[-1]

        for lot in open_lots:
            exit_price = data_df.loc[final_price_date, 'Close']
            pnl = (exit_price - lot['entry_price']) * lot['quantity']

            booked_profit += pnl
            available_capital += lot['lot_capital']

            trades.append({
                'entry_date': lot['entry_date'],
                'exit_date': final_price_date,
                'entry_price': lot['entry_price'],
                'exit_price': exit_price,
                'quantity': lot['quantity'],
                'pnl': pnl,
                'status': 'WIN (Forced)' if pnl > 0 else 'LOSS (Forced)'
            })

        final_total_value = available_capital + booked_profit
        portfolio_history.append({'date': final_price_date, 'value': final_total_value})

        metrics = calculate_trading_metrics(portfolio_history, trades, initial_capital, end_date, start_date)

        return {
            'metrics': metrics,
            'trades': trades,
            'portfolio_history': portfolio_history,
            'parameters': {
                'initial_target_pct': self.initial_target_pct,
                'trail_pct': self.trail_pct,
                'chunk_percentage': self.chunk_percentage,
                'investment_frequency': self.investment_frequency,
                'ema_short_period': self.ema_short_period,
                'ema_long_period': self.ema_long_period
            }
        }


def optimize_investment_day():
    """
    Test different days of the week for investment timing and find the optimal day.
    """
    # Load NIFTYBEES data
    file_path = r"NIFTYBEES/NIFTYBEES_EQ_D_Min.csv"

    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        print(f"Loaded data: {len(df)} records from {df['Date'].min().date()} to {df['Date'].max().date()}")
    except FileNotFoundError:
        print(f"Data file not found at {file_path}")
        return None

    print("\n" + "="*100)
    print("DAY OF WEEK OPTIMIZATION FOR TRAILING STOP STRATEGY")
    print("="*100)

    # Days to test
    days_to_test = ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI']
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    results = {}

    print("\nTESTING DIFFERENT INVESTMENT DAYS:")
    print("-" * 50)

    for freq, day_name in zip(days_to_test, day_names):
        print(f"\nTesting {day_name} ({freq}):")

        strategy = DayOptimizedTrailingStopStrategy(investment_frequency=freq)
        result = strategy.backtest(df)

        results[day_name] = result

        metrics = result['metrics']
        print(f"  Total Return: {metrics['Total Return']}")
        print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']}")
        print(f"  Max Drawdown: {metrics['Max Drawdown']}")
        print(f"  Win Rate: {metrics['Win Rate']}")
        print(f"  Total Trades: {len(result['trades'])}")

    # Find best performing day
    best_day = max(results.keys(), key=lambda x: results[x]['metrics']['Sharpe Ratio'])
    best_result = results[best_day]

    print("\n" + "="*100)
    print(f"BEST PERFORMING DAY: {best_day.upper()}")
    print("="*100)

    print(f"\nOPTIMIZED PARAMETERS FOR {best_day.upper()}:")
    for param, value in best_result['parameters'].items():
        if 'pct' in param:
            print(f"  {param}: {value:.1%}")
        else:
            print(f"  {param}: {value}")

    print(f"\nPERFORMANCE METRICS FOR {best_day.upper()}:")
    for key, value in best_result['metrics'].items():
        print(f"{key:<20}: {value}")

    print(f"\nTotal Trades: {len(best_result['trades'])}")

    # Save best results
    trades_df = pd.DataFrame(best_result['trades'])
    trades_df.to_csv(f'optimized_trailing_stop_{best_day.lower()}_trades.csv', index=False)

    portfolio_df = pd.DataFrame(best_result['portfolio_history'])
    portfolio_df.to_csv(f'optimized_trailing_stop_{best_day.lower()}_portfolio.csv', index=False)

    print(f"\nBest results saved to:")
    print(f"- optimized_trailing_stop_{best_day.lower()}_trades.csv")
    print(f"- optimized_trailing_stop_{best_day.lower()}_portfolio.csv")

    # Summary comparison
    print("\n" + "="*100)
    print("DAY-BY-DAY COMPARISON SUMMARY")
    print("="*100)
    print(f"{'Day':<12} {'Total Return':<15} {'Sharpe Ratio':<15} {'Max Drawdown':<15} {'Win Rate':<12} {'Trades':<8}")
    print("-" * 80)

    for day in day_names:
        metrics = results[day]['metrics']
        print(f"{day:<12} {metrics['Total Return']:<15} {metrics['Sharpe Ratio']:<15} {metrics['Max Drawdown']:<15} {metrics['Win Rate']:<12} {len(results[day]['trades']):<8}")

    return results


if __name__ == "__main__":
    results = optimize_investment_day()
