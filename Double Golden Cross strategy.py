import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def double_golden_cross_strategy(data: pd.DataFrame, fast_period: int = 8, slow_period: int = 21) -> pd.DataFrame:
    """
    Double Golden Cross Strategy with signal generation and return calculation.
    """
    df = data.copy()

    # Ensure 'Close' column exists
    if 'Close' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    # Calculate MAs
    df['Fast_MA'] = df['Close'].rolling(window=fast_period, min_periods=1).mean()
    df['Slow_MA'] = df['Close'].rolling(window=slow_period, min_periods=1).mean()

    # Generate signals
    df['Signal'] = 0
    df.loc[(df['Fast_MA'] > df['Slow_MA']) & (df['Fast_MA'].shift(1) <= df['Slow_MA'].shift(1)), 'Signal'] = 1
    df.loc[(df['Fast_MA'] < df['Slow_MA']) & (df['Fast_MA'].shift(1) >= df['Slow_MA'].shift(1)), 'Signal'] = 0

    # Position: maintain until next crossover
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0)

    # Log returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)

    # Strategy returns
    df['Strategy_Return'] = df['Position'].shift(1).fillna(0) * df['Log_Return']
    df['Cumulative_Strategy_Return'] = df['Strategy_Return'].cumsum()
    df['Cumulative_Market_Return'] = df['Log_Return'].cumsum()

    return df

def compute_backtest_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate Sharpe Ratio, Max Drawdown, and Total Return from strategy returns.
    """
    result = {}

    # Total Return
    result['Total Return (%)'] = df['Cumulative_Strategy_Return'].iloc[-1] * 100

    # Sharpe Ratio (assuming 252 trading days, 390 mins per day = 98480 minutes/year)
    annual_factor = np.sqrt(98480)
    result['Sharpe Ratio'] = (df['Strategy_Return'].mean() / df['Strategy_Return'].std()) * annual_factor

    # Max Drawdown
    cumulative = df['Cumulative_Strategy_Return']
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    result['Max Drawdown (%)'] = drawdown.min() * 100

    # Win Rate
    wins = (df['Strategy_Return'] > 0).sum()
    trades = (df['Signal'] == 1).sum()
    result['Win Rate (%)'] = (wins / trades) * 100 if trades else np.nan

    return result

def plot_strategy(df: pd.DataFrame, title: str = "Double Golden Cross Strategy"):
    """
    Plot cumulative returns and moving averages.
    """
    plt.figure(figsize=(14, 7))

    ax1 = plt.subplot(2, 1, 1)
    df['Close'].plot(label='Close Price', alpha=0.5)
    df['Fast_MA'].plot(label='Fast MA')
    df['Slow_MA'].plot(label='Slow MA')
    plt.title(title + " - Price & MAs")
    plt.legend()

    ax2 = plt.subplot(2, 1, 2)
    df['Cumulative_Market_Return'].plot(label='Market Return')
    df['Cumulative_Strategy_Return'].plot(label='Strategy Return')
    plt.title("Cumulative Returns")
    plt.legend()

    plt.tight_layout()
    plt.show()

# ------------------- Example Usage -------------------

if __name__ == "__main__":
    # Load OHLCV data
    data = pd.read_csv('AAPL_1min.csv', parse_dates=True, index_col=0)

    # Apply strategy
    result_df = double_golden_cross_strategy(data)

    # Compute metrics
    metrics = compute_backtest_metrics(result_df)

    # Print metrics
    print("Backtest Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    # Plot
    plot_strategy(result_df)
