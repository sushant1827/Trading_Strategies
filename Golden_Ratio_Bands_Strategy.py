import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def golden_ratio_bands_strategy(data: pd.DataFrame, window: int = 390, upper_ratio: float = 0.618, lower_ratio: float = 0.382) -> pd.DataFrame:
    """
    Golden Ratio Bands strategy using rolling high/low over a window.

    Parameters:
        - data: DataFrame with OHLC data (must contain 'Close', 'High', 'Low')
        - window: Rolling window to compute high/low range (default=390, 1 trading day in minutes)
        - upper_ratio: Golden ratio for upper band (default=0.618)
        - lower_ratio: Golden ratio for lower band (default=0.382)

    Returns:
        - DataFrame with signals, returns, and cumulative performance
    """
    df = data.copy()

    # Validate columns
    if not {'Close', 'High', 'Low'}.issubset(df.columns):
        raise ValueError("Input data must contain 'High', 'Low', and 'Close' columns.")

    # Rolling high-low bands
    df['Rolling_High'] = df['High'].rolling(window).max()
    df['Rolling_Low'] = df['Low'].rolling(window).min()
    df['Diff'] = df['Rolling_High'] - df['Rolling_Low']
    
    # Calculate upper and lower bands
    df['Upper_Band'] = df['Rolling_High'] - df['Diff'] * upper_ratio
    df['Lower_Band'] = df['Rolling_Low'] + df['Diff'] * lower_ratio

    # Generate signals
    df['Signal'] = 0
    df.loc[df['Close'] > df['Upper_Band'], 'Signal'] = 1
    df.loc[df['Close'] < df['Lower_Band'], 'Signal'] = 0

    # Maintain position until opposite signal
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0)

    # Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    df['Strategy_Return'] = df['Position'].shift(1).fillna(0) * df['Log_Return']
    df['Cumulative_Strategy_Return'] = df['Strategy_Return'].cumsum()
    df['Cumulative_Market_Return'] = df['Log_Return'].cumsum()

    return df


def compute_backtest_metrics(df: pd.DataFrame) -> dict:
    """
    Computes Sharpe Ratio, Max Drawdown, Win Rate, and Total Return for strategy.
    """
    result = {}

    # Total return
    result['Total Return (%)'] = df['Cumulative_Strategy_Return'].iloc[-1] * 100

    # Sharpe Ratio (annualized, minute-level data)
    annual_factor = np.sqrt(98480)
    sharpe = df['Strategy_Return'].mean() / df['Strategy_Return'].std()
    result['Sharpe Ratio'] = sharpe * annual_factor if df['Strategy_Return'].std() != 0 else np.nan

    # Max Drawdown
    cum_returns = df['Cumulative_Strategy_Return']
    peak = cum_returns.cummax()
    drawdown = cum_returns - peak
    result['Max Drawdown (%)'] = drawdown.min() * 100

    # Win Rate
    wins = (df['Strategy_Return'] > 0).sum()
    trades = (df['Signal'] == 1).sum()
    result['Win Rate (%)'] = (wins / trades) * 100 if trades > 0 else np.nan

    return result


def plot_strategy(df: pd.DataFrame, title: str = "Golden Ratio Bands Strategy"):
    """
    Plots the strategy chart with bands and cumulative returns.
    """
    plt.figure(figsize=(14, 7))

    ax1 = plt.subplot(2, 1, 1)
    df['Close'].plot(label='Close', alpha=0.6)
    df['Upper_Band'].plot(label='Upper Band', linestyle='--')
    df['Lower_Band'].plot(label='Lower Band', linestyle='--')
    plt.title(title + " - Price with Bands")
    plt.legend()

    ax2 = plt.subplot(2, 1, 2)
    df['Cumulative_Market_Return'].plot(label='Market Return')
    df['Cumulative_Strategy_Return'].plot(label='Strategy Return')
    plt.title("Cumulative Returns")
    plt.legend()

    plt.tight_layout()
    plt.show()


def resample_to_15min(data: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples 1-minute OHLC data to 15-minute intervals.

    Parameters:
        - data: DataFrame with 1-min OHLC data indexed by datetime

    Returns:
        - Resampled DataFrame with 15-min OHLCV
    """
    ohlc = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum' if 'Volume' in data.columns else 'first'
    }
    data_15min = data.resample('15T').agg(ohlc).dropna()
    return data_15min


# ------------------- Example Usage -------------------

if __name__ == "__main__":
    # Load raw 1-min data
    data = pd.read_csv("AAPL_1min.csv", parse_dates=True, index_col=0)

    # Convert index to datetime if not already
    data.index = pd.to_datetime(data.index)

    # Resample to 15-min intervals
    data_15min = resample_to_15min(data)

    # Run the Golden Ratio Bands strategy
    result_df = golden_ratio_bands_strategy(data_15min, window=26)  # 26 * 15min = ~1 trading day

    # Compute and print metrics
    metrics = compute_backtest_metrics(result_df)
    print("\nBacktest Metrics on 15-Minute Data:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    # Plot
    plot_strategy(result_df, title="Golden Ratio Bands Strategy (15-Min)")

