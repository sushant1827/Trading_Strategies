import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

# Parameters
ATR_LEN = 10
FACT = 3.0
TRAINING_DATA_PERIOD = 100
HIGH_VOL_PERCENTILE = 0.75
MID_VOL_PERCENTILE = 0.5
LOW_VOL_PERCENTILE = 0.25

def calculate_heikin_ashi(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = np.zeros(len(df))
    ha_open[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
    ha_high = np.maximum(df['High'], np.maximum(ha_open, ha_close))
    ha_low = np.minimum(df['Low'], np.minimum(ha_open, ha_close))
    df['HA_Open'] = ha_open
    df['HA_High'] = ha_high
    df['HA_Low'] = ha_low
    df['HA_Close'] = ha_close
    return df

def load_data(filepath):
    df = pd.read_csv(filepath, index_col=0, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']]
    df = calculate_heikin_ashi(df)
    # df = df.head(1000)  # For testing
    return df

def calculate_atr(df):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    df['atr'] = tr.ewm(span=ATR_LEN, adjust=False).mean()
    return df

def kmeans_volatility_clustering(atr_series, high_vol, mid_vol, low_vol):
    # Initial centroids
    centroids = [high_vol, mid_vol, low_vol]
    prev_centroids = [0, 0, 0]
    atr_list = [val for val in atr_series.tolist() if not np.isnan(val)]
    while centroids != prev_centroids:
        prev_centroids = centroids.copy()
        clusters = [[], [], []]
        for val in atr_list:
            dists = [abs(val - c) for c in centroids]
            cluster_idx = dists.index(min(dists))
            clusters[cluster_idx].append(val)
        centroids = [np.mean(clusters[0]) if clusters[0] else prev_centroids[0],
                     np.mean(clusters[1]) if clusters[1] else prev_centroids[1],
                     np.mean(clusters[2]) if clusters[2] else prev_centroids[2]]
    return np.array(centroids)

def assign_cluster(current_atr, centroids):
    distances = np.abs(centroids - current_atr)
    cluster = np.argmin(distances)
    return cluster, centroids[cluster]

def generate_signals(df):
    df = calculate_atr(df)
    n = len(df)
    high = df['HA_High'].values
    low = df['HA_Low'].values
    close = df['HA_Close'].values
    atr = df['atr'].values
    src = (high + low) / 2
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    super_trend = np.full(n, np.nan)
    signal = np.zeros(n)

    for i in range(TRAINING_DATA_PERIOD, n):
        if i % 100 == 0:
            print(f"Processing signal generation: {i}/{n}")
        
        # 1. Get the historical ATR data for the window ending at i-1
        hist_atr = atr[i - TRAINING_DATA_PERIOD : i]

        # 2. Calculate initial guesses based on the historical window
        hist_atr_clean = [val for val in hist_atr if not np.isnan(val)]
        if not hist_atr_clean:
            continue

        upper_val = np.max(hist_atr_clean)
        lower_val = np.min(hist_atr_clean)
        high_vol = lower_val + (upper_val - lower_val) * HIGH_VOL_PERCENTILE
        mid_vol = lower_val + (upper_val - lower_val) * MID_VOL_PERCENTILE
        low_vol = lower_val + (upper_val - lower_val) * LOW_VOL_PERCENTILE

        # 3. Perform K-Means on the historical data to get converged centroids
        centroids = kmeans_volatility_clustering(pd.Series(hist_atr_clean), high_vol, mid_vol, low_vol)

        # 4. Use the converged centroids to classify the current bar's ATR
        if np.isnan(atr[i]):
            continue
            
        cluster, assigned_atr = assign_cluster(atr[i], centroids)

        # The rest of the SuperTrend and signal generation logic remains the same
        upper_band[i] = src[i] + FACT * assigned_atr
        lower_band[i] = src[i] - FACT * assigned_atr

        if i == TRAINING_DATA_PERIOD:
            prev_lower = lower_band[i]
            prev_upper = upper_band[i]
            prev_super = np.nan
        else:
            prev_lower = lower_band[i-1]
            prev_upper = upper_band[i-1]
            prev_super = super_trend[i-1]

        lower_band[i] = lower_band[i] if (lower_band[i] > prev_lower or close[i-1] < prev_lower) else prev_lower
        upper_band[i] = upper_band[i] if (upper_band[i] < prev_upper or close[i-1] > prev_upper) else prev_upper

        if np.isnan(atr[i-1]):
            direction[i] = 1
        elif prev_super == prev_upper:
            direction[i] = -1 if close[i] > upper_band[i] else 1
        else:
            direction[i] = 1 if close[i] < lower_band[i] else -1

        super_trend[i] = lower_band[i] if direction[i] == -1 else upper_band[i]

        # Signal
        if i > TRAINING_DATA_PERIOD:
            prev_dir = direction[i-1]
            curr_dir = direction[i]
            if not np.isnan(prev_dir) and prev_dir != curr_dir:
                if curr_dir == -1:  # Bullish
                    signal[i] = 1
                elif curr_dir == 1:  # Bearish
                    signal[i] = -1

    df['signal'] = signal
    df['super_trend'] = super_trend
    df['direction'] = direction
    print("Signal generation complete")
    return df

class AdaptiveSuperTrendStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        signal = self.data.signal[-1]
        if signal == 1:
            if self.position.is_short:
                self.position.close()
            self.buy(size=1)
        elif signal == -1:
            if self.position.is_long:
                self.position.close()
            self.sell(size=1)

def run_backtest(df):
    print("Running backtest...")
    bt = Backtest(df, AdaptiveSuperTrendStrategy, cash=1000000, commission=.002)
    stats = bt.run()
    print("Backtest complete")
    print(stats)
    # Save trades to CSV
    trades_df = bt._results._trades
    trades_df.to_csv('adaptive_supertrend_trades.csv', index=False)
    print("Trades saved to adaptive_supertrend_trades.csv")
    # bt.plot()

if __name__ == "__main__":
    filepath = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'
    df = load_data(filepath)
    df = generate_signals(df)
    run_backtest(df)