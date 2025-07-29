import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Data Loading ---
def load_ohlc_data(file_path):
    """
    Loads OHLC data from a CSV file.
    Assumes the CSV has columns like 'Date', 'Open', 'High', 'Low', 'Close'.
    The 'Date' column is converted to datetime and set as the index.
    """
    print(f"Loading OHLC data from {file_path}...")
    df = pd.read_csv(file_path)

    # Rename columns to lowercase for consistency with existing code
    df.columns = df.columns.str.lower()

    # Ensure 'date' column exists and is parsed as datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif 'time' in df.columns: # Sometimes 'time' is used instead of 'date' for intraday
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    else:
        raise ValueError("CSV must contain a 'date' or 'time' column for the index.")

    # Ensure required OHLC columns exist
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required OHLC columns in CSV: {missing_cols}")

    print(f"Loaded {len(df)} rows of OHLC data.")
    return df[required_cols] # Return only the OHLC columns

# --- 2. Feature Engineering ---

def calculate_technical_indicators(df):
    """
    Calculates various technical indicators and adds them as features to the DataFrame.
    VWAP and VWAP Deviation are removed as they require volume data.
    """
    print("Calculating technical indicators...")

    # Moving Averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()

    # RSI (Relative Strength Index)
    # Calculation adapted from standard RSI formula
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Handle division by zero for avg_loss in RSI calculation
    avg_gain = gain.ewm(com=14 - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=14 - 1, adjust=False).mean()
    # Replace zero avg_loss with a small number to avoid division by zero, then calculate rs
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    # EMA 12, EMA 26, Signal 9
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    window = 20
    df['BB_Middle'] = df['close'].rolling(window=window).mean()
    df['BB_StdDev'] = df['close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)

    # VWAP and VWAP Deviation removed as they require volume data

    return df

def calculate_price_action_features(df):
    """
    Calculates price action features based on past returns and simple candlestick patterns.
    """
    print("Calculating price action features...")

    # Returns over past 5, 10, 15 mins
    df['Return_5min'] = df['close'].pct_change(periods=5) * 100
    df['Return_10min'] = df['close'].pct_change(periods=10) * 100
    df['Return_15min'] = df['close'].pct_change(periods=15) * 100

    # Candlestick Patterns (Simplified)
    # Body size relative to range
    df['Candle_Body_Size'] = abs(df['close'] - df['open'])
    df['Candle_Range'] = df['high'] - df['low']
    df['Candle_Body_Ratio'] = df['Candle_Body_Size'] / df['Candle_Range'].replace(0, np.nan) # Avoid division by zero

    # Upper/Lower shadow size relative to body
    df['Upper_Shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['Lower_Shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['Upper_Shadow_Ratio'] = df['Upper_Shadow'] / df['Candle_Body_Size'].replace(0, np.nan)
    df['Lower_Shadow_Ratio'] = df['Lower_Shadow'] / df['Candle_Body_Size'].replace(0, np.nan)

    # Candlestick type (bullish/bearish)
    df['Candle_Type'] = np.where(df['close'] > df['open'], 1, np.where(df['close'] < df['open'], -1, 0))

    return df

# --- 3. Target Variable Definition ---
def define_target_variable(df, price_increase_threshold=0.05, prediction_horizon_minutes=5):
    """
    Defines the target variable: 1 if price increases by threshold in next X minutes, else 0.
    """
    print(f"Defining target variable (price increase by {price_increase_threshold}% in next {prediction_horizon_minutes} mins)...")
    # Shift future close prices back to the current row
    future_close = df['close'].shift(-prediction_horizon_minutes)
    df['Future_Price_Change_Pct'] = ((future_close - df['close']) / df['close']) * 100
    df['Target'] = (df['Future_Price_Change_Pct'] >= price_increase_threshold).astype(int)
    return df

# --- Main Backtesting and Forward Testing Function ---
def run_trading_strategy(df, price_increase_threshold=0.05, prediction_horizon_minutes=5,
                         train_test_split_ratio=0.8, forward_test_ratio=0.1):
    """
    Executes the trading strategy: feature engineering, model training, backtesting, and forward testing.

    Args:
        df (pd.DataFrame): OHLC DataFrame (no volume).
        price_increase_threshold (float): Percentage threshold for target variable.
        prediction_horizon_minutes (int): How many minutes into the future to predict.
        train_test_split_ratio (float): Ratio for training data (e.g., 0.8 for 80% train, 20% test).
        forward_test_ratio (float): Ratio of data to reserve for forward testing from the end of the dataset.
    """
    print("\n--- Starting Trading Strategy Execution ---")

    # Step 1: Feature Engineering
    df = calculate_technical_indicators(df)
    df = calculate_price_action_features(df)

    # Step 2: Define Target Variable
    df = define_target_variable(df, price_increase_threshold, prediction_horizon_minutes)

    # Drop rows with NaN values introduced by feature/target calculation
    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {initial_rows - len(df)} rows due to NaN values after feature/target calculation.")

    # Define features (X) and target (y)
    # Removed volume-related columns from features
    features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close',
                                                          'Future_Price_Change_Pct', 'Target']]
    X = df[features]
    y = df['Target']

    print(f"\nTotal samples after cleaning: {len(df)}")
    print(f"Features used: {features}")

    # --- Data Splitting for Backtesting and Forward Testing ---
    # Reserve data for forward testing
    total_samples = len(X)
    forward_test_samples = int(total_samples * forward_test_ratio)
    backtest_data_end_idx = total_samples - forward_test_samples

    X_backtest = X.iloc[:backtest_data_end_idx]
    y_backtest = y.iloc[:backtest_data_end_idx]
    X_forward_test = X.iloc[backtest_data_end_idx:]
    y_forward_test = y.iloc[backtest_data_end_idx:]

    print(f"\nBacktesting data size: {len(X_backtest)} samples")
    print(f"Forward testing data size: {len(X_forward_test)} samples")

    # Split backtesting data into train and test sets (time-series split)
    train_size = int(len(X_backtest) * train_test_split_ratio)
    X_train = X_backtest.iloc[:train_size]
    y_train = y_backtest.iloc[:train_size]
    X_test = X_backtest.iloc[train_size:]
    y_test = y_backtest.iloc[train_size:]

    print(f"Backtesting training data size: {len(X_train)} samples")
    print(f"Backtesting testing data size: {len(X_test)} samples")

    # --- Model Ensemble Training ---
    print("\n--- Training Ensemble Models ---")

    # Initialize individual models
    log_reg = LogisticRegression(solver='liblinear', random_state=42)
    rand_forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # Using LightGBM for Gradient Boosting
    gbm = lgb.LGBMClassifier(objective='binary', n_estimators=100, random_state=42, n_jobs=-1)

    # Create an ensemble (VotingClassifier with majority vote)
    # Weights can be adjusted based on individual model performance or domain knowledge
    ensemble_model = VotingClassifier(
        estimators=[
            ('lr', log_reg),
            ('rf', rand_forest),
            ('gbm', gbm)
        ],
        voting='hard' # 'hard' for majority vote, 'soft' for weighted average of probabilities
    )

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)
    print("Ensemble model trained successfully.")

    # --- Backtesting ---
    print("\n--- Running Backtest ---")
    y_pred_backtest = ensemble_model.predict(X_test)

    print("\nBacktest Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_backtest):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_backtest):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_backtest):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_backtest):.4f}")
    print("\nConfusion Matrix (Backtest):")
    print(confusion_matrix(y_test, y_pred_backtest))

    # Simple Backtest Strategy Simulation (for demonstration)
    # This is a very basic simulation, not a full trading backtester
    backtest_results_df = df.iloc[train_size:backtest_data_end_idx].copy()
    backtest_results_df['Predicted_Signal'] = y_pred_backtest
    backtest_results_df['Actual_Target'] = y_test

    # Assume we take a position if Predicted_Signal is 1
    # Calculate returns only on predicted signals
    backtest_results_df['Strategy_Returns'] = backtest_results_df.apply(
        lambda row: row['Future_Price_Change_Pct'] if row['Predicted_Signal'] == 1 else 0, axis=1
    )
    total_strategy_return = backtest_results_df['Strategy_Returns'].sum()
    num_trades = backtest_results_df['Predicted_Signal'].sum()

    print(f"\nBacktest Strategy Simulation:")
    print(f"Number of predicted 'buy' signals (trades): {num_trades}")
    print(f"Total simulated return from trades: {total_strategy_return:.2f}%")
    if num_trades > 0:
        print(f"Average return per trade: {total_strategy_return / num_trades:.2f}%")
    else:
        print("No trades were made in backtest.")


    # --- Forward Testing ---
    print("\n--- Running Forward Test ---")
    if not X_forward_test.empty:
        y_pred_forward_test = ensemble_model.predict(X_forward_test)

        print("\nForward Test Results:")
        print(f"Accuracy: {accuracy_score(y_forward_test, y_pred_forward_test):.4f}")
        print(f"Precision: {precision_score(y_forward_test, y_pred_forward_test):.4f}")
        print(f"Recall: {recall_score(y_forward_test, y_pred_forward_test):.4f}")
        print(f"F1-Score: {f1_score(y_forward_test, y_pred_forward_test):.4f}")
        print("\nConfusion Matrix (Forward Test):")
        print(confusion_matrix(y_forward_test, y_pred_forward_test))

        # Simple Forward Test Strategy Simulation
        forward_test_results_df = df.iloc[backtest_data_end_idx:].copy()
        forward_test_results_df['Predicted_Signal'] = y_pred_forward_test
        forward_test_results_df['Actual_Target'] = y_forward_test

        forward_test_results_df['Strategy_Returns'] = forward_test_results_df.apply(
            lambda row: row['Future_Price_Change_Pct'] if row['Predicted_Signal'] == 1 else 0, axis=1
        )
        total_strategy_return_forward = forward_test_results_df['Strategy_Returns'].sum()
        num_trades_forward = forward_test_results_df['Predicted_Signal'].sum()

        print(f"\nForward Test Strategy Simulation:")
        print(f"Number of predicted 'buy' signals (trades): {num_trades_forward}")
        print(f"Total simulated return from trades: {total_strategy_return_forward:.2f}%")
        if num_trades_forward > 0:
            print(f"Average return per trade: {total_strategy_return_forward / num_trades_forward:.2f}%")
        else:
            print("No trades were made in forward test.")
    else:
        print("No data available for forward testing.")

    print("\n--- Trading Strategy Execution Complete ---")

# --- Example Usage ---
if __name__ == "__main__":
    # Define the path to the Nifty50 1-minute data
    nifty_data_path = r'D:\Sushant\Fyers_AlgoTrade\Fyers_Data\Nifty50_Index\NIFTY50_INDEX_15_Min.csv'

    # Load OHLC data from the specified CSV file
    try:
        ohlc_data = load_ohlc_data(nifty_data_path)
    except FileNotFoundError:
        print(f"Error: The file '{nifty_data_path}' was not found. Please check the path.")
        exit()
    except ValueError as e:
        print(f"Error loading data: {e}")
        exit()

    # Run the trading strategy
    run_trading_strategy(ohlc_data.copy(),
                         price_increase_threshold=0.25, # Target: 0.05% price increase
                         prediction_horizon_minutes=60,   # Predict 5 minutes ahead
                         train_test_split_ratio=0.7,     # 70% of backtest data for training
                         forward_test_ratio=0.15)        # 15% of total data for forward testing
