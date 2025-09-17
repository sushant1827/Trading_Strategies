# """
# rl_trading_agent_run.py

# Use your local file:
#  D:\Sushant\Fyers_AlgoTrade\Fyers_Data\Nifty50_Index\NIFTY50_INDEX_30_Min.csv

# This script:
#  - Loads OHLCV data
#  - Computes indicators and scales features
#  - Builds TradingEnv supporting 'discrete' and 'continuous' actions
#  - Trains algorithms (PPO, A2C, DDPG, SAC, TD3) with permissioned hyperparams
#  - Evaluates on test set and prints metrics and equity curve
# """

import os
import random
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import BaseCallback
except Exception as e:
    raise ImportError("Install stable-baselines3 with extras: pip install stable-baselines3[extra]") from e

import torch

# -------------------------
# Reproducibility / Seeds
# -------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# -------------------------
# User config - set your CSV path here
# -------------------------
DATA_PATH = "Nifty50_Index/NIFTY50_INDEX_30_Min.csv"
INITIAL_BALANCE = 100000.0
TRANSACTION_COST = 0.0005
REWARD_SCALING = 1e4
TOTAL_TIMESTEPS = 50_000  # sufficient for training
TRAIN_TEST_SPLIT = 0.8

# Choose action mode: 'discrete' (Buy/Hold/Sell) or 'continuous' (fractional exposure)
ACTION_MODE = 'discrete'  # set to 'continuous' to use SAC/TD3/DDPG naturally

# -------------------------
# Feature engineering
# -------------------------
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['ret'] = df['close'].pct_change().fillna(0)
    df['log_ret'] = np.log1p(df['ret'])

    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / (bb_mid + 1e-9)

    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    df['volatility'] = df['log_ret'].rolling(20).std()
    df['mom_5'] = df['close'] - df['close'].shift(5)
    df['mom_10'] = df['close'] - df['close'].shift(10)

    df = df.bfill().fillna(0)
    return df

def prepare_features(df: pd.DataFrame, scaler: StandardScaler = None):
    df = add_technical_indicators(df)
    candidate_cols = [
        'log_ret', 'sma_5', 'sma_20', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_hist',
        'rsi', 'bb_upper', 'bb_lower', 'bb_bandwidth', 'atr', 'volatility', 'mom_5', 'mom_10'
    ]
    feature_columns = [c for c in candidate_cols if c in df.columns]
    X = df[feature_columns].values.astype(np.float32)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X)
    X_scaled = scaler.transform(X)

    for i, col in enumerate(feature_columns):
        df[f"f_{col}"] = X_scaled[:, i]

    feature_columns_scaled = [f"f_{c}" for c in feature_columns]
    return df, scaler, feature_columns_scaled

# -------------------------
# Trading Environment (supports discrete & continuous)
# -------------------------
class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df: pd.DataFrame,
                 feature_columns: List[str],
                 action_type: str = 'discrete',
                 initial_balance: float = 100000.0,
                 max_position: float = 1.0,
                 transaction_cost: float = 0.0005,
                 reward_scaling: float = 1.0,
                 seed: int = 42):
        super().__init__()
        assert action_type in ('discrete', 'continuous')
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.action_type = action_type
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling

        self.start_index = 0
        self.current_step = 0
        self.end_index = len(self.df) - 1

        obs_dim = len(feature_columns) + 2  # features + last price + current position

        if action_type == 'discrete':
            self.action_space = spaces.Discrete(3)  # sell, hold, buy
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.position_price = None
        self.seed(seed)

        self.trades = []
        self.history = {'portfolio_value': [], 'position': [], 'close': [], 'action': [], 'reward': []}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        start_index = options.get('start_index') if options else None
        if start_index is None:
            self.start_index = 0
        else:
            self.start_index = start_index

        self.current_step = self.start_index
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_price = None
        self.portfolio_value = self.initial_balance

        self.trades = []
        self.history = {'portfolio_value': [], 'position': [], 'close': [], 'action': [], 'reward': []}
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        features = row[self.feature_columns].values.astype(np.float32)
        last_price = np.array([row['close']], dtype=np.float32)
        pos = np.array([self.position], dtype=np.float32)
        obs = np.concatenate([features, last_price, pos])
        return obs

    def _map_discrete_to_pos(self, action_int: int) -> float:
        mapping = {0: -1.0, 1: 0.0, 2: 1.0}
        return mapping.get(int(action_int), 0.0)

    def step(self, action):
        if self.action_type == 'discrete':
            action_val = int(np.asarray(action).item())
            desired_pos = self._map_discrete_to_pos(action_val)
            action_for_logging = action_val
        else:
            desired_pos = float(np.clip(action, -1.0, 1.0))
            if isinstance(desired_pos, np.ndarray):
                desired_pos = float(desired_pos[0])
            action_for_logging = float(desired_pos)

        prev_price = self.df.iloc[self.current_step]['close']
        self.current_step += 1
        done = self.current_step >= self.end_index
        current_price = self.df.iloc[self.current_step]['close']

        position_change = desired_pos - self.position
        traded_notional = abs(position_change) * self.portfolio_value
        cost = traded_notional * self.transaction_cost

        price_return = (current_price - prev_price) / (prev_price + 1e-9)
        pnl_from_hold = self.position * price_return * self.portfolio_value

        self.portfolio_value += pnl_from_hold - cost
        self.position = float(desired_pos)

        if abs(position_change) > 1e-6:
            self.trades.append({'step': self.current_step, 'price': current_price, 'position': self.position, 'change': position_change, 'cost': cost})

        reward = (pnl_from_hold - cost) / max(self.initial_balance, 1.0)
        if abs(self.position) > 0.8:
            reward -= 0.0001 * (abs(self.position) - 0.8)
        reward = reward * self.reward_scaling

        self.history['portfolio_value'].append(self.portfolio_value)
        self.history['position'].append(self.position)
        self.history['close'].append(current_price)
        self.history['action'].append(action_for_logging)
        self.history['reward'].append(reward)

        obs = self._get_observation()
        info = {'portfolio_value': self.portfolio_value, 'position': self.position, 'step': self.current_step}
        return obs, float(reward), bool(done), False, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Close: {self.df.iloc[self.current_step]['close']:.2f}, "
              f"Position: {self.position:.3f}, Portfolio: {self.portfolio_value:.2f}")

# -------------------------
# Metrics
# -------------------------
def cumulative_return(series: pd.Series) -> float:
    return series.iloc[-1] / series.iloc[0] - 1.0

def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, period: int = 252) -> float:
    mean = np.mean(returns - risk_free)
    std = np.std(returns)
    if std == 0:
        return 0.0
    return np.sqrt(period) * mean / std

def max_drawdown(equity: np.ndarray) -> float:
    roll_max = np.maximum.accumulate(equity)
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())

def win_rate_from_trades(history: Dict[str, Any]) -> float:
    pv = np.array(history['portfolio_value'])
    if len(pv) < 2:
        return 0.0
    returns = np.diff(pv) / (pv[:-1] + 1e-9)
    wins = (returns > 0).sum()
    return float(wins) / len(returns)

# -------------------------
# Training utilities
# -------------------------
class SimpleLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        return True

def train_agents(env_fn, action_mode: str, algo_hyperparams: Dict[str, Dict], total_timesteps: int = 50_000, seed: int = SEED):
    trained_models = {}
    for name, cfg in algo_hyperparams.items():
        # compatibility checks
        if action_mode == 'discrete' and cfg.get('action_type', 'discrete') == 'continuous':
            print(f"Skipping {name} (continuous-only) for discrete env.")
            continue

        print(f"\n--- Training {name} ---")
        alg_class = cfg['class']
        params = cfg.get('params', {}).copy()

        venv = DummyVecEnv([env_fn])
        venv.seed(SEED)
        model = alg_class('MlpPolicy', venv, seed=SEED, verbose=0, **params)
        model.learn(total_timesteps=total_timesteps, callback=SimpleLoggerCallback())
        trained_models[name] = model
        print(f"{name} training finished.")
    return trained_models

def evaluate_model(model, env, deterministic=True):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.history

# -------------------------
# Plotting
# -------------------------
def plot_equity_curves(histories: Dict[str, Dict], title: str = "Equity Curves"):
    plt.figure(figsize=(10,6))
    for name, h in histories.items():
        pv = np.array(h['portfolio_value'])
        plt.plot(pv, label=name)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.savefig('equity_curves.png')
    print("Equity curves saved to equity_curves.png")

# -------------------------
# Main workflow
# -------------------------
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV not found at {DATA_PATH}. Update DATA_PATH.")

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df = df.rename(columns={'Date': 'datetime'})
    df.columns = df.columns.str.lower()

    required_cols = set(['datetime', 'open', 'high', 'low', 'close', 'volume'])
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.drop_duplicates(subset='datetime')
    df = df[df['datetime'].dt.year >= 2021]
    df = df.sort_values('datetime').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = df[col].round(2)

    df_with_feats, scaler, feature_columns = prepare_features(df)

    split_idx = int(TRAIN_TEST_SPLIT * len(df_with_feats))
    train_df = df_with_feats.iloc[:split_idx].reset_index(drop=True)
    test_df = df_with_feats.iloc[split_idx:].reset_index(drop=True)
    split_date = df_with_feats['datetime'].iloc[split_idx]
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}; split_date={split_date.date()}")

    n_actions = 1
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Permissioned hyperparameters - tweakable
    algo_hyperparams = {
        'PPO': {'class': PPO, 'params': {'batch_size':64, 'n_epochs':10, 'learning_rate':3e-4}, 'action_type': 'both'},
        'A2C': {'class': A2C, 'params': {'n_steps':5, 'learning_rate':7e-4}, 'action_type': 'both'},
        'DDPG': {'class': DDPG, 'params': {'action_noise': action_noise, 'buffer_size':50_000, 'learning_rate':1e-3}, 'action_type': 'continuous'},
        'SAC': {'class': SAC, 'params': {'buffer_size':100_000, 'learning_rate':3e-4}, 'action_type': 'continuous'},
        'TD3': {'class': TD3, 'params': {'action_noise': action_noise, 'buffer_size':100_000, 'learning_rate':1e-3}, 'action_type': 'continuous'}
    }

    def make_env_train():
        return TradingEnv(train_df, feature_columns, action_type=ACTION_MODE,
                          initial_balance=INITIAL_BALANCE, transaction_cost=TRANSACTION_COST,
                          reward_scaling=REWARD_SCALING, seed=SEED)

    def make_env_test():
        return TradingEnv(test_df, feature_columns, action_type=ACTION_MODE,
                          initial_balance=INITIAL_BALANCE, transaction_cost=TRANSACTION_COST,
                          reward_scaling=REWARD_SCALING, seed=SEED)

    trained_models = train_agents(make_env_train, ACTION_MODE, algo_hyperparams, total_timesteps=TOTAL_TIMESTEPS, seed=SEED)

    histories = {}
    metrics = {}
    for name, model in trained_models.items():
        print(f"\nEvaluating {name} on test set...")
        env_test = make_env_test()
        history = evaluate_model(model, env_test, deterministic=True)
        histories[name] = history

        pv = np.array(history['portfolio_value'])
        returns = np.diff(pv) / (pv[:-1] + 1e-9)
        cumret = cumulative_return(pd.Series(pv))
        sr = sharpe_ratio(returns, period=252)
        mdd = max_drawdown(pv)
        wr = win_rate_from_trades(history)
        metrics[name] = {
            'cumulative_return': cumret,
            'sharpe': sr,
            'max_drawdown': mdd,
            'win_rate': wr,
            'final_portfolio': float(pv[-1])
        }

    print("\n=== Backtest Metrics (Test Set) ===")
    for name, m in metrics.items():
        print(f"\n{name}:")
        print(f"  Final Portfolio: {m['final_portfolio']:.2f}")
        print(f"  Cumulative Return: {m['cumulative_return'] * 100:.2f}%")
        print(f"  Sharpe Ratio (annualized): {m['sharpe']:.2f}")
        print(f"  Max Drawdown: {m['max_drawdown'] * 100:.2f}%")
        print(f"  Win Rate (periods profitable): {m['win_rate'] * 100:.2f}%")

    if len(histories) > 0:
        plot_equity_curves(histories, title=f"Equity Curves on Test Set (action_mode={ACTION_MODE})")

    os.makedirs("trained_models", exist_ok=True)
    for name, model in trained_models.items():
        model.save(os.path.join("trained_models", f"{name}_model"))
    print("Saved trained models to trained_models/")

if __name__ == "__main__":
    main()
