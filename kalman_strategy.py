# kalman_strategy.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILEPATH = "D:/Sushant/Fyers_AlgoTrade/Fyers_Data/Nifty50_Index/NIFTY50_INDEX_60_Min.csv"

def compute_atr(df,length=14):
    h,l,c = df['high'], df['low'], df['close']
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def compute_kalman(df, atr_len=14, atr_ma_len=50, Q0_scale=1e-4, vel_scale=1e-6):
    df = df.copy()
    if len(df) == 0:
        df['kf_mu'] = pd.Series(dtype=float)
        df['kf_vel'] = pd.Series(dtype=float)
        df['indicator'] = pd.Series(dtype=float)
        df['indicator_slope'] = pd.Series(dtype=float)
        df['atr'] = pd.Series(dtype=float)
        df['atr_ma'] = pd.Series(dtype=float)
        return df
    price = df['close'].values; n=len(price)
    df['atr'] = compute_atr(df, atr_len)
    df['atr_ma'] = df['atr'].rolling(atr_ma_len, min_periods=1).mean().bfill()
    V = (df['atr'] / df['atr_ma']).fillna(1.0).ewm(span=5).mean().clip(0.5,3.0).values
    x = np.zeros(2); x[0]=price[0]; x[1]=0.0
    P = np.eye(2) * 0.1
    F = np.array([[1,1],[0,1]])
    H = np.array([[1.0,0.0]])
    mu = np.zeros(n); vel = np.zeros(n)
    R0 = max(1e-6, np.nanstd(np.diff(price))**2)
    for t in range(n):
        Q = np.diag([Q0_scale, vel_scale]) * V[t]
        R = R0 / V[t]
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        z = price[t]
        y = z - (H @ x_pred)[0]
        S = (H @ P_pred @ H.T)[0,0] + R
        K = (P_pred @ H.T).flatten() / S
        x = x_pred + K * y
        P = (np.eye(2) - np.outer(K, H[0])) @ P_pred
        mu[t] = x[0]; vel[t] = x[1]
    df['kf_mu'] = mu; df['kf_vel'] = vel; df['indicator'] = df['kf_mu']; df['indicator_slope'] = df['indicator'].diff()
    return df

def generate_signals(df):
    df = df.copy()
    # Use Kalman level as trend; buy when price crosses above level and velocity positive
    df['above'] = df['close'] > df['indicator']
    df['above_prev'] = df['above'].shift(1).fillna(False)
    df['cross_up'] = (~df['above_prev']) & df['above']
    df['long_signal'] = df['cross_up'] & (df['kf_vel'] > 0)
    df['exit_signal'] = (df['close'] < df['indicator']) | (df['kf_vel'] < 0)
    return df

def backtest(df, start_equity=100000, risk_pct=0.01, m_init=2.0, warmup=60):
    df = df.copy().reset_index(); trades=[]; cash=start_equity; pos=0; ep=ei=stop=None; eq=[]
    for i in range(warmup, len(df)-1):
        eq.append({'dt':df.at[i,df.columns[0]], 'equity':cash + pos*df.at[i,'close']})
        if pos>0:
            if df.at[i,'low'] <= stop:
                exit_price=stop; pnl=(exit_price-ep)*pos; cash+=exit_price*pos
                trades.append({'entry_idx':ei,'exit_idx':i,'entry_price':ep,'exit_price':exit_price,'size':pos,'pnl':pnl})
                pos=0; ep=ei=stop=None; continue
            if i-1>=0 and df.at[i-1,'exit_signal']:
                exit_price = df.at[i,'open']; pnl=(exit_price-ep)*pos; cash+=exit_price*pos
                trades.append({'entry_idx':ei,'exit_idx':i,'entry_price':ep,'exit_price':exit_price,'size':pos,'pnl':pnl})
                pos=0; ep=ei=stop=None; continue
        if pos==0 and df.at[i,'long_signal']:
            exec_idx=i+1
            if exec_idx>=len(df): break
            ep = df.at[exec_idx,'open']; ei = exec_idx
            atr = df.at[i,'atr']; stop = ep - m_init * atr
            risk_unit = ep - stop
            size = np.floor((cash * risk_pct)/risk_unit) if risk_unit>0 else 0
            if size<=0: continue
            pos = size; cash -= pos * ep; continue
    eq.append({'dt':df.at[len(df)-1, df.columns[0]], 'equity': cash + pos * df.at[len(df)-1,'close']})
    return pd.DataFrame(trades), pd.DataFrame(eq).set_index('dt')

if __name__=='__main__':
    if FILEPATH:
        try:
            df = pd.read_csv(FILEPATH, parse_dates=True, index_col=0)
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns=lambda x: x.strip().lower())
            # sanitize typical variations
            if 'date' in df.columns and 'datetime' in df.columns:
                # if both exist, assume index is fine
                pass
            # drop nuisance columns
            df = df.drop(columns=['unnamed: 0', 'volume'], errors='ignore')
            # ensure index is datetime
            if df.index.dtype == 'int64':
                df.index = pd.to_datetime(df.index, errors='coerce')
            else:
                df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[~df.index.duplicated(keep='first')]
            # optionally filter years - keep as you had originally if needed
            try:
                df = df[df.index.year >= 2021]
            except Exception:
                pass
        except Exception as e:
            print("Failed to load CSV; falling back to synthetic:", e)
            FILEPATH = None

    if 'df' in locals() and df.empty:
        print("Loaded DataFrame is empty; falling back to synthetic data")
        FILEPATH = None

    if not FILEPATH:
        rng=np.random.default_rng(3); n=1000; dt=pd.date_range(end=pd.Timestamp.today(), periods=n, freq='D')
        steps=rng.normal(0.0003,0.01,n); close=100+np.cumsum(steps); open_=np.roll(close,1); open_[0]=close[0]
        high=np.maximum(open_,close)+abs(rng.normal(0,0.25,n)); low=np.minimum(open_,close)-abs(rng.normal(0,0.25,n))
        df=pd.DataFrame({'open':open_,'high':high,'low':low,'close':close}, index=dt)
    df = compute_kalman(df)
    df = generate_signals(df)
    trades, eq = backtest(df)
    print("Trades:", len(trades)); print(trades.head() if not trades.empty else "No trades"); print("Final equity:", eq['equity'].iloc[-1])
    plt.figure(figsize=(12,5)); plt.plot(df.index, df['close']); plt.plot(df.index, df['indicator']); plt.show()
