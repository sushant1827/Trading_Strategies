# adaptive_rsi_strategy.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILEPATH = "D:/Sushant/Fyers_AlgoTrade/Fyers_Data/Nifty50_Index/NIFTY50_INDEX_60_Min.csv"

def compute_atr(df,length=14):
    h,l,c = df['high'], df['low'], df['close']
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def rsi(series,n=14):
    delta=series.diff(); up=delta.clip(lower=0); down=-delta.clip(upper=0)
    ma_up=up.ewm(alpha=1/n, adjust=False).mean(); ma_down=down.ewm(alpha=1/n, adjust=False).mean()
    rs=ma_up/(ma_down+1e-12); return 100 - 100/(1+rs)

def compute_adaptive_rsi(df, n_base=14, n_min=7, n_max=40, atr_len=14, atr_ma_len=50, s=0.5, a=0.5):
    df = df.copy()
    df['atr'] = compute_atr(df, atr_len)
    df['atr_ma'] = df['atr'].rolling(atr_ma_len, min_periods=1).mean().fillna(method='bfill')
    V = (df['atr'] / df['atr_ma']).fillna(1.0)
    n_t = (n_base * (1 + s*(V-1.0))).round().clip(n_min, n_max).astype(int)
    for n in range(n_min, n_max + 1):
        df[f'rsi_{n}'] = rsi(df['close'], n)
    df['indicator'] = [ df.at[idx, f"rsi_{n}"] for idx,n in zip(df.index, n_t) ]
    df['indicator_slope'] = pd.Series(df['indicator']).diff()
    df['rsi_lower'] = 30 - a * (V - 1.0) * 10
    df['rsi_upper'] = 70 + a * (V - 1.0) * 10
    df['indicator_period'] = n_t
    return df

def generate_signals(df):
    df = df.copy()
    # use RSI thresholds if present
    if 'rsi_lower' in df.columns:
        df['long_signal'] = df['indicator'] < df['rsi_lower']
        df['exit_signal'] = df['indicator'] > 50
    else:
        df['long_signal'] = (df['indicator'].diff() > 0) & (df['indicator'] < 30)
        df['exit_signal'] = df['indicator'] > 50
    return df

def backtest(df, start_equity=100000, risk_pct=0.01, m_init=2.0, warmup=60):
    df = df.copy().reset_index(); trades=[]; cash=start_equity; pos=0; ep=ei=stop=None; eq=[]
    for i in range(warmup, len(df)-1):
        eq.append({'dt':df.at[i, df.columns[0]], 'equity': cash + pos * df.at[i,'close']})
        if pos>0:
            if df.at[i,'low'] <= stop:
                exit_price = stop; pnl=(exit_price-ep)*pos; cash+=exit_price*pos
                trades.append({'entry_idx':ei,'exit_idx':i,'entry_price':ep,'exit_price':exit_price,'size':pos,'pnl':pnl})
                pos=0; ep=ei=stop=None; continue
            if i-1>=0 and df.at[i-1,'exit_signal']:
                exit_price = df.at[i,'open']; pnl=(exit_price-ep)*pos; cash += exit_price*pos
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
                df.index = pd.to_datetime(df.index)
            else:
                df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep='first')]
            # optionally filter years - keep as you had originally if needed
            try:
                df = df[df.index.year >= 2021]
            except Exception:
                pass
        except Exception as e:
            print("Failed to load CSV; falling back to synthetic:", e)
            FILEPATH = None
    else:
        rng=np.random.default_rng(4); n=1000; dt=pd.date_range(end=pd.Timestamp.today(), periods=n, freq='D')
        steps=rng.normal(0.0003,0.01,n); close=100+np.cumsum(steps); open_=np.roll(close,1); open_[0]=close[0]
        high=np.maximum(open_,close)+abs(rng.normal(0,0.25,n)); low=np.minimum(open_,close)-abs(rng.normal(0,0.25,n))
        df=pd.DataFrame({'open':open_,'high':high,'low':low,'close':close}, index=dt)
    df = compute_adaptive_rsi(df)
    df = generate_signals(df)
    trades, eq = backtest(df)
    print("Trades:", len(trades)); print(trades.head() if not trades.empty else "No trades"); print("Final equity:", eq['equity'].iloc[-1])
    plt.figure(figsize=(12,5)); plt.plot(df.index, df['close']); plt.plot(df.index, df['indicator']); plt.show()
