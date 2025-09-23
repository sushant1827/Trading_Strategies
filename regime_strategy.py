# regime_strategy.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILEPATH = "D:/Sushant/Fyers_AlgoTrade/Fyers_Data/Nifty50_Index/NIFTY50_INDEX_60_Min.csv"

def compute_atr(df, length=14):
    high, low, close = df['high'], df['low'], df['close']
    tr1 = (high-low).abs(); tr2 = (high-close.shift(1)).abs(); tr3=(low-close.shift(1)).abs()
    tr = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_vema(df, N_base=50, N_min=8, atr_len=14, atr_ma_len=50, k=1.5):
    df = df.copy()
    df['atr'] = compute_atr(df, atr_len)
    df['atr_ma'] = df['atr'].rolling(atr_ma_len, min_periods=1).mean().bfill()
    v = (df['atr'] / df['atr_ma']).fillna(1.0)
    w_v = ((v - 1.0) * k).clip(0,1)
    N_eff = (N_base * (1 - w_v) + N_min * w_v).clip(N_min, N_base)
    alpha = 2.0 / (N_eff + 1.0)
    prices = df['close'].values
    vema = np.full(len(prices), np.nan); vema[0]=prices[0]
    for t in range(1,len(prices)):
        a = alpha.iloc[t]; vema[t] = a * prices[t] + (1-a) * vema[t-1]
    df['vema'] = vema; df['vema_slope']=df['vema'].diff(); df['vema_period']=N_eff
    return df

def compute_regime_indicator(df, atr_short=14, atr_long=50, ema_short=12, ema_long=26, V_low=0.9, V_high=1.2, mom_mult=0.1):
    df = df.copy()
    df['atr_s'] = compute_atr(df, atr_short); df['atr_l'] = compute_atr(df, atr_long)
    df['V'] = (df['atr_s'] / df['atr_l']).fillna(1.0)
    df['ema_s'] = ema(df['close'], ema_short); df['ema_l'] = ema(df['close'], ema_long)
    df['momentum'] = df['ema_s'] - df['ema_l']
    df['mom_norm'] = df['momentum'] / (df['atr_s'] + 1e-12)
    def decide(v,m):
        if v < V_low and abs(m) < mom_mult: return 'LowVol_Sideways'
        if v < V_low and abs(m) >= mom_mult: return 'LowVol_Trend'
        if v >= V_high and abs(m) >= mom_mult: return 'HighVol_Trend'
        return 'HighVol_Sideways'
    df['regime'] = [decide(v,m) for v,m in zip(df['V'], df['mom_norm'])]
    df = compute_vema(df)
    df['sideways_lvl'] = df['close'].rolling(20, min_periods=1).median()
    df['indicator'] = df['vema'].where(df['regime'].str.contains('Trend'), df['sideways_lvl'])
    df['indicator_slope'] = df['indicator'].diff()
    return df

def generate_signals(df):
    df = df.copy()
    df['above'] = df['close'] > df['indicator']
    df['above_prev'] = df['above'].shift(1).fillna(False)
    df['cross_up'] = (~df['above_prev']) & (df['above']); df['cross_down'] = (df['above_prev']) & (~df['above'])
    # in trend regimes prefer trend rules; in sideways use RSI mean reversion (simple median)
    df['long_signal'] = (df['regime'].str.contains('Trend') & df['cross_up'] & (df['indicator_slope']>0)) | \
                        (df['regime'].str.contains('Sideways') & (df['close'] < df['indicator'] * 0.995))
    df['exit_signal'] = (df['regime'].str.contains('Trend') & df['cross_down']) | (df['regime'].str.contains('Sideways') & (df['close'] > df['indicator']))
    return df

# backtester same as prior patterns (next-bar open execution)
def backtest(df, start_equity=100000, risk_pct=0.01, m_init=2.0, warmup=80):
    df = df.copy().reset_index(); trades=[]; cash=start_equity; pos=0; entry_price=entry_idx=stop=None; eq=[]
    for i in range(warmup, len(df)-1):
        eq.append({'dt':df.at[i,df.columns[0]], 'equity': cash + pos * df.at[i,'close']})
        if pos>0:
            if df.at[i,'low'] <= stop:
                exit_price = stop; pnl = (exit_price-entry_price)*pos; cash += exit_price*pos
                trades.append({'entry_idx':entry_idx,'exit_idx':i,'entry_price':entry_price,'exit_price':exit_price,'size':pos,'pnl':pnl})
                pos=0; entry_price=entry_idx=stop=None; continue
            if i-1>=0 and df.at[i-1,'exit_signal']:
                exit_price = df.at[i,'open']; pnl=(exit_price-entry_price)*pos; cash+=exit_price*pos
                trades.append({'entry_idx':entry_idx,'exit_idx':i,'entry_price':entry_price,'exit_price':exit_price,'size':pos,'pnl':pnl})
                pos=0; entry_price=entry_idx=stop=None; continue
        if pos==0 and df.at[i,'long_signal']:
            exec_idx = i+1
            if exec_idx>=len(df): break
            entry_price = df.at[exec_idx,'open']; entry_idx = exec_idx
            atr = df.at[i,'atr_s'] if 'atr_s' in df.columns else compute_atr(df.set_index(df.columns[0])).iloc[i]
            stop = entry_price - m_init * atr
            risk_unit = entry_price - stop
            size = np.floor((cash * risk_pct)/risk_unit) if risk_unit>0 else 0
            if size<=0: continue
            pos = size; cash -= pos*entry_price; continue
    eq.append({'dt':df.at[len(df)-1, df.columns[0]], 'equity': cash + pos * df.at[len(df)-1,'close']})
    trades_df = pd.DataFrame(trades); eq_df = pd.DataFrame(eq).set_index('dt')
    return trades_df, eq_df

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
            if df.empty:
                print("Loaded DataFrame is empty; falling back to synthetic data")
                FILEPATH = None
        except Exception as e:
            print("Failed to load CSV; falling back to synthetic:", e)
            FILEPATH = None

    if not FILEPATH:
        rng=np.random.default_rng(1); n=1000; dt=pd.date_range(end=pd.Timestamp.today(), periods=n, freq='D')
        steps=rng.normal(0.0003,0.01,n); close=100+np.cumsum(steps); open_=np.roll(close,1); open_[0]=close[0]
        high=np.maximum(open_,close)+abs(rng.normal(0,0.25,n)); low=np.minimum(open_,close)-abs(rng.normal(0,0.25,n))
        df=pd.DataFrame({'open':open_,'high':high,'low':low,'close':close}, index=dt)

    df = compute_regime_indicator(df)
    df = generate_signals(df)
    trades, eq = backtest(df)
    print("Trades:", len(trades))
    print(trades.head())
    print("Final equity:", eq['equity'].iloc[-1])
    plt.figure(figsize=(12,5)); plt.plot(df.index, df['close'], label='close'); plt.plot(df.index, df['indicator'], label='indicator'); plt.legend(); plt.show()
