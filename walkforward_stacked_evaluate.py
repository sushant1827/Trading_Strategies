#!/usr/bin/env python3
"""
walkforward_stacked_evaluate.py

Run time-series safe stacked ensemble (RF + XGB -> Logistic meta) walk-forward evaluation.
Saves per-fold metrics and equity curves, model pickles and a packaged ZIP.

Usage:
 python walkforward_stacked_evaluate.py --data /path/to/ohlc.csv --outdir ./wf_stack_out
"""

import argparse, warnings, zipfile
from pathlib import Path
import numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# ---------------------------
# Supertrend + features + labels
# ---------------------------
def compute_atr(df, length=14):
    h,l,c = df['high'], df['low'], df['close']
    tr1 = (h-l).abs(); tr2 = (h-c.shift(1)).abs(); tr3 = (l-c.shift(1)).abs()
    tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def supertrend(df, period=10, multiplier=3.0):
    df = df.copy().reset_index(drop=True)
    df['atr'] = compute_atr(df, length=period)
    hl2 = (df['high'] + df['low'])/2.0
    df['upperband'] = hl2 + multiplier * df['atr']
    df['lowerband'] = hl2 - multiplier * df['atr']
    df['supertrend'] = np.nan; df['trend'] = 1
    prev_up, prev_dn, prev_trend = df['upperband'].iloc[0], df['lowerband'].iloc[0], 1
    for i in range(len(df)):
        if i==0:
            df.at[i,'supertrend'] = prev_dn if df['close'].iloc[i] > prev_dn else prev_up
            df.at[i,'trend'] = prev_trend
            continue
        up = df['upperband'].iloc[i]; dn = df['lowerband'].iloc[i]; close_prev = df['close'].iloc[i-1]
        up = max(up, prev_up) if close_prev > prev_up else up
        dn = min(dn, prev_dn) if close_prev < prev_dn else dn
        trend = prev_trend
        if prev_trend == -1 and df['close'].iloc[i] > prev_dn:
            trend = 1
        elif prev_trend == 1 and df['close'].iloc[i] < prev_up:
            trend = -1
        df.at[i,'supertrend'] = up if trend==1 else dn
        df.at[i,'trend'] = trend
        prev_up, prev_dn, prev_trend = up, dn, trend
    return df

def generate_signals_from_trend(df):
    df = df.copy().reset_index(drop=True)
    df['signal'] = 0
    df.loc[(df['trend']==1) & (df['trend'].shift(1)==-1), 'signal'] = 1
    df.loc[(df['trend']==-1) & (df['trend'].shift(1)==1), 'signal'] = -1
    return df

def build_features_labels(df, lookahead=10):
    st = supertrend(df, period=10, multiplier=3.0)
    st = generate_signals_from_trend(st).reset_index(drop=True)
    # safe features (compute then shift)
    st['ret1'] = st['close'].pct_change(1).shift(1).fillna(0)
    st['ret5'] = st['close'].pct_change(5).shift(1).fillna(0)
    st['vol20'] = st['ret1'].rolling(20, min_periods=1).std().shift(1).fillna(0)
    st['atr_feat'] = st['atr'].shift(1).fillna(method='bfill')
    st['mom1'] = st['ret1'].shift(1).fillna(0)
    st['mom5'] = st['ret5'].shift(1).fillna(0)
    st['cor'] = (st['close']/st['open'] - 1.0).shift(1).fillna(0)

    signals = st[st['signal']==1].copy().reset_index()
    features=[]; labels=[]; rets=[]; indices=[]
    for _, row in signals.iterrows():
        i = int(row['index'])
        if i + lookahead < len(st)-1:
            entry = st.loc[i+1,'open']; fut = st.loc[i+lookahead,'close']
            future_ret = fut - entry
            label = 1 if future_ret > 0 else 0
            feat = {'atr': st.at[i,'atr_feat'], 'mom1': st.at[i,'mom1'], 'mom5': st.at[i,'mom5'], 'vol20': st.at[i,'vol20'], 'cor': st.at[i,'cor']}
            features.append(feat); labels.append(label); rets.append(future_ret); indices.append(i)
    X = pd.DataFrame(features).fillna(0)
    y = np.array(labels); yreg = np.array(rets)
    return st, X, y, yreg, indices

# ---------------------------
# Time-safe local OOF inside a training window
# ---------------------------
def local_expanding_oof(clf_ctor, clf_kwargs, X_window, y_window, n_splits_local=4, min_train_local=40):
    """
    Build OOF probabilities _within_ a training window (expanding-window inside the training set).
    Returns array length = len(X_window) with OOF probs (zeros for early rows without OOF).
    """
    n = len(X_window)
    oof = np.zeros(n)
    fold_size = max(1, n // (n_splits_local + 1))
    for k in range(1, n_splits_local+1):
        tr_end = k * fold_size
        te_end = min((k+1) * fold_size, n)
        if te_end <= tr_end or tr_end < min_train_local:
            continue
        X_tr = X_window.iloc[:tr_end]; y_tr = y_window[:tr_end]
        X_te = X_window.iloc[tr_end:te_end]
        clf = clf_ctor(**clf_kwargs)
        clf.fit(X_tr, y_tr)
        oof[tr_end:te_end] = clf.predict_proba(X_te)[:,1]
    return oof

# ---------------------------
# Backtest (no tx costs)
# ---------------------------
def backtest_no_txcosts(st, size_series):
    st = st.copy().reset_index(drop=True)
    pos=0.0; entry=None; trades=[]; cum=0.0; equity=[]
    for i in range(len(st)-1):
        equity.append(cum)
        if pos==0:
            if st.loc[i,'signal']==1 and size_series[i]>0:
                entry = st.loc[i+1,'open']; pos = float(size_series[i])
        else:
            if st.loc[i,'signal']==-1:
                exitp = st.loc[i+1,'open']; profit = (exitp - entry) * pos
                trades.append({'entry':i,'exit':i+1,'profit':profit})
                cum += profit; pos=0.0; entry=None
    equity.append(cum)
    return pd.DataFrame(trades), pd.Series(equity, index=range(len(equity)))

def summary_metrics(trades_df):
    if trades_df.empty:
        return {'Total Trades':0, 'Net Profit':0.0, 'Profit Factor':np.nan, 'Win Rate (%)':np.nan}
    wins = trades_df[trades_df['profit']>0]; losses = trades_df[trades_df['profit']<=0]
    total = len(trades_df); net = trades_df['profit'].sum()
    pf = wins['profit'].sum() / (-losses['profit'].sum()) if losses['profit'].sum()!=0 else np.nan
    win_rate = len(wins)/total*100
    return {'Total Trades': total, 'Net Profit': round(net,2), 'Profit Factor': round(pf,2), 'Win Rate (%)': round(win_rate,2)}

# ---------------------------
# Walk-forward stacking evaluation
# ---------------------------
def walkforward_stacked(df, outdir, n_estimators=300, lookahead=10, n_splits=5, min_train=200, n_local_splits=4):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    plots = outdir / "plots"; plots.mkdir(exist_ok=True)
    models = outdir / "models"; models.mkdir(exist_ok=True)

    st, X, y, yreg, indices = build_features_labels(df, lookahead=lookahead)
    n = len(X)
    if n < 50:
        raise RuntimeError("Too few labeled samples for walk-forward.")

    fold_size = max(1, n // (n_splits + 1))
    wf_rows = []

    for k in range(1, n_splits+1):
        train_end = k * fold_size
        test_end = min((k+1) * fold_size, n)
        if test_end <= train_end:
            break
        print(f"WF fold {k}: train_end={train_end}, test_end={test_end}")

        # Training window and test window for labeled samples
        X_tr, y_tr = X.iloc[:train_end], y[:train_end]
        X_te, y_te = X.iloc[train_end:test_end], y[train_end:test_end]
        indices_te = indices[train_end:test_end]

        # Train base models on X_tr
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X_tr, y_tr)
        joblib.dump(rf, models / f"rf_fold_{k}.joblib")

        if HAVE_XGB:
            xgb = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=42)
            xgb.fit(X_tr, y_tr)
            joblib.dump(xgb, models / f"xgb_fold_{k}.joblib")
            base_list = [('RF', rf), ('XGB', xgb)]
        else:
            base_list = [('RF', rf)]

        # Build OOF probs inside X_tr for meta training (time-safe)
        oof_cols = []
        for name,mdl in base_list:
            oof = local_expanding_oof(lambda **kw: mdl.__class__(**kw), mdl.get_params(), X_tr, y_tr, n_splits_local=n_local_splits, min_train_local=min_train)
            # The helper above returns OOF on X_tr length; we'll use it as training meta feature
            oof_cols.append(oof)

        meta_X_tr = pd.DataFrame(np.column_stack(oof_cols), columns=[n for n,_ in base_list])
        # If OOF not available sufficiently inside training window, fallback to base full probs on X_tr
        if (meta_X_tr.sum(axis=1) != 0).sum() < max(10, int(0.1*len(X_tr))):
            print("Not enough local OOF inside train window; using full-model probs fallback for meta training.")
            base_full_train = np.column_stack([mdl.predict_proba(X_tr)[:,1] for _,mdl in base_list])
            meta_X_tr = pd.DataFrame(base_full_train, columns=[n for n,_ in base_list])

        meta_y_tr = y_tr
        meta_clf = LogisticRegression(max_iter=400)
        meta_clf.fit(meta_X_tr, meta_y_tr)
        joblib.dump(meta_clf, models / f"meta_fold_{k}.joblib")

        # Train sizing regressor on X_tr -> yreg_tr (simple GB)
        reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
        reg.fit(X_tr, yreg[:train_end])
        joblib.dump(reg, models / f"reg_fold_{k}.joblib")

        # Create stacked probabilities for all labeled rows using the fold base models
        base_full_probs_all = np.column_stack([mdl.predict_proba(X)[:,1] for _,mdl in base_list])
        stacked_probs_all = meta_clf.predict_proba(pd.DataFrame(base_full_probs_all, columns=[n for n,_ in base_list]))[:,1]
        # sizing predictions
        try:
            pred_reg_all = reg.predict(X)
        except:
            pred_reg_all = np.zeros(len(X))

        # normalize predicted return by train distribution
        minr, maxr = yreg[:train_end].min(), yreg[:train_end].max()
        rrange = maxr-minr if maxr>minr else 1.0
        norm_ret_all = np.clip((pred_reg_all - minr)/rrange, 0, 1)

        # Build size series for test fold using stacked combined sizing (size only for indices in test window)
        size_series = np.zeros(len(st))
        for idx_sig, sp, nr in zip(indices[train_end:test_end], stacked_probs_all[train_end:test_end], norm_ret_all[train_end:test_end]):
            # you can tune threshold here (we use 0.5)
            size_series[idx_sig] = float(sp * nr)

        # Backtest size_series on full series to collect equity and metrics for fold
        trades, eq = backtest_no_txcosts(st, size_series)
        summ = summary_metrics(trades)
        summ.update({'fold':k, 'train_end':train_end, 'test_end':test_end, 'test_trades': len(trades)})
        wf_rows.append(summ)

        # Save fold equity plot (we plot equity across the whole series but it's driven by test fold sizes)
        plt.figure(figsize=(10,6))
        plt.plot(eq.reset_index(drop=True), label=f'WF fold {k} equity (stacked combined)')
        plt.title(f"WF fold {k} equity (stacked combined)")
        plt.xlabel("Bar index"); plt.ylabel("Cumulative points")
        plt.grid(True); plt.legend()
        pfile = plots / f"wf_fold_{k}_stacked_comb.png"
        plt.savefig(pfile, bbox_inches='tight'); plt.close()

    # Save walk-forward results
    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(outdir / "walkforward_stacked_folds.csv", index=False)

    # Zip outputs
    pkg = outdir / "walkforward_stacked_package.zip"
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in (plots.glob("*.png")): zf.write(p, arcname=f"plots/{p.name}")
        for p in (models.glob("*.joblib")): zf.write(p, arcname=f"models/{p.name}")
        zf.write(outdir / "walkforward_stacked_folds.csv", arcname="walkforward_stacked_folds.csv")
    print("Walk-forward evaluation complete. Outputs in:", outdir, "package:", pkg)
    return wf_df

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with OHLC columns")
    parser.add_argument("--outdir", default="wf_stack_out", help="output directory")
    parser.add_argument("--n_estimators", type=int, default=300, help="trees for RF/XGB")
    parser.add_argument("--lookahead", type=int, default=10, help="lookahead bars for label")
    parser.add_argument("--n_splits", type=int, default=5, help="walk-forward folds")
    parser.add_argument("--min_train", type=int, default=200, help="min train size for local OOF")
    parser.add_argument("--n_local_splits", type=int, default=4, help="local OOF splits inside train")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError("Data not found: "+str(data_path))
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {'open','high','low','close'}
    if not required.issubset(set(df.columns)):
        raise ValueError("CSV must contain columns: open, high, low, close")

    if not HAVE_XGB:
        print("Warning: XGBoost not available; running RF-only stacked WF.")

    walkforward_stacked(df, args.outdir, n_estimators=args.n_estimators, lookahead=args.lookahead, n_splits=args.n_splits, min_train=args.min_train, n_local_splits=args.n_local_splits)

if __name__ == "__main__":
    main()
