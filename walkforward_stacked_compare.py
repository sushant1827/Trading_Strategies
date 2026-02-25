#!/usr/bin/env python3
"""
walkforward_stacked_compare.py

Walk-forward evaluation that compares BINARY vs COMBINED stacked sizing per fold.

Outputs:
 - outdir/walkforward_stacked_compare.csv  (per-fold metrics for both methods)
 - outdir/plots/wf_fold_{k}_binary.png
 - outdir/plots/wf_fold_{k}_combined.png
 - outdir/models/... (per-fold models)
 - outdir/walkforward_stacked_compare_package.zip
"""
import argparse, warnings, zipfile
from pathlib import Path
import numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")

# Optional XGBoost support
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# ---------------------------
# Supertrend, features, labels
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
    # safe lagged features
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
# local expanding OOF (inside training window)
# ---------------------------
def local_expanding_oof(clf_ctor, clf_kwargs, X_window, y_window, n_splits_local=4, min_train_local=40):
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
# backtest (no tx costs)
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
# Walk-forward compare
# ---------------------------
def walkforward_compare(df, outdir, n_estimators=300, lookahead=10, n_splits=5, min_train=200, n_local_splits=4, threshold=0.5):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    plots = outdir / "plots"; plots.mkdir(exist_ok=True)
    models = outdir / "models"; models.mkdir(exist_ok=True)

    st, X, y, yreg, indices = build_features_labels(df, lookahead=lookahead)
    n = len(X)
    if n < 20:
        raise RuntimeError("Too few labeled samples for walk-forward.")

    fold_size = max(1, n // (n_splits + 1))
    rows = []

    for k in range(1, n_splits+1):
        train_end = k * fold_size
        test_end = min((k+1) * fold_size, n)
        if test_end <= train_end:
            break
        print(f"Fold {k}: train_end={train_end}, test_end={test_end}")

        X_tr, y_tr = X.iloc[:train_end], y[:train_end]
        X_te, y_te = X.iloc[train_end:test_end], y[train_end:test_end]

        # train base models on X_tr
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X_tr, y_tr)
        joblib.dump(rf, models / f"rf_fold_{k}.joblib")

        base_list = [('RF', rf)]
        if HAVE_XGB:
            xgb = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=42)
            xgb.fit(X_tr, y_tr)
            joblib.dump(xgb, models / f"xgb_fold_{k}.joblib")
            base_list.append(('XGB', xgb))

        # build local OOF inside train window for meta training (time-safe)
        oof_cols = []
        for name, mdl in base_list:
            # use class & params from the trained model for constructor convenience
            oof = local_expanding_oof(lambda **kw: mdl.__class__(**kw), mdl.get_params(), X_tr, y_tr, n_splits_local=n_local_splits, min_train_local=min_train)
            oof_cols.append(oof)
        meta_X_tr = pd.DataFrame(np.column_stack(oof_cols), columns=[n for n,_ in base_list])

        # fallback if not enough local OOF
        if (meta_X_tr.sum(axis=1) != 0).sum() < max(10, int(0.1*len(X_tr))):
            print("Insufficient local OOF; using full-probs fallback for meta training.")
            base_full_train = np.column_stack([mdl.predict_proba(X_tr)[:,1] for _,mdl in base_list])
            meta_X_tr = pd.DataFrame(base_full_train, columns=[n for n,_ in base_list])
        meta_y_tr = y_tr
        meta_clf = LogisticRegression(max_iter=400)
        meta_clf.fit(meta_X_tr, meta_y_tr)
        joblib.dump(meta_clf, models / f"meta_fold_{k}.joblib")

        # sizing regressor on X_tr
        reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
        reg.fit(X_tr, yreg[:train_end])
        joblib.dump(reg, models / f"reg_fold_{k}.joblib")

        # create base full probs for all labeled rows using fold base models
        base_full_probs_all = np.column_stack([mdl.predict_proba(X)[:,1] for _,mdl in base_list])
        stacked_probs_all = meta_clf.predict_proba(pd.DataFrame(base_full_probs_all, columns=[n for n,_ in base_list]))[:,1]
        predreg_all = reg.predict(X)
        minr, maxr = yreg[:train_end].min(), yreg[:train_end].max()
        rrange = maxr - minr if maxr > minr else 1.0
        norm_ret_all = np.clip((predreg_all - minr)/rrange, 0, 1)

        # Build size series for test window indices only (so fold evaluates only test-period signals)
        size_binary = np.zeros(len(st))
        size_combined = np.zeros(len(st))
        for idx_sig, sp, nr in zip(indices[train_end:test_end], stacked_probs_all[train_end:test_end], norm_ret_all[train_end:test_end]):
            size_binary[idx_sig] = 1.0 if sp > threshold else 0.0
            size_combined[idx_sig] = float(sp * nr)

        # Backtest each sizing and get metrics
        trades_bin, eq_bin = backtest_no_txcosts(st, size_binary)
        trades_comb, eq_comb = backtest_no_txcosts(st, size_combined)
        summ_bin = summary_metrics(trades_bin); summ_comb = summary_metrics(trades_comb)

        # Save per-fold equity plots separately
        plt.figure(figsize=(10,5)); plt.plot(eq_bin.reset_index(drop=True)); plt.title(f"Fold {k} - Binary Equity"); plt.grid(True)
        plt.savefig(plots / f"wf_fold_{k}_binary.png", bbox_inches='tight'); plt.close()
        plt.figure(figsize=(10,5)); plt.plot(eq_comb.reset_index(drop=True)); plt.title(f"Fold {k} - Combined Equity"); plt.grid(True)
        plt.savefig(plots / f"wf_fold_{k}_combined.png", bbox_inches='tight'); plt.close()

        # collect metrics
        rows.append({
            'fold': k, 'train_end': train_end, 'test_end': test_end, 'test_trades': len(trades_bin),
            # binary metrics
            'bin_net': summ_bin['Net Profit'], 'bin_trades': summ_bin['Total Trades'], 'bin_pf': summ_bin['Profit Factor'], 'bin_wr': summ_bin['Win Rate (%)'],
            # combined metrics
            'comb_net': summ_comb['Net Profit'], 'comb_trades': summ_comb['Total Trades'], 'comb_pf': summ_comb['Profit Factor'], 'comb_wr': summ_comb['Win Rate (%)']
        })

    # Save fold results and package
    wf_df = pd.DataFrame(rows)
    wf_df.to_csv(outdir / "walkforward_stacked_compare.csv", index=False)

    pkg = outdir / "walkforward_stacked_compare_package.zip"
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in (plots.glob("*.png")): zf.write(p, arcname=f"plots/{p.name}")
        for p in (models.glob("*.joblib")): zf.write(p, arcname=f"models/{p.name}")
        zf.write(outdir / "walkforward_stacked_compare.csv", arcname="walkforward_stacked_compare.csv")

    print("Done. Outputs in:", outdir, "package:", pkg)
    return wf_df

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with OHLC columns")
    parser.add_argument("--outdir", default="wf_stack_compare_out", help="output directory")
    parser.add_argument("--n_estimators", type=int, default=300, help="trees per base model")
    parser.add_argument("--lookahead", type=int, default=10, help="lookahead bars")
    parser.add_argument("--n_splits", type=int, default=5, help="walk-forward folds")
    parser.add_argument("--min_train", type=int, default=200, help="min train size for local OOF")
    parser.add_argument("--n_local_splits", type=int, default=4, help="local OOF splits inside train")
    parser.add_argument("--threshold", type=float, default=0.5, help="binary threshold for stacked prob")
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
        print("Warning: XGBoost not available; running RF-only stacked WF comparison.")

    walkforward_compare(df, args.outdir, n_estimators=args.n_estimators, lookahead=args.lookahead, n_splits=args.n_splits, min_train=args.min_train, n_local_splits=args.n_local_splits, threshold=args.threshold)

if __name__ == "__main__":
    main()
