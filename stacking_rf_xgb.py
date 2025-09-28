#!/usr/bin/env python3
"""
stacking_rf_xgb.py
RF + XGBoost base learners + LogisticRegression stacker.
Time-series-safe expanding-window OOF -> meta-learner -> final backtests + walk-forward CV.
No transaction costs (easy to add later).

Usage:
 python stacking_rf_xgb.py --data /path/to/ohlc.csv --outdir ./out --n_estimators 300 --lookahead 10 --n_jobs -1
"""
import argparse, warnings, os, zipfile
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")

# Try XGBoost
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# ------------------------
# Supertrend, features, labels
# ------------------------
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
    df.loc[(df['trend']==1) & (df['trend'].shift(1)==-1),'signal'] = 1
    df.loc[(df['trend']==-1) & (df['trend'].shift(1)==1),'signal'] = -1
    return df

def build_features_labels(df, lookahead=10):
    st = supertrend(df, period=10, multiplier=3.0)
    st = generate_signals_from_trend(st).reset_index(drop=True)
    # simple safe features (lagged)
    st['ret1'] = st['close'].pct_change(1).fillna(0)
    st['ret5'] = st['close'].pct_change(5).fillna(0)
    st['vol20'] = st['ret1'].rolling(20,min_periods=1).std().fillna(0)
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
    y = np.array(labels)
    yreg = np.array(rets)
    return st, X, y, yreg, indices

# ------------------------
# OOF expanding-window (time-safe)
# ------------------------
def expanding_oof_classifier(clf_ctor, clf_kwargs, X, y, n_splits=6, min_train=40):
    n = len(X)
    oof = np.zeros(n)
    fold_size = max(1, n // (n_splits + 1))
    for k in range(1, n_splits+1):
        train_end = k * fold_size
        test_end = min((k+1)*fold_size, n)
        if test_end <= train_end or train_end < min_train:
            continue
        X_tr, y_tr = X.iloc[:train_end], y[:train_end]
        X_te = X.iloc[train_end:test_end]
        clf = clf_ctor(**clf_kwargs)
        clf.fit(X_tr, y_tr)
        oof[train_end:test_end] = clf.predict_proba(X_te)[:,1]
    clf_full = clf_ctor(**clf_kwargs)
    clf_full.fit(X, y)
    return oof, clf_full

def expanding_oof_regressor(reg_ctor, reg_kwargs, X, yreg, n_splits=6, min_train=40):
    n = len(X)
    oof = np.zeros(n)
    fold_size = max(1, n // (n_splits + 1))
    for k in range(1, n_splits+1):
        train_end = k * fold_size
        test_end = min((k+1)*fold_size, n)
        if test_end <= train_end or train_end < min_train:
            continue
        X_tr = X.iloc[:train_end]; y_tr = yreg[:train_end]
        X_te = X.iloc[train_end:test_end]
        reg = reg_ctor(**reg_kwargs)
        reg.fit(X_tr, y_tr)
        oof[train_end:test_end] = reg.predict(X_te)
    reg_full = reg_ctor(**reg_kwargs); reg_full.fit(X, yreg)
    return oof, reg_full

# ------------------------
# Backtest (no costs)
# ------------------------
def backtest_no_txcosts(df, size_series):
    df = df.copy().reset_index(drop=True)
    pos = 0.0; entry_price=None; trades=[]; cum=0.0; equity=[]
    for i in range(len(df)-1):
        equity.append(cum)
        if pos == 0:
            if df.loc[i,'signal']==1 and size_series[i]>0:
                entry_price = df.loc[i+1,'open']; pos = float(size_series[i])
        else:
            if df.loc[i,'signal']==-1:
                exit_price = df.loc[i+1,'open']
                profit = (exit_price - entry_price) * pos
                trades.append({'entry':i,'exit':i+1,'profit':profit})
                cum += profit; pos=0.0; entry_price=None
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

# ------------------------
# Main pipeline
# ------------------------
def run_pipeline(df, outdir, n_estimators=300, lookahead=10, n_jobs=1):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"; plots_dir.mkdir(exist_ok=True)
    models_dir = outdir / "models"; models_dir.mkdir(exist_ok=True)

    st, X, y, yreg, indices = build_features_labels(df, lookahead=lookahead)
    if len(X) < 30:
        print("Warning: few labeled samples:", len(X))

    # chronological train/test split for evaluation
    split = int(0.7 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    yreg_train, yreg_test = yreg[:split], yreg[split:]

    # Prepare base models: RF always, XGB if available
    base_models = []
    base_models.append(('RF', RandomForestClassifier, {'n_estimators': n_estimators, 'random_state':42, 'n_jobs': n_jobs}))
    if HAVE_XGB:
        base_models.append(('XGB', XGBClassifier, {'n_estimators': n_estimators, 'use_label_encoder':False, 'eval_metric':'logloss', 'n_jobs': n_jobs, 'random_state':42}))
    else:
        print("XGBoost not available; running RF-only stack (stacker will be RF only).")

    # OOF for classifiers and regressor
    oof_matrix = np.zeros((len(X), len(base_models)))
    full_models = {}
    for j,(name, ctor, kwargs) in enumerate(base_models):
        print("Running expanding OOF for", name)
        oof_probs, full = expanding_oof_classifier(ctor, kwargs, X, y, n_splits=6, min_train=40)
        oof_matrix[:, j] = oof_probs
        full_models[name] = full
        joblib.dump(full, models_dir / f"{name}_full.joblib")

    # regressor OOF
    oof_reg, reg_full = expanding_oof_regressor(GradientBoostingRegressor, {'n_estimators':100, 'random_state':42}, X, yreg, n_splits=6, min_train=40)
    joblib.dump(reg_full, models_dir / "reg_full.joblib")

    # Meta features: OOF matrix as columns
    meta_df = pd.DataFrame(oof_matrix, columns=[b[0] for b in base_models])
    mask = (meta_df.sum(axis=1) != 0)
    # train meta classifier on OOF rows within training slice; fallback to full-model probs if insufficient OOF
    meta_train_mask = mask & (np.arange(len(meta_df)) < split)
    if meta_train_mask.sum() < 10:
        print("Not enough OOF rows in training slice; using full-model probs fallback for meta training.")
        base_full_train_probs = np.column_stack([full_models[name].predict_proba(X)[:,1] for name,_,_ in base_models])
        meta_X_train = pd.DataFrame(base_full_train_probs[:split,:], columns=[b[0] for b in base_models])
        meta_y_train = y_train
    else:
        meta_X_train = meta_df[meta_train_mask]
        meta_y_train = y[meta_train_mask]

    meta_clf = LogisticRegression(max_iter=500)
    meta_clf.fit(meta_X_train, meta_y_train)
    joblib.dump(meta_clf, models_dir / "meta_clf.joblib")

    # meta regressor: train on OOF reg preds if possible else fallback
    meta_reg_feats = pd.DataFrame({'oof_reg': oof_reg})
    meta_reg_train_mask = (np.arange(len(meta_reg_feats)) < split)
    if meta_reg_feats[meta_reg_train_mask].shape[0] < 10:
        meta_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
        meta_reg.fit(X_train, yreg_train)
    else:
        meta_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
        meta_reg.fit(meta_reg_feats[meta_reg_train_mask].values, yreg[meta_reg_train_mask])
    joblib.dump(meta_reg, models_dir / "meta_reg.joblib")

    # Evaluate base models and stacked model on holdout test slice
    base_full_probs = np.column_stack([full_models[name].predict_proba(X)[:,1] for name,_,_ in base_models])
    stacked_probs_all = meta_clf.predict_proba(pd.DataFrame(base_full_probs, columns=[b[0] for b in base_models]))[:,1]
    # predict meta reg (size)
    try:
        pred_reg_all = meta_reg.predict(meta_reg_feats.values)
    except Exception:
        pred_reg_all = meta_reg.predict(X)
    minr, maxr = yreg_train.min(), yreg_train.max()
    rrange = maxr-minr if maxr>minr else 1.0
    norm_pred_ret_all = np.clip((pred_reg_all - minr)/rrange, 0, 1)

    # Build sizing series (map stacked probs back to original bars)
    size_base_dict = {}
    for j,(name,_,_) in enumerate(base_models):
        size_series = np.zeros(len(st))
        probs = base_full_probs[:, j]
        for idx_sig, p in zip(indices, probs):
            size_series[idx_sig] = 1.0 if p > 0.5 else 0.0
        size_base_dict[name] = size_series

    size_stacked_bin = np.zeros(len(st))
    size_stacked_comb = np.zeros(len(st))
    threshold = 0.5
    for idx_sig, sp, nr in zip(indices, stacked_probs_all, norm_pred_ret_all):
        size_stacked_bin[idx_sig] = 1.0 if sp > threshold else 0.0
        size_stacked_comb[idx_sig] = float(sp * nr)

    # Backtest each
    results_rows = []
    for name in size_base_dict:
        trades, eq = backtest_no_txcosts(st, size_base_dict[name])
        summ = summary_metrics(trades)
        # save plot
        plt.figure(figsize=(8,5)); plt.plot(eq.reset_index(drop=True)); plt.title(f"Equity - {name} (binary)"); plt.grid(True)
        plt.savefig(plots_dir / f"equity_{name}.png", bbox_inches='tight'); plt.close()
        results_rows.append({'model': name, 'test_acc': None, 'test_prec': None, 'test_rec': None, 'test_f1': None,
                             'bin_net': summ['Net Profit'], 'bin_trades': summ['Total Trades']})

    # stacked results
    trades_bin, eq_bin = backtest_no_txcosts(st, size_stacked_bin)
    trades_comb, eq_comb = backtest_no_txcosts(st, size_stacked_comb)
    summ_bin = summary_metrics(trades_bin); summ_comb = summary_metrics(trades_comb)
    # save stacked plots
    plt.figure(figsize=(8,5)); plt.plot(eq_bin.reset_index(drop=True), label='stacked bin'); plt.plot(eq_comb.reset_index(drop=True), label='stacked comb'); plt.legend(); plt.grid(True)
    plt.title("Equity - Stacked (binary & combined)"); plt.savefig(plots_dir / "equity_stacked.png", bbox_inches='tight'); plt.close()

    # compute classification metrics for base & stacked on test slice
    for j,(name,_,_) in enumerate(base_models):
        probs = base_full_probs[:, j]; ypred_test = (probs[split:] > 0.5).astype(int)
        acc = accuracy_score(y_test, ypred_test); prec = precision_score(y_test, ypred_test, zero_division=0); rec = recall_score(y_test, ypred_test, zero_division=0); f1 = f1_score(y_test, ypred_test, zero_division=0)
        # find the entry in results_rows and set metrics
        for r in results_rows:
            if r['model'] == name:
                r.update({'test_acc': acc, 'test_prec': prec, 'test_rec': rec, 'test_f1': f1})
                break

    results_rows.append({'model': 'StackedMeta', 'test_acc': None, 'test_prec': None, 'test_rec': None, 'test_f1': None,
                         'bin_net': summ_bin['Net Profit'], 'bin_trades': summ_bin['Total Trades'],
                         'comb_net': summ_comb['Net Profit'], 'comb_trades': summ_comb['Total Trades']})

    # compute stacked classifiers' test metrics
    ypred_stack_test = (stacked_probs_all[split:] > threshold).astype(int)
    acc_s = accuracy_score(y_test, ypred_stack_test); prec_s = precision_score(y_test, ypred_stack_test, zero_division=0); rec_s = recall_score(y_test, ypred_stack_test, zero_division=0); f1_s = f1_score(y_test, ypred_stack_test, zero_division=0)
    # update stacked row
    results_rows[-1].update({'test_acc': acc_s, 'test_prec': prec_s, 'test_rec': rec_s, 'test_f1': f1_s})

    # Write CSV summary
    pd.DataFrame(results_rows).to_csv(outdir / "model_comparison_summary.csv", index=False)

    # ------------------------
    # Walk-forward CV: expand-window retrain per fold, produce fold equity curves and metrics for stacked combined sizing
    # ------------------------
    n_splits_wf = 4
    n = len(X)
    fold = max(1, n // (n_splits_wf + 1))
    wf_rows = []
    for k in range(1, n_splits_wf+1):
        train_end = k * fold
        test_end = min((k+1)*fold, n)
        if test_end <= train_end:
            break
        X_tr, y_tr = X.iloc[:train_end], y[:train_end]
        X_te, y_te = X.iloc[train_end:test_end], y[train_end:test_end]
        idx_te = indices[train_end:test_end]
        # train base models on X_tr
        fold_base = {}
        for name, ctor, kwargs in base_models:
            mdl = ctor(**kwargs); mdl.fit(X_tr, y_tr); fold_base[name] = mdl
        # train meta on small OOF inside X_tr (or fallback)
        # simple fallback: meta trained on full base probs over X_tr
        base_probs_tr = np.column_stack([fold_base[name].predict_proba(X_tr)[:,1] for name,_,_ in base_models])
        meta_fold = LogisticRegression(max_iter=300); meta_fold.fit(pd.DataFrame(base_probs_tr, columns=[b[0] for b in base_models]), y_tr)
        # size regressor on X_tr
        reg_fold = GradientBoostingRegressor(n_estimators=100, random_state=42); reg_fold.fit(X_tr, yreg[:train_end])
        # produce stacked probs for all labeled indices (using fold models)
        base_probs_all_fold = np.column_stack([fold_base[name].predict_proba(X)[:,1] for name,_,_ in base_models])
        stacked_probs_fold = meta_fold.predict_proba(pd.DataFrame(base_probs_all_fold, columns=[b[0] for b in base_models]))[:,1]
        predreg_fold = reg_fold.predict(X)
        minr_f, maxr_f = yreg[:train_end].min(), yreg[:train_end].max(); rrange_f = maxr_f-minr_f if maxr_f>minr_f else 1.0
        norm_predreg = np.clip((predreg_fold - minr_f)/rrange_f, 0, 1)
        size_series = np.zeros(len(st))
        for idx_sig, sp, nr in zip(indices, stacked_probs_fold, norm_predreg):
            size_series[idx_sig] = float(sp * nr)
        trades_wf, eq_wf = backtest_no_txcosts(st, size_series)
        summ = summary_metrics(trades_wf); summ.update({'fold':k, 'train_end':train_end, 'test_end':test_end})
        wf_rows.append(summ)
        # save per-fold plot
        plt.figure(figsize=(8,5)); plt.plot(eq_wf.reset_index(drop=True)); plt.title(f"WF fold {k} equity (stacked combined)"); plt.grid(True)
        plt.savefig(plots_dir / f"wf_fold_{k}.png", bbox_inches='tight'); plt.close()

    pd.DataFrame(wf_rows).to_csv(outdir / "walkforward_stacked_metrics.csv", index=False)

    # Zip outputs
    pkg = outdir / "stacking_rf_xgb_package.zip"
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in plots_dir.glob("*.png"): zf.write(p, arcname=f"plots/{p.name}")
        for p in models_dir.glob("*.joblib"): zf.write(p, arcname=f"models/{p.name}")
        for p in outdir.glob("*.csv"): zf.write(p, arcname=p.name)
    print("Done. Outputs in:", outdir, "package:", pkg)
    return outdir, pkg

# ------------------------
# CLI
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with open,high,low,close")
    parser.add_argument("--outdir", default="rf_xgb_out", help="output dir")
    parser.add_argument("--n_estimators", type=int, default=300, help="trees per base model")
    parser.add_argument("--lookahead", type=int, default=10, help="lookahead bars for labeling")
    parser.add_argument("--n_jobs", type=int, default=1, help="n_jobs for RF/XGB (-1 for all cores)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError("Data file not found: "+str(data_path))
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {'open','high','low','close'}
    if not required.issubset(set(df.columns)):
        raise ValueError("CSV must contain columns: open, high, low, close")

    if not HAVE_XGB:
        print("Warning: XGBoost not installed or not importable. Script will run RF-only stack. Install xgboost if you want XGB included.")

    run_pipeline(df, args.outdir, n_estimators=args.n_estimators, lookahead=args.lookahead, n_jobs=args.n_jobs)

if __name__ == "__main__":
    main()
