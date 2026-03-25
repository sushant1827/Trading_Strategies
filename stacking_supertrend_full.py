#!/usr/bin/env python3
"""
stacking_supertrend_full.py

Full stacking pipeline over Supertrend signals with walk-forward CV & backtest.

Usage:
 python stacking_supertrend_full.py --data /path/to/NIFTY50_INDEX_60_Min.csv --outdir ./stack_outputs --n_estimators 300 --lookahead 10

Outputs:
 - outdir/models/*.joblib  (base full models + meta models + regressors)
 - outdir/plots/*.png      (equity curves per-fold and model)
 - outdir/*.csv            (summaries)
 - outdir/full_package.zip (everything)
"""
import argparse
import os
import warnings
from pathlib import Path
import zipfile
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Optional libraries
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False


# ----------------------------
# Supertrend and feature builder
# ----------------------------
def compute_atr(df, length=14):
    h = df['high']; l = df['low']; c = df['close']
    tr1 = (h - l).abs()
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def supertrend(df, period=10, multiplier=3.0):
    df = df.copy().reset_index(drop=True)
    df['atr'] = compute_atr(df, length=period)
    hl2 = (df['high'] + df['low'])/2.0
    df['upperband'] = hl2 + multiplier * df['atr']
    df['lowerband'] = hl2 - multiplier * df['atr']
    df['supertrend'] = np.nan; df['trend'] = 1
    prev_up = df['upperband'].iloc[0]; prev_dn = df['lowerband'].iloc[0]; prev_trend = 1
    for i in range(len(df)):
        if i == 0:
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
        df.at[i,'supertrend'] = up if trend == 1 else dn
        df.at[i,'trend'] = trend
        prev_up, prev_dn, prev_trend = up, dn, trend
    return df

def generate_signals_from_trend(df):
    df = df.copy().reset_index(drop=True)
    df['signal'] = 0
    df.loc[(df['trend'] == 1) & (df['trend'].shift(1) == -1), 'signal'] = 1
    df.loc[(df['trend'] == -1) & (df['trend'].shift(1) == 1), 'signal'] = -1
    return df

# ----------------------------
# Backtest (no tx costs)
# ----------------------------
def backtest_no_txcosts(df, size_series, use_trail=False, trail_mult=2.0):
    df = df.copy().reset_index(drop=True)
    pos = 0.0; entry_price=None; trades=[]; cum=0.0; equity=[]
    trail_stop = None
    for i in range(len(df)-1):
        equity.append(cum)
        if pos == 0:
            if df.loc[i,'signal'] == 1 and size_series[i] > 0:
                entry_price = df.loc[i+1,'open']
                pos = float(size_series[i])
                if use_trail:
                    trail_stop = entry_price - trail_mult * df.loc[i+1,'atr']
        else:
            if df.loc[i,'signal'] == -1:
                exit_price = df.loc[i+1,'open']
                profit = (exit_price - entry_price) * pos
                trades.append({'entry_index':i, 'exit_index':i+1, 'entry_price':entry_price, 'exit_price':exit_price, 'position':pos, 'profit':profit})
                cum += profit; pos = 0.0; entry_price=None; trail_stop=None
            elif use_trail:
                current_close = df.loc[i,'close']
                new_stop = current_close - trail_mult * df.loc[i,'atr']
                trail_stop = max(trail_stop, new_stop) if trail_stop is not None else new_stop
                next_open = df.loc[i+1,'open']
                if next_open <= trail_stop:
                    exit_price = next_open
                    profit = (exit_price - entry_price) * pos
                    trades.append({'entry_index':i, 'exit_index':i+1, 'entry_price':entry_price, 'exit_price':exit_price, 'position':pos, 'profit':profit, 'stopped':True})
                    cum += profit; pos = 0.0; entry_price=None; trail_stop=None
    equity.append(cum)
    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(equity, index=range(len(equity)))
    return trades_df, equity_series

def summary_metrics(trades_df):
    if trades_df.empty:
        return {'Total Trades':0, 'Net Profit':0.0, 'Profit Factor':np.nan, 'Win Rate (%)':np.nan}
    wins = trades_df[trades_df['profit']>0]; losses = trades_df[trades_df['profit']<=0]
    total = len(trades_df); net = trades_df['profit'].sum()
    pf = wins['profit'].sum() / (-losses['profit'].sum()) if losses['profit'].sum()!=0 else np.nan
    win_rate = len(wins) / total * 100
    return {'Total Trades': total, 'Net Profit': round(net,2), 'Profit Factor': round(pf,2), 'Win Rate (%)': round(win_rate,2)}

# ----------------------------
# OOF expanding-window functions
# ----------------------------
def expanding_oof_classifier(clf_ctor, clf_kwargs, X, y, n_splits=6, min_train=50):
    """
    Produces expanding-window OOF probabilities for classifier.
    Returns:
      - oof_probs: array(len(X)) with probabilities (0 for early indices where we couldn't produce OOF)
      - trained_full_model: model trained on whole X,y
    """
    n = len(X)
    oof = np.zeros(n)
    fold_size = max(1, n // (n_splits + 1))
    for k in range(1, n_splits+1):
        train_end = k * fold_size
        test_end = min((k+1) * fold_size, n)
        if test_end <= train_end or train_end < min_train:
            continue
        X_tr, y_tr = X.iloc[:train_end], y[:train_end]
        X_te = X.iloc[train_end:test_end]
        clf = clf_ctor(**clf_kwargs)
        clf.fit(X_tr, y_tr)
        oof[train_end:test_end] = clf.predict_proba(X_te)[:,1]
    # train full
    clf_full = clf_ctor(**clf_kwargs)
    clf_full.fit(X, y)
    return oof, clf_full

def expanding_oof_regressor(reg_ctor, reg_kwargs, X, y_reg, n_splits=6, min_train=50):
    n = len(X)
    oof = np.zeros(n)
    fold_size = max(1, n // (n_splits + 1))
    for k in range(1, n_splits+1):
        train_end = k * fold_size
        test_end = min((k+1) * fold_size, n)
        if test_end <= train_end or train_end < min_train:
            continue
        X_tr, y_tr = X.iloc[:train_end], y_reg[:train_end]
        X_te = X.iloc[train_end:test_end]
        reg = reg_ctor(**reg_kwargs)
        reg.fit(X_tr, y_tr)
        oof[train_end:test_end] = reg.predict(X_te)
    reg_full = reg_ctor(**reg_kwargs); reg_full.fit(X, y_reg)
    return oof, reg_full

# ----------------------------
# Pipeline
# ----------------------------
def build_features_and_labels(df, atr_period=10, multiplier=3.0, lookahead=10):
    st = supertrend(df, period=atr_period, multiplier=multiplier)
    st = generate_signals_from_trend(st)
    st = st.reset_index(drop=True)
    # safe features
    st['ret_1'] = st['close'].pct_change(1).fillna(0)
    st['ret_5'] = st['close'].pct_change(5).fillna(0)
    st['vol_20'] = st['ret_1'].rolling(20, min_periods=1).std().fillna(0)
    st['atr_feat'] = st['atr'].shift(1).fillna(method='bfill')
    st['mom1'] = st['ret_1'].shift(1).fillna(0)
    st['mom5'] = st['ret_5'].shift(1).fillna(0)
    st['vol20'] = st['vol_20'].shift(1).fillna(0)
    st['cor'] = (st['close']/st['open'] - 1.0).shift(1).fillna(0)

    signals = st[(st['signal'] == 1)].copy().reset_index()
    features = []; labels = []; rets = []; indices = []
    for _, row in signals.iterrows():
        i = int(row['index'])
        if i + lookahead < len(st) - 1:
            entry_price = st.loc[i+1,'open']
            future_price = st.loc[i+lookahead,'close']
            future_return = future_price - entry_price
            label = 1 if future_return > 0 else 0
            feat = {
                'atr': st.at[i,'atr_feat'],
                'mom1': st.at[i,'mom1'],
                'mom5': st.at[i,'mom5'],
                'vol20': st.at[i,'vol20'],
                'cor': st.at[i,'cor']
            }
            features.append(feat); labels.append(label); rets.append(future_return); indices.append(i)
    X = pd.DataFrame(features).fillna(0)
    y = np.array(labels)
    yreg = np.array(rets)
    return st, X, y, yreg, indices

def run_full_stack(df, outdir, n_estimators=300, lookahead=10):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"; plots_dir.mkdir(exist_ok=True)
    models_dir = outdir / "models"; models_dir.mkdir(exist_ok=True)

    st, X, y, yreg, indices = build_features_and_labels(df, lookahead=lookahead)
    if len(X) < 30:
        print("Warning: small number of labeled signals:", len(X))

    # chronological train/test split for final evaluation (train on first 70% of labeled signals)
    split = int(0.7 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    yreg_train, yreg_test = yreg[:split], yreg[split:]

    # Base model list
    base_constructors = []
    base_constructors.append(('RF', RandomForestClassifier, {'n_estimators': n_estimators, 'random_state':42}))
    if HAVE_XGB:
        base_constructors.append(('XGB', XGBClassifier, {'n_estimators': n_estimators, 'use_label_encoder':False, 'eval_metric':'logloss', 'random_state':42}))
    if HAVE_LGB:
        base_constructors.append(('LGB', lgb.LGBMClassifier, {'n_estimators': n_estimators, 'random_state':42}))

    print("Base models:", [b[0] for b in base_constructors])

    # Produce OOF probs per base model on training set (expanding windows)
    oof_matrix = np.zeros((len(X), len(base_constructors)))
    full_models = {}
    # Use same n_splits for OOF
    n_splits_oof = 6
    for j, (name, ctor, kwargs) in enumerate(base_constructors):
        print(f"Doing expanding OOF for {name} ...")
        oof_probs, full_model = expanding_oof_classifier(ctor, kwargs, X, y, n_splits=n_splits_oof, min_train=50)
        oof_matrix[:, j] = oof_probs
        full_models[name] = full_model
        # Save full base model
        joblib.dump(full_model, models_dir / f"{name}_full.joblib")

    # Train OOF regressor as well (for sizing) on X,yreg
    print("Producing OOF regressor predictions ...")
    oof_reg, reg_full = expanding_oof_regressor(GradientBoostingRegressor, {'n_estimators': 100, 'random_state':42}, X, yreg, n_splits=n_splits_oof, min_train=50)
    joblib.dump(reg_full, models_dir / f"reg_full.joblib")

    # Build meta features: columns are base OOF probs (for rows where available)
    meta_df = pd.DataFrame(oof_matrix, columns=[b[0] for b in base_constructors])
    mask = (meta_df.sum(axis=1) != 0)  # rows where we have OOF predictions
    if mask.sum() < 10:
        print("Warning: very few OOF rows available for meta training:", mask.sum())

    # Meta-learner training (classifier)
    meta_X_train = meta_df[mask & (np.arange(len(meta_df)) < split)]
    meta_y_train = y[mask & (np.arange(len(meta_df)) < split)]
    # fallback: if meta X empty, use base full model probabilities on training set
    if meta_X_train.shape[0] < 5:
        print("Not enough OOF rows inside training slice for meta learning. Using full-model probs fallback.")
        base_probs_train = np.column_stack([full_models[name].predict_proba(X)[:,1] for name,_,_ in base_constructors])
        meta_X_train = pd.DataFrame(base_probs_train[:split,:], columns=[b[0] for b in base_constructors])
        meta_y_train = y_train

    meta_clf = LogisticRegression(max_iter=500)
    meta_clf.fit(meta_X_train, meta_y_train)
    joblib.dump(meta_clf, models_dir / "meta_clf.joblib")

    # For final sizing regressor, train a meta regressor on OOF reg preds
    meta_reg_feats = pd.DataFrame({'oof_reg': oof_reg})
    meta_reg_train = meta_reg_feats[mask & (np.arange(len(meta_reg_feats)) < split)]
    meta_reg_y = yreg[mask & (np.arange(len(meta_reg_feats)) < split)]
    if meta_reg_train.shape[0] < 5:
        print("Fallback: train meta regressor on full training set X_train -> yreg_train")
        meta_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
        meta_reg.fit(X_train, yreg_train)
    else:
        meta_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
        meta_reg.fit(meta_reg_train.values, meta_reg_y)
    joblib.dump(meta_reg, models_dir / "meta_reg.joblib")

    # ----------------------------
    # Evaluate on held-out test set using stacked model built from training
    # ----------------------------
    # Build base full-model probs for all labeled rows
    base_full_probs = np.column_stack([full_models[name].predict_proba(X)[:,1] for name,_,_ in base_constructors])
    # Meta stacked prob on all labeled rows
    stacked_probs_all = meta_clf.predict_proba(pd.DataFrame(base_full_probs, columns=[b[0] for b in base_constructors]))[:,1]
    # For size, use meta_reg predictions (if meta_reg uses oof_reg feature, produce input likewise)
    # Here simply combine stacked prob * normalized meta_reg prediction (fallback)
    # Predict reg on X
    try:
        pred_reg_all = meta_reg.predict(meta_reg_feats.values)
    except Exception:
        pred_reg_all = meta_reg.predict(X)

    # Normalize predicted returns using training distribution:
    min_r, max_r = yreg_train.min(), yreg_train.max()
    rrange = max_r - min_r if max_r > min_r else 1.0
    norm_pred_ret_all = np.clip((pred_reg_all - min_r) / rrange, 0, 1)

    # Build size arrays mapping back to the original bars (length = len(st))
    size_mlbin = np.zeros(len(st))
    size_comb = np.zeros(len(st))
    # Choose threshold via nested holdout on training meta set (simple approach: 0.5)
    chosen_threshold = 0.5
    for idx_sig, sp, nr in zip(indices, stacked_probs_all, norm_pred_ret_all):
        size_mlbin[idx_sig] = 1.0 if sp > chosen_threshold else 0.0
        size_comb[idx_sig] = float(sp * nr)

    # Backtest on full series
    trades_bin, eq_bin = backtest_no_txcosts(st, size_mlbin)
    trades_comb, eq_comb = backtest_no_txcosts(st, size_comb)
    summ_bin = summary_metrics(trades_bin); summ_comb = summary_metrics(trades_comb)
    print("\nFinal holdout-style backtest (stacked model applied to full series):")
    print("ML-binary:", summ_bin)
    print("Combined :", summ_comb)

    # Save plots for final
    plt.figure(figsize=(10,6))
    plt.plot(eq_bin.reset_index(drop=True), label='Stacked ML-binary')
    plt.plot(eq_comb.reset_index(drop=True), label='Stacked Combined')
    plt.title("Final Stacked Model Equity (no tx costs)")
    plt.xlabel("Bar index"); plt.ylabel("Cumulative points")
    plt.legend(); plt.grid(True)
    plt.savefig(plots_dir / "final_stacked_equity.png", bbox_inches='tight'); plt.close()

    # Save summary CSV including base-model and stacked-model metrics
    rows = []
    # Base model metrics (compute classifier metrics on test slice using base_full_probs)
    for j, (name,_,_) in enumerate(base_constructors):
        probs = base_full_probs[:, j]
        ypred_test = (probs[split:] > 0.5).astype(int)
        acc = accuracy_score(y_test, ypred_test); prec = precision_score(y_test, ypred_test, zero_division=0)
        rec = recall_score(y_test, ypred_test, zero_division=0); f1 = f1_score(y_test, ypred_test, zero_division=0)
        # backtest using base model binary sizes
        size_base = np.zeros(len(st))
        for idx_sig, p in zip(indices, probs):
            size_base[idx_sig] = 1.0 if p > 0.5 else 0.0
        trades_base, eq_base = backtest_no_txcosts(st, size_base)
        sum_base = summary_metrics(trades_base)
        rows.append({'model': name, 'test_acc': acc, 'test_prec': prec, 'test_rec': rec, 'test_f1': f1,
                     'bin_net': sum_base['Net Profit'], 'bin_trades': sum_base['Total Trades']})

    # stacked meta metrics
    stacked_test_preds = (stacked_probs_all[split:] > chosen_threshold).astype(int)
    acc_s = accuracy_score(y_test, stacked_test_preds); prec_s = precision_score(y_test, stacked_test_preds, zero_division=0)
    rec_s = recall_score(y_test, stacked_test_preds, zero_division=0); f1_s = f1_score(y_test, stacked_test_preds, zero_division=0)
    rows.append({'model': 'StackedMeta', 'test_acc': acc_s, 'test_prec': prec_s, 'test_rec': rec_s, 'test_f1': f1_s,
                 'bin_net': summ_bin['Net Profit'], 'bin_trades': summ_bin['Total Trades']})
    pd.DataFrame(rows).to_csv(outdir / "model_comparison_summary.csv", index=False)

    # ----------------------------
    # Walk-forward CV (expanding-window) per-fold: train base+meta on expanding-window and test per-fold
    # ----------------------------
    print("\nRunning walk-forward CV (expanding-window)...")
    n_splits_wf = 4
    n = len(X)
    fold_size = max(1, n // (n_splits_wf + 1))
    wf_rows = []
    wf_plot_files = []
    for k in range(1, n_splits_wf + 1):
        train_end = k * fold_size
        test_end = min((k+1) * fold_size, n)
        if test_end <= train_end:
            break
        print(f"WF fold {k}: train_end={train_end}, test_end={test_end}")
        X_tr, y_tr = X.iloc[:train_end], y[:train_end]
        X_te, y_te = X.iloc[train_end:test_end], y[train_end:test_end]
        idx_te = indices[train_end:test_end]

        # Train base models on X_tr
        base_full_fold = {}
        for name, ctor, kwargs in base_constructors:
            mdl = ctor(**kwargs)
            mdl.fit(X_tr, y_tr)
            base_full_fold[name] = mdl

        # produce stacked features for X_te from base_full_fold
        base_probs_te = np.column_stack([base_full_fold[name].predict_proba(X_te)[:,1] for name,_,_ in base_constructors])
        meta_feats_te = pd.DataFrame(base_probs_te, columns=[b[0] for b in base_constructors])
        # train meta on expanding OOF inside X_tr: to keep it time-safe we'll build OOF inside X_tr
        # building small OOF within training window
        oof_train_small = np.zeros((len(X_tr), len(base_constructors)))
        small_foldsize = max(1, len(X_tr)//4)
        for j,(nm,ctor,kw) in enumerate(base_constructors):
            for kk in range(1,4):
                tr_end = kk * small_foldsize
                te_end = min((kk+1)*small_foldsize, len(X_tr))
                if te_end <= tr_end or tr_end < 10:
                    continue
                local_clf = ctor(**kw)
                local_clf.fit(X_tr.iloc[:tr_end], y_tr[:tr_end])
                oof_train_small[tr_end:te_end, j] = local_clf.predict_proba(X_tr.iloc[tr_end:te_end])[:,1]
        meta_X_tr_small = pd.DataFrame(oof_train_small, columns=[b[0] for b in base_constructors])
        mask_small = (meta_X_tr_small.sum(axis=1) != 0)
        if mask_small.sum() < 5:
            # fallback: meta features from full base models
            meta_clf_fold = LogisticRegression(max_iter=300)
            base_probs_tr_full = np.column_stack([base_full_fold[name].predict_proba(X_tr)[:,1] for name,_,_ in base_constructors])
            meta_clf_fold.fit(pd.DataFrame(base_probs_tr_full, columns=[b[0] for b in base_constructors]), y_tr)
        else:
            meta_clf_fold = LogisticRegression(max_iter=300)
            meta_clf_fold.fit(meta_X_tr_small[mask_small], y_tr[mask_small])

        # sizing regressor for fold: reg on X_tr -> yreg_tr
        reg_fold = GradientBoostingRegressor(n_estimators=100, random_state=42)
        reg_fold.fit(X_tr, yreg[:train_end])

        # produce sizes for test indices (map probabilities back to full st indices)
        base_probs_full_te = []
        for name in [b[0] for b in base_constructors]:
            base_probs_full_te.append(base_full_fold[name].predict_proba(X)[:,1])  # full on all labeled rows
        base_probs_full_te = np.column_stack(base_probs_full_te)
        stacked_probs_fold = meta_clf_fold.predict_proba(pd.DataFrame(base_probs_full_te, columns=[b[0] for b in base_constructors]))[:,1]
        # reg predict for all labeled rows
        predreg_full = reg_fold.predict(X)
        minr, maxr = yreg[:train_end].min(), yreg[:train_end].max()
        rrange_fold = maxr - minr if maxr > minr else 1.0
        normreg_full = np.clip((predreg_full - minr)/rrange_fold, 0, 1)

        size_fold = np.zeros(len(st))
        for idx_sig, sp, nr in zip(indices, stacked_probs_fold, normreg_full):
            size_fold[idx_sig] = float(sp * nr)

        trades_wf, eq_wf = backtest_no_txcosts(st, size_fold)
        summ = summary_metrics(trades_wf)
        summ['fold'] = k; summ['train_end'] = train_end; summ['test_end'] = test_end
        wf_rows.append(summ)

        # save plot per fold
        plt.figure(figsize=(10,6))
        plt.plot(eq_wf.reset_index(drop=True), label=f'WF fold {k} equity (stacked combined)')
        plt.title(f"Walk-forward Fold {k} Equity (stacked combined)")
        plt.xlabel("Bar index"); plt.ylabel("Cumulative points")
        plt.grid(True); plt.legend()
        pfile = plots_dir / f"wf_fold_{k}_stacked.png"
        plt.savefig(pfile, bbox_inches='tight'); plt.close()
        wf_plot_files.append(str(pfile))

    # write walk-forward csv
    pd.DataFrame(wf_rows).to_csv(outdir / "walkforward_stacked_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(outdir / "model_comparison_summary.csv", index=False)

    # zip outputs
    pkg = outdir / "stacking_full_package.zip"
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in (plots_dir.glob("*.png")): zf.write(p, arcname=f"plots/{p.name}")
        for p in (models_dir.glob("*.joblib")): zf.write(p, arcname=f"models/{p.name}")
        for p in outdir.glob("*.csv"): zf.write(p, arcname=p.name)
    print(f"All artifacts saved to {outdir}, package: {pkg}")
    return outdir, pkg

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV OHLC file path (open,high,low,close required)")
    parser.add_argument("--outdir", default="stack_outputs", help="output directory")
    parser.add_argument("--n_estimators", type=int, default=300, help="trees for base learners")
    parser.add_argument("--lookahead", type=int, default=10, help="lookahead bars used to label positive returns")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {'open','high','low','close'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")

    print("Starting stacking + walk-forward pipeline...")
    outdir, pkg = run_full_stack(df, args.outdir, n_estimators=args.n_estimators, lookahead=args.lookahead)
    print("Done. Outputs:", outdir)
    print("Package:", pkg)

if __name__ == "__main__":
    main()
