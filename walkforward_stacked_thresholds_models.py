#!/usr/bin/env python3
"""
walkforward_stacked_thresholds_models.py

Walk-forward stacked ensemble with per-fold threshold grid search and multi-model base set (RF, XGB, LGB, CAT if available).
- Trains base models per fold (time-safe)
- Builds local expanding-window OOF inside training window for meta training
- Trains meta per-fold, searches threshold grid inside training to pick best binary threshold
- Evaluates Binary (thresholded stacked) vs Combined (prob * normalized regressor) per fold
- Saves per-fold metrics + per-threshold metrics + equity plots + model pickles + package zip

Usage:
 python walkforward_stacked_thresholds_models.py --data /path/to/ohlc.csv --outdir ./wf_out
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
import argparse, warnings, zipfile
from pathlib import Path
import numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")

# Optional libs
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

try:
    from catboost import CatBoostClassifier
    HAVE_CAT = True
except Exception:
    HAVE_CAT = False

# ---------------------------
# Supertrend + safe features
# ---------------------------
def compute_atr(df, length=14):
    h, l, c = df['high'], df['low'], df['close']
    tr1 = (h - l).abs()
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def supertrend(df, period=10, multiplier=1.0):
    df = df.copy().reset_index(drop=True)
    df['atr'] = compute_atr(df, length=period)
    hl2 = (df['high'] + df['low']) / 2.0
    df['upperband'] = hl2 + multiplier * df['atr']
    df['lowerband'] = hl2 - multiplier * df['atr']
    df['supertrend'] = np.nan
    df['trend'] = 1
    prev_up, prev_dn, prev_trend = df['upperband'].iloc[0], df['lowerband'].iloc[0], 1
    for i in range(len(df)):
        if i == 0:
            df.at[i, 'supertrend'] = prev_dn if df['close'].iloc[i] > prev_dn else prev_up
            df.at[i, 'trend'] = prev_trend
            continue
        up = df['upperband'].iloc[i]; dn = df['lowerband'].iloc[i]; close_prev = df['close'].iloc[i-1]
        up = max(up, prev_up) if close_prev > prev_up else up
        dn = min(dn, prev_dn) if close_prev < prev_dn else dn
        trend = prev_trend
        if prev_trend == -1 and df['close'].iloc[i] > prev_dn:
            trend = 1
        elif prev_trend == 1 and df['close'].iloc[i] < prev_up:
            trend = -1
        df.at[i, 'supertrend'] = up if trend == 1 else dn
        df.at[i, 'trend'] = trend
        prev_up, prev_dn, prev_trend = up, dn, trend
    return df

def generate_signals_from_trend(df):
    df = df.copy().reset_index(drop=True)
    df['signal'] = 0
    df.loc[(df['trend'] == 1) & (df['trend'].shift(1) == -1), 'signal'] = 1
    df.loc[(df['trend'] == -1) & (df['trend'].shift(1) == 1), 'signal'] = -1
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
# local expanding OOF for training window (time-safe)
# ---------------------------
def local_expanding_oof_ctor(ctor, ctor_kwargs, X_window, y_window, n_splits_local=4, min_train_local=40):
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
        clf = ctor(**ctor_kwargs)
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
# Main walk-forward with thresholds and expanded base models
# ---------------------------
def walkforward_compare_all(df, outdir, n_estimators=300, lookahead=10, n_splits=5, min_train=200,
                            n_local_splits=4, threshold_min=0.30, threshold_max=0.70, threshold_steps=21):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    plots = outdir / "plots"; plots.mkdir(exist_ok=True)
    models = outdir / "models"; models.mkdir(exist_ok=True)

    st, X, y, yreg, indices = build_features_labels(df, lookahead=lookahead)
    n = len(X)
    if n < 20:
        raise RuntimeError("Too few labeled samples for walk-forward.")

    fold_size = max(1, n // (n_splits + 1))
    all_fold_rows = []
    per_threshold_records = []

    thresholds = np.linspace(threshold_min, threshold_max, threshold_steps)

    for k in range(1, n_splits+1):
        train_end = k * fold_size
        test_end = min((k+1) * fold_size, n)
        if test_end <= train_end:
            break
        print(f"\nFold {k}: train_end={train_end}, test_end={test_end} (labels indices)")

        X_tr, y_tr = X.iloc[:train_end], y[:train_end]
        X_te, y_te = X.iloc[train_end:test_end], y[train_end:test_end]

        # Define base model constructors for this fold
        base_specs = []
        base_specs.append(('RF', RandomForestClassifier, {'n_estimators': n_estimators, 'random_state':42, 'n_jobs': -1}))
        if HAVE_XGB:
            base_specs.append(('XGB', XGBClassifier, {'n_estimators': n_estimators, 'use_label_encoder':False, 'eval_metric':'logloss', 'random_state':42, 'n_jobs':1}))
        if HAVE_LGB:
            base_specs.append(('LGB', lgb.LGBMClassifier, {'n_estimators': n_estimators, 'random_state':42, 'n_jobs': -1}))
        if HAVE_CAT:
            base_specs.append(('CAT', CatBoostClassifier, {'iterations': n_estimators, 'verbose':0, 'random_seed':42}))

        # Train base full models on X_tr (they will be used to produce full-probs)
        base_models = {}
        for name, ctor, kwargs in base_specs:
            print(f"  Train base model {name}")
            mdl = ctor(**kwargs)
            mdl.fit(X_tr, y_tr)
            base_models[name] = mdl
            joblib.dump(mdl, models / f"{name}_fold_{k}.joblib")

        # Build local OOF inside X_tr for each base model (time-safe meta features)
        oof_cols = {}
        for name, ctor, kwargs in base_specs:
            print(f"  Local OOF for {name} (inside train window)")
            oof = local_expanding_oof_ctor(ctor, kwargs, X_tr, y_tr, n_splits_local=n_local_splits, min_train_local=min_train)
            oof_cols[name] = oof

        meta_X_tr = pd.DataFrame(np.column_stack([oof_cols[n] for n,_,_ in base_specs]), columns=[n for n,_,_ in base_specs])
        # fallback if OOF insufficient
        if (meta_X_tr.sum(axis=1) != 0).sum() < max(10, int(0.1*len(X_tr))):
            print("    Insufficient local OOF rows -> fallback to base full-probs on X_tr for meta training")
            base_full_train = np.column_stack([base_models[n].predict_proba(X_tr)[:,1] for n,_,_ in base_specs])
            meta_X_tr = pd.DataFrame(base_full_train, columns=[n for n,_,_ in base_specs])

        # Train meta classifier on meta_X_tr
        meta_clf = LogisticRegression(max_iter=500)
        meta_clf.fit(meta_X_tr, y_tr)
        joblib.dump(meta_clf, models / f"meta_fold_{k}.joblib")

        # Train sizing regressor on X_tr
        reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
        reg.fit(X_tr, yreg[:train_end])
        joblib.dump(reg, models / f"reg_fold_{k}.joblib")

        # Produce base full-probs for all labeled rows (X)
        base_full_probs_all = np.column_stack([base_models[n].predict_proba(X)[:,1] for n,_,_ in base_specs])
        # Stacked probs (meta applied to base full-probs)
        stacked_probs_all = meta_clf.predict_proba(pd.DataFrame(base_full_probs_all, columns=[n for n,_,_ in base_specs]))[:,1]
        # reg predicted returns for all labeled rows
        pred_reg_all = reg.predict(X)
        minr, maxr = yreg[:train_end].min(), yreg[:train_end].max()
        rrange = maxr - minr if maxr > minr else 1.0
        norm_ret_all = np.clip((pred_reg_all - minr) / rrange, 0, 1)

        # --- THRESHOLD GRID SEARCH inside the training window ---
        # Build stacked probs for train slice (we want to tune threshold without peeking into test)
        stacked_train = stacked_probs_all[:train_end]
        y_train_for_grid = y_tr
        best_thresh = thresholds[len(thresholds)//2]
        best_f1 = -1.0
        thresh_rows = []
        for t in thresholds:
            preds_t = (stacked_train > t).astype(int)
            # require at least some positives to compute metrics
            if preds_t.sum() == 0:
                f1 = 0.0
                prec = rec = acc = 0.0
            else:
                f1 = f1_score(y_train_for_grid, preds_t, zero_division=0)
                prec = precision_score(y_train_for_grid, preds_t, zero_division=0)
                rec = recall_score(y_train_for_grid, preds_t, zero_division=0)
                acc = accuracy_score(y_train_for_grid, preds_t)
            thresh_rows.append({'fold':k, 'threshold':t, 'train_f1':f1, 'train_prec':prec, 'train_rec':rec, 'train_acc':acc})
            if f1 > best_f1:
                best_f1 = f1; best_thresh = t

        # Save per-threshold training metrics for this fold
        per_threshold_records.extend(thresh_rows)

        # Use best_thresh (found on training stacked probs) on test stacked probs
        stacked_test = stacked_probs_all[train_end:test_end]
        # Build size series (binary using best_thresh and combined using normalized regression)
        size_binary = np.zeros(len(st))
        size_combined = np.zeros(len(st))
        for idx_sig, sp, nr in zip(indices[train_end:test_end], stacked_test, norm_ret_all[train_end:test_end]):
            size_binary[idx_sig] = 1.0 if sp > best_thresh else 0.0
            size_combined[idx_sig] = float(sp * nr)

        # Backtest both sizing approaches (for this fold)
        trades_bin, eq_bin = backtest_no_txcosts(st, size_binary)
        trades_comb, eq_comb = backtest_no_txcosts(st, size_combined)
        summ_bin = summary_metrics(trades_bin)
        summ_comb = summary_metrics(trades_comb)

        # Also compute simple stacked test classification metrics (threshold applied)
        y_test_slice = y[train_end:test_end]
        ypred_test = (stacked_test > best_thresh).astype(int)
        # if sizes of y_test_slice and ypred_test mismatch (shouldn't) handle gracefully:
        try:
            acc_t = accuracy_score(y_test_slice, ypred_test)
            prec_t = precision_score(y_test_slice, ypred_test, zero_division=0)
            rec_t = recall_score(y_test_slice, ypred_test, zero_division=0)
            f1_t = f1_score(y_test_slice, ypred_test, zero_division=0)
        except Exception:
            acc_t = prec_t = rec_t = f1_t = np.nan

        # Save per-fold summary (including best threshold found)
        row = {
            'fold': k, 'train_end': train_end, 'test_end': test_end,
            'best_threshold': best_thresh, 'best_train_f1': best_f1,
            'bin_net': summ_bin['Net Profit'], 'bin_trades': summ_bin['Total Trades'], 'bin_pf': summ_bin['Profit Factor'], 'bin_wr': summ_bin['Win Rate (%)'],
            'comb_net': summ_comb['Net Profit'], 'comb_trades': summ_comb['Total Trades'], 'comb_pf': summ_comb['Profit Factor'], 'comb_wr': summ_comb['Win Rate (%)'],
            'stacked_test_acc': acc_t, 'stacked_test_prec': prec_t, 'stacked_test_rec': rec_t, 'stacked_test_f1': f1_t
        }
        all_fold_rows.append(row)

        # Save per-fold equity plots
        plt.figure(figsize=(10,5))
        plt.plot(eq_bin.reset_index(drop=True), label=f'Fold{k} Binary (th={best_thresh:.3f})')
        plt.title(f'Fold {k} Binary Equity (threshold={best_thresh:.3f})'); plt.grid(True); plt.legend()
        plt.savefig(plots / f"wf_fold_{k}_binary_thresh_{best_thresh:.3f}.png", bbox_inches='tight'); plt.close()

        plt.figure(figsize=(10,5))
        plt.plot(eq_comb.reset_index(drop=True), label=f'Fold{k} Combined')
        plt.title(f'Fold {k} Combined Equity'); plt.grid(True); plt.legend()
        plt.savefig(plots / f"wf_fold_{k}_combined.png", bbox_inches='tight'); plt.close()

    # Save outputs
    pd.DataFrame(all_fold_rows).to_csv(outdir / "walkforward_thresholds_summary.csv", index=False)
    pd.DataFrame(per_threshold_records).to_csv(outdir / "per_threshold_train_metrics.csv", index=False)

    # Zip artifacts
    pkg = outdir / "walkforward_thresholds_models_package.zip"
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in (plots.glob("*.png")): zf.write(p, arcname=f"plots/{p.name}")
        for p in (models.glob("*.joblib")): zf.write(p, arcname=f"models/{p.name}")
        for p in (outdir.glob("*.csv")): zf.write(p, arcname=p.name)

    print("\nSaved walk-forward thresholds summary and per-threshold metrics.")
    print("Summary:", outdir / "walkforward_thresholds_summary.csv")
    print("Per-threshold training metrics:", outdir / "per_threshold_train_metrics.csv")
    print("Package:", pkg)
    return pd.DataFrame(all_fold_rows), pd.DataFrame(per_threshold_records)

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV OHLC path (open,high,low,close)")
    parser.add_argument("--outdir", default="wf_stack_all_out", help="output directory")
    parser.add_argument("--n_estimators", type=int, default=300, help="trees / iterations per base model")
    parser.add_argument("--lookahead", type=int, default=10, help="lookahead bars for label")
    parser.add_argument("--n_splits", type=int, default=5, help="walk-forward folds")
    parser.add_argument("--min_train", type=int, default=200, help="min local OOF train size inside window")
    parser.add_argument("--n_local_splits", type=int, default=4, help="local OOF splits inside train window")
    parser.add_argument("--threshold_min", type=float, default=0.30, help="grid min threshold")
    parser.add_argument("--threshold_max", type=float, default=0.70, help="grid max threshold")
    parser.add_argument("--threshold_steps", type=int, default=21, help="grid steps")
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
        print("Note: XGBoost not installed — XGB base model will be skipped.")
    if not HAVE_LGB:
        print("Note: LightGBM not installed — LGB base model will be skipped.")
    if not HAVE_CAT:
        print("Note: CatBoost not installed — CAT base model will be skipped.")

    walkforward_compare_all(df, args.outdir, n_estimators=args.n_estimators, lookahead=args.lookahead,
                            n_splits=args.n_splits, min_train=args.min_train, n_local_splits=args.n_local_splits,
                            threshold_min=args.threshold_min, threshold_max=args.threshold_max, threshold_steps=args.threshold_steps)

if __name__ == "__main__":
    main()
