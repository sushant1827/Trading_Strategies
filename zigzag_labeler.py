#!/usr/bin/env python3
"""
zigzag_labeler.py

Detects ZigZag-like pivots (Highs and Lows) and labels them as:
   HH - Higher High
   HL - Higher Low
   LL - Lower Low
   LH - Lower High

Inputs:
   - CSV file with OHLC data (Date, Open, High, Low, Close)
     If not provided/found, a sample OHLC series will be generated.

Outputs:
   - CSV with additional columns: ZigzagPivot, ZigzagLabel, HH, HL, LL, LH
     If Heikin Ashi is enabled, also includes HA_Open, HA_High, HA_Low, HA_Close

Usage:
   python zigzag_labeler.py --input ohlc.csv --output zigzag_points.csv --depth 12 --backstep 2
   python zigzag_labeler.py --input ohlc.csv --output zigzag_ha_points.csv --depth 12 --backstep 2 --use_heikin_ashi
"""
import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd


def add_heikin_ashi_columns(df):
    """
    Adds Heikin Ashi OHLC columns to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with 'Open', 'High', 'Low', 'Close' columns.

    Returns:
        pd.DataFrame: The DataFrame with added Heikin Ashi columns.
    """
    df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    # Calculate HA_Open
    # For the first candle, HA_Open = (Open + Close) / 2
    # For subsequent candles, HA_Open = (Previous HA_Open + Previous HA_Close) / 2
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + df['HA_Close'].iloc[i-1]) / 2
    df['HA_Open'] = ha_open

    df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

    # Round all numerical columns to 2 decimal places
    df = df.round(2)

    print("Added Heikin Ashi columns to DataFrame and rounded values to 2 decimal places.")
    return df


def detect_pivots(df, depth=12, backstep=2, use_heikin_ashi=False):
    """
    Detect pivot highs and lows using a centered window of size (2*depth + 1).
    Returns a list of tuples (index, 'H' or 'L', price).
    backstep: number of bars to skip after detecting a pivot
    use_heikin_ashi: if True, use HA_High and HA_Low for pivot detection
    """
    n = len(df)
    pivot_type = [None] * n
    pivot_price = [np.nan] * n

    i = depth
    skip_until = -1
    high_col = 'HA_High' if use_heikin_ashi else 'High'
    low_col = 'HA_Low' if use_heikin_ashi else 'Low'
    highs = df[high_col].to_numpy()
    lows = df[low_col].to_numpy()

    while i < n - depth:
        if i <= skip_until:
            i += 1
            continue

        window_high = highs[i - depth: i + depth + 1]
        window_low = lows[i - depth: i + depth + 1]
        cur_high = highs[i]
        cur_low = lows[i]

        # pivot high if current high is the unique maximum in the window
        if cur_high >= window_high.max() and (window_high == cur_high).sum() == 1:
            pivot_type[i] = 'H'
            pivot_price[i] = cur_high
            skip_until = i + backstep
        # pivot low if current low is the unique minimum in the window
        elif cur_low <= window_low.min() and (window_low == cur_low).sum() == 1:
            pivot_type[i] = 'L'
            pivot_price[i] = cur_low
            skip_until = i + backstep

        i += 1

    pivots = [(idx, pivot_type[idx], pivot_price[idx])
              for idx in range(n) if pivot_type[idx] is not None]
    return pivots


def label_pivots(pivots):
    """
    Given chronological pivots (index, type, price), produce labels HH/HL/LL/LH.
    Returns a dict mapping pivot_index -> label (or None for the first pivot).
    """
    labels = {}
    last_point_price = None

    for idx, ptype, price in pivots:
        if last_point_price is None:
            labels[idx] = None  # can't compare first pivot
            last_point_price = price
            continue

        if ptype == 'H':
            labels[idx] = 'HH' if price > last_point_price else 'LH'
        elif ptype == 'L':
            labels[idx] = 'LL' if price < last_point_price else 'HL'
        else:
            labels[idx] = None

        last_point_price = price

    return labels


def prepare_dataframe(df, pivots, labels):
    """
    Add columns ZigzagPivot, ZigzagLabel, HH, HL, LL, LH to the dataframe.
    For rows without a pivot, these will be NaN / None.
    """
    df = df.copy().reset_index()
    df['ZigzagPivot'] = np.nan
    df['ZigzagLabel'] = None
    df['HH'] = np.nan
    df['HL'] = np.nan
    df['LL'] = np.nan
    df['LH'] = np.nan

    for idx, ptype, price in pivots:
        lbl = labels.get(idx, None)
        df.at[idx, 'ZigzagPivot'] = price
        df.at[idx, 'ZigzagLabel'] = lbl
        if lbl == 'HH':
            df.at[idx, 'HH'] = price
        elif lbl == 'HL':
            df.at[idx, 'HL'] = price
        elif lbl == 'LL':
            df.at[idx, 'LL'] = price
        elif lbl == 'LH':
            df.at[idx, 'LH'] = price

    return df


def load_or_generate_ohlc(input_path):
    """
    Try to load CSV with columns Date, Open, High, Low, Close.
    If not present, generate a sample OHLC dataframe.
    """
    # if input_path and os.path.exists(input_path):
    #     df = pd.read_csv(input_path)
    #     # Try to parse Date column if present
    #     if 'Date' in df.columns:
    #         try:
    #             df['Date'] = pd.to_datetime(df['Date'])
    #         except Exception:
    #             pass
    #     # Ensure required columns
    #     for c in ['Open', 'High', 'Low', 'Close']:
    #         if c not in df.columns:
    #             raise ValueError(f"Input CSV must contain column '{c}'.")
    #     return df.reset_index(drop=True)

    # # Generate sample random-walk OHLC if no file provided
    # np.random.seed(42)
    # n = 600
    # dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='T')
    # price = 100 + np.cumsum(np.random.randn(n) * 0.2)  # random walk
    # high = price + np.abs(np.random.randn(n) * 0.2)
    # low = price - np.abs(np.random.randn(n) * 0.2)
    # openp = price + np.random.randn(n) * 0.05
    # closep = price + np.random.randn(n) * 0.05
    # df = pd.DataFrame({'Date': dates, 'Open': openp, 'High': high, 'Low': low, 'Close': closep})
    # return df


def main(args):
    # df = load_or_generate_ohlc(args.input)
    csv_file_path = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'
    df = pd.read_csv(csv_file_path)
    df = df.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
    df = df.rename(columns={'Date': 'DateTime'})
    df = df.set_index('DateTime')
    df.index = pd.to_datetime(df.index)

    df = df[~df.index.duplicated(keep='first')]
    print(f"After dropping duplicates, df index is unique: {df.index.is_unique}")

    # Filter data from 2021 onwards
    df = df[df.index.year >= 2021]
    print(f"After filtering for 2021 onwards, df shape: {df.shape}")

    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].round(2)

    # Add Heikin Ashi columns if enabled
    if args.use_heikin_ashi:
        df = add_heikin_ashi_columns(df)

    pivots = detect_pivots(df, depth=args.depth, backstep=args.backstep, use_heikin_ashi=args.use_heikin_ashi)
    labels = label_pivots(pivots)
    result = prepare_dataframe(df, pivots, labels)
    result.to_csv(args.output, index=False)
    print(f"Processed {len(df)} rows.")
    print(f"Detected {len(pivots)} pivots.")
    print(f"Output saved to: {args.output}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="ZigZag pivot labeler (HH/HL/LL/LH).")
    p.add_argument('--input', '-i', default='ohlc.csv', help='Input OHLC CSV (Date,Open,High,Low,Close)')
    p.add_argument('--output', '-o', default='zigzag_points.csv', help='Output CSV filename')
    p.add_argument('--depth', '-d', type=int, default=12, help='Depth (lookback/lookahead window size)')
    p.add_argument('--backstep', '-b', type=int, default=2, help='Backstep (bars to skip after a pivot)')
    p.add_argument('--use_heikin_ashi', action='store_true', help='Use Heikin Ashi candles for pivot calculation instead of normal OHLC')
    args = p.parse_args()

    try:
        main(args)
    except Exception as e:
        print("Error:", e)
        raise
