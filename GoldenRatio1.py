# Golden Ration with 1 Minute Data
# ---------------------------------------------

import pandas as pd
import numpy as np
import glob
import os

# Load the OHLC data into a Pandas DataFrame
path = r'D:\AlgoTrade\Fyers_AlgoTrade\BankNifty_Index'

csv_files = glob.glob(path + "/*.csv")

# This creates a list of dataframes
df_list = (pd.read_csv(file) for file in csv_files)

for f in csv_files:
    
    # read the csv file
    data = pd.read_csv(f)

    filename = os.path.basename(f)
    name, extension = os.path.splitext(filename)

    # data = pd.read_csv('stock_data.csv')

    # Calculate the high-low range for the last 30 periods
    data['HL_Range'] = data['High'].rolling(window=20).max() - data['Low'].rolling(window=20).min()

    # Calculate the midpoint of the high-low range
    data['HL_Midpoint'] = (data['High'].rolling(window=30).max() 
                           + data['Low'].rolling(window=30).min()) / 2

    # Calculate the Golden Ratio levels
    data['Level_0'] = data['HL_Midpoint'] - 0.618 * data['HL_Range']
    data['Level_1'] = data['HL_Midpoint'] - 0.382 * data['HL_Range']
    data['Level_2'] = data['HL_Midpoint'] + 0.382 * data['HL_Range']
    data['Level_3'] = data['HL_Midpoint'] + 0.618 * data['HL_Range']

    # Determine the current position based on the previous period's position and the current price
    data['Position'] = 0
    data['Position'] = np.where(data['Close'] > data['Level_2'], 1, 
                       np.where(data['Close'] < data['Level_1'], -1, 
                       data['Position'].shift()))

    # Calculate the daily returns and the cumulative returns
    data['Returns'] = data['Close'].pct_change() * data['Position']
    data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()

    # Calculate the drawdowns
    data['Peak'] = data['Cumulative_Returns'].cummax()
    data['Drawdown'] = data['Cumulative_Returns'] / data['Peak'] - 1

    # Calculate the financial metrics
    returns = data['Returns']
    sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
    sortino_ratio = np.sqrt(252) * (returns.mean() / returns[returns < 0].std())
    drawdown = data['Drawdown'].min()
    winrate = (data['Returns'] > 0).sum() / len(data)

    print('----------------------------------')
    print(filename)
    print('----------------------------------')
    # Print the financial metrics
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Maximum Drawdown: {drawdown:.2%}")
    print(f"Win Rate: {winrate:.2%}")

    data.to_csv(name + '_Golden_Ratio1.csv')
