import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import random

def ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def sma(data, period):
    return data.rolling(window=period).mean()

def calculate_macd(data, fast_length=12, slow_length=26, signal_length=9):
    
    data['EMA_Fast']  = ema(data['Close'], fast_length) # ema_fast
    data['EMA_Slow']  = ema(data['Close'], slow_length) # ema_slow
    data['MACD']      = data['EMA_Fast'] - data['EMA_Slow'] # macd 
    data['Signal']    = sma(data['MACD'], signal_length) # signal
    data['Histogram'] = data['MACD'] - data['Signal']

    return data

# -----------------------------------------------------------------------------

def knn(data, N, K):
    nearest_neighbors = []
    distances = []
    
    for i in range(N-1):
        d = abs(data.iloc[i] - data.iloc[i+1])
        # d = np.linalg.norm(data.iloc[i] - data.iloc[i+1])
        # d = np.sum(np.log(1 + np.abs(data.iloc[i] - data.iloc[i+1])))

        distances.append(d)
        size = len(distances)
        
        # new_neighbor = data.iloc[i+1] if d < np.min(distances[-K:] if size > K else distances) else data.iloc[i]

        if size > K:
            min_distances = np.min(distances[-K:])
        else:
            min_distances = np.min(distances)

        if d < min_distances:
            new_neighbor = data.iloc[i + 1]
        else:
            new_neighbor = data.iloc[i]

        nearest_neighbors.append(new_neighbor)
    
    return nearest_neighbors

# -----------------------------------------------------------------------------

def predict(neighbors, data):
    prediction = np.mean(neighbors)
    # direction = 1 if prediction < data[0] if adjust_prediction else data[1] else -1 if prediction > data[0] if adjust_prediction else data[1] else 0
    # int   direction  = prediction < data[ADJ?0:1] ? 1 : prediction > data[ADJ?0:1] ? -1 : 0

    # if ADJ:
    reference = data.iloc[-1]
    # else:
    #     reference = data.iloc[-2]

    if prediction < reference:
        direction = 1
    elif prediction > reference:
        direction = -1
    else:
        direction = 0

    # direction = 1 if prediction < data[0] if adjust_prediction else data[1] else -1 if prediction > data[0] if adjust_prediction else data[1] else 0

    return prediction, direction

# -----------------------------------------------------------------------------

def zscore_color(data, lagz, direction):
    # data = np.array(data)
    # zs = (data - np.mean(data[-lagz:])) / np.std(data[-lagz:])  # standardize
    zs = (data - np.mean(data.iloc[-lagz:])) / np.std(data.iloc[-lagz:])
    zs1 = zs / (lagz / 5)

    if zs1.iloc[-1] > 0 and direction == 1:
        return 'Buy' # "#006400" # Green
    elif zs1.iloc[-1] < 0 and direction == -1:
        return 'Sell' # "#E00000" # Red
    else:
        return None

# -----------------------------------------------------------------------------

# def plot_macd(macd, signal, histogram, hist_color_change=True, macd_color_change=True):
def generate_signals(data, macd, signal, histogram, N, K, LAGZ):

    # df['Signal_Day'] = np.nan
    # df['Signal_Min'] = np.nan

    histA_IsUp = (histogram > histogram.shift(1)) #&| (histogram > 0)
    histA_IsDown = (histogram < histogram.shift(1)) #| (histogram < 0)
    # histB_IsDown = (histogram < histogram.shift(1)) & (histogram <= 0)
    # histB_IsUp = (histogram > histogram.shift(1)) & (histogram <= 0)

    macd_IsAbove = macd >= signal
    macd_IsBelow = macd < signal

    # data['Signal_Day'] = np.where(macd_IsAbove, 'Buy', np.where(macd_IsBelow, 'Sell', 'Hold'))
    data['Signal_Day'] = np.where((histA_IsUp & macd_IsAbove), 'Buy', np.where((histA_IsDown & macd_IsBelow), 'Sell', 'Hold'))
    data['Signal_Day'] = data['Signal_Day'].shift(1)
    # data['Signal_Day'] = data['Signal_Day'].apply(lambda x: 'Buy' if x else 'Sell')
    data['Signal_Min'] = data['Signal_Day']

    data.dropna(axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # for i in range(N-1, len(data)):

    #     # open_min  = df.loc[i-N-1:i, 'Open_15min']
    #     # high_min  = df.loc[i-N-1:i, 'High_15min']
    #     # low_min   = df.loc[i-N-1:i, 'Low_15min']
    #     # close_min = df.loc[i-N-1:i, 'Close_15min']
    #     # hlc3_min = df.loc[i-(N-1):i, 'Close_15min']
    #     hlc3_min  = data.loc[i-(N-1):i, 'Close']

    #     nn_min = knn(hlc3_min, N, K)
    #     pred_min, dir_min = predict(nn_min, hlc3_min)
    #     data.loc[i, 'Signal_Min'] = zscore_color(hlc3_min, LAGZ, dir_min)
    #     # df.loc[i, 'Signal_Day'] = zscore_color(hlc3_min, LAGZ, dir_min)

    # data['Signal_Min'] = data['Signal_Min'].shift(1)
    # data.dropna(axis=0, inplace=True)
    # data.reset_index(drop=True, inplace=True)

# ---------------------------------------------------------------------------------------

interval        = 15
candles_in_day  = 25
# initial_capital = 170000.
initial_capital = 110000.
# position_size   = 25
position_size   = 50
brokerage_trade = 110.0

# Backtest the strategy

def backtest(df):

    total_profit_loss   = initial_capital
    buy_value           = 0.
    buy_sell_value      = 0.
    sell_value          = 0.
    sell_buy_value      = 0.
    profit_loss         = 0.
    buy_profit_loss     = 0.
    sell_profit_loss    = 0.
    slippage            = 3.0
    trade_count         = 0
    Buy_ON              = False
    Sell_ON             = False
    max_loss            = 100000
    max_profit          = -100000
    max_buy_count       = 0
    max_sell_count      = 0
    max_buy             = 0
    max_sell            = 0
    buy_date_start      = None
    buy_date_end        = None
    sell_date_start     = None
    sell_date_end       = None
    sl_points           = 100
    sl_buy_value        = 0.0
    sl_sell_value       = 0.0

    trade_df = pd.DataFrame(columns=['Signal','DateTime','Price','Profit','Cum Profit'])

    convert_dict = {'Price': float,
                    'Profit': float,
                    'Cum Profit': float,
                    }
 
    trade_df = trade_df.astype(convert_dict)

    j = 0
    
    for i in range(len(df)):

        signalD = df.iloc[i, -2]
        signalM = df.iloc[i, -1]

        if Buy_ON == True:

            prev_low    = df.loc[i-1, 'Low'] #df.iloc[i-1, 2]

            if prev_low <= sl_buy_value :

                buy_sell_value      = sl_buy_value #df.iloc[i, 0] - slippage
                profit_loss         = (buy_sell_value - buy_value) * position_size
                total_profit_loss  += profit_loss
                buy_profit_loss    += profit_loss
                trade_count        += 1
                Buy_ON              = False

                j += 1

                buy_date_end  = df.loc[i, 'date'] #df.iloc[i, 6]
                delta_buy     = buy_date_end - buy_date_start
                max_buy_count = delta_buy.days
                
                if max_buy < max_buy_count:
                    max_buy = max_buy_count

                trade_df.loc[j] = (['Closed Buy', str(df.index[i]), str(buy_sell_value), str(profit_loss), str(total_profit_loss)])

                buy_value       = 0.0
                sl_buy_value    = 0.0
                buy_sell_value  = 0.0
                buy_date_start  = None
                buy_date_end    = None
                max_buy_count   = 0

        if Sell_ON == True:

            prev_high    = df.loc[i-1, 'High'] #df.iloc[i-1, 1]

            if prev_high >= sl_sell_value :
                
                sell_buy_value      = sl_sell_value #df.iloc[i, 0] + slippage
                profit_loss         = (sell_value - sell_buy_value) * position_size
                total_profit_loss  += profit_loss
                sell_profit_loss   += profit_loss
                trade_count        += 1
                Sell_ON             = False

                j += 1

                sell_date_end  = df.loc[i, 'date'] #df.iloc[i, 6]
                delta_sell     = sell_date_end - sell_date_start
                max_sell_count = delta_sell.days

                if max_sell < max_sell_count:                
                    max_sell = max_sell_count

                trade_df.loc[j] = (['Closed Sell', str(df.index[i]), str(sell_buy_value), str(profit_loss), str(total_profit_loss)])

                sell_value      = 0.0
                sl_sell_value   = 0.0
                sell_buy_value  = 0.0
                sell_date_start = None
                sell_date_end   = None
                max_sell_count  = 0


        # if signal == 'Sell' and Buy_ON == True:
        if ((signalD == 'Buy' and signalM == 'Sell')
            or (signalD == 'Sell' and signalM == 'Sell')) and Buy_ON == True:

            # buy_sell_value      = df.iloc[i, 0] * (1 - (slippage / 100.0))
            buy_sell_value      = df.loc[i, 'Open'] - slippage
            profit_loss         = (buy_sell_value - buy_value) * position_size
            total_profit_loss  += profit_loss
            buy_profit_loss    += profit_loss
            trade_count        += 1
            Buy_ON              = False

            j += 1

            buy_date_end  = df.loc[i, 'date'] #df.iloc[i, 6]
            delta_buy     = buy_date_end - buy_date_start
            max_buy_count = delta_buy.days
            
            if max_buy < max_buy_count:
                max_buy = max_buy_count

            trade_df.loc[j] = (['Closed Buy', str(df.index[i]), str(buy_sell_value), str(profit_loss), str(total_profit_loss)])

            buy_value       = 0.0
            buy_sell_value  = 0.0
            buy_date_start  = None
            buy_date_end    = None
            max_buy_count   = 0


        # if signal == 'Buy' and Sell_ON == True:
        if ((signalD == 'Sell' and signalM == 'Buy') 
           or (signalD == 'Buy' and signalM == 'Buy')) and Sell_ON == True:

            sell_buy_value      = df.loc[i, 'Open'] + slippage
            # sell_buy_value      = df.iloc[i, 0] * (1 + (slippage / 100.0))
            profit_loss         = (sell_value - sell_buy_value) * position_size
            total_profit_loss  += profit_loss
            sell_profit_loss   += profit_loss
            trade_count        += 1
            Sell_ON             = False

            j += 1

            sell_date_end  = df.loc[i, 'date'] #df.iloc[i, 6]
            delta_sell     = sell_date_end - sell_date_start
            max_sell_count = delta_sell.days

            if max_sell < max_sell_count:                
                max_sell = max_sell_count

            trade_df.loc[j] = (['Closed Sell', str(df.index[i]), str(sell_buy_value), str(profit_loss), str(total_profit_loss)])

            sell_value      = 0.0
            sell_buy_value  = 0.0
            sell_date_start  = None
            sell_date_end    = None
            max_sell_count  = 0


        # if signal == 'Buy' and Buy_ON == False:
        if signalD == 'Buy' and signalM == 'Buy' and Buy_ON == False:
            
            buy_value       = df.loc[i, 'Open'] + slippage
            sl_buy_value    = buy_value - sl_points
            # buy_value       = df.iloc[i, 0] * (1 + (slippage / 100.0))
            trade_count    += 1
            Buy_ON          = True
            buy_date_start  = df.loc[i, 'date'] #df.iloc[i, 6]
            
            j += 1

            trade_df.loc[j] = (['Buy', str(df.index[i]), str(buy_value), '', ''])

        
        # if signal == 'Sell' and Sell_ON == False:
        if signalD == 'Sell' and signalM == 'Sell' and Sell_ON == False:

            sell_value       = df.loc[i, 'Open'] - slippage
            sl_sell_value    = sell_value + sl_points
            # sell_value      = df.iloc[i, 0] * (1 - (slippage / 100.0))
            trade_count    += 1
            Sell_ON         = True
            sell_date_start = df.loc[i, 'date']

            j += 1

            trade_df.loc[j] = (['Sell', str(df.index[i]), str(sell_value), '', ''])


        if profit_loss and profit_loss > max_profit:
            max_profit = profit_loss

        if profit_loss and profit_loss < max_loss:
            max_loss = profit_loss


    return total_profit_loss, trade_count, max_profit, max_loss, trade_df,  max_buy, max_sell, buy_profit_loss, sell_profit_loss

# ---------------------------------------------------------------------------------------

def MDD(trade_df):
    
    DD_df = pd.DataFrame()
    DD_df = trade_df.copy()
    DD_df.replace('', np.nan, inplace=True)
    DD_df = DD_df.dropna()

    max_values = DD_df['Cum Profit'].rolling(window=len(DD_df), min_periods = 1).max()
    DD_values =  DD_df['Cum Profit'].astype(float) / max_values - 1
    MDD_values = DD_values.rolling(window=len(DD_df), min_periods = 1).min()
    DD = MDD_values.min() * 100    

    return DD

# ---------------------------------------------------------------------------------------

# Load CSV data (replace 'data.csv' with your CSV file path)
# df = pd.read_csv(r'D:\AlgoTrade\Fyers_AlgoTrade\Nifty50_Index\NIFTY50_INDEX_60_Min.csv')
df = pd.read_csv(r'D:\AlgoTrade\Fyers_AlgoTrade\Nifty50_Index\NIFTY50_INDEX_15_Min.csv')
# Assuming the 'close' column contains the closing prices
# close_data = df['close']

# df = df[5909:] # 60 Mins
df = df[21391:] # 15 Mins
# print(df.head())

# ---------------------------------------------------------------------------------------

# df1['HLC3'] = (df1['High'] + df1['Low'] + df1['Close']) / 3.0
# df1['HLC3'] = df1['HLC3'].shift(1)
df['Date'] = pd.to_datetime(df['Date'])
df['time'] = df['Date'].dt.time
df['date'] = df['Date'].dt.date
df = df[df['time'] != dt.time(9,00,00)]
df.drop(['Unnamed: 0', 'Volume'], inplace=True, axis=1)
# print(df1.head())

i = 0
max_pnl = 0.0   

while i <= 0:

    N       = random.randrange(3, 10) #13        # # of Data Points [2:n]
    K       = 100 #random.randrange(50, 30) #100       # # of Nearest Neighbors (K) [1:252]
    LAGZ    = random.randrange(3, 12) #10        # Z-Score Lag [2:n] if selected

    fast_length= 17 #random.randrange(5, 10) #17
    slow_length= 38 #random.randrange(20, 30) #38
    signal_length= 26 #random.randrange(7, 12) #26

    # Calculate MACD
    df = calculate_macd(df, fast_length=fast_length, slow_length=slow_length, signal_length=signal_length)

    # df['Signal_Day'] = np.nan
    # df['Signal_Min'] = np.nan

    generate_signals(df, df['MACD'], df['Signal'], df['Histogram'], N, K, LAGZ)

    # print(df.head(5))
    # print(df.tail(5))

    # print('Done')

    df.dropna(axis=0, inplace=True)
    # ohlcv_data.drop('index', inplace=True, axis=1) 
    df.reset_index(drop=True, inplace=True)

    df.to_csv('macd.csv')

    profit_loss, trade_count, max_profit, max_loss, trade_df, max_buydays, max_selldays, buy_pnl, sell_pnl = backtest(df)

    # trade_df.to_csv('EMA_Day_' + str(EMA_period_day) + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins'+ '.csv')
    # trade_df.to_csv(Supertrend_name1 + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins'+ '.csv')

    pct_Pnl = round((((profit_loss - (trade_count * brokerage_trade) - initial_capital) / initial_capital) * 100.), 2)
    print(pct_Pnl)

    if pct_Pnl > max_pnl :
        
        max_pnl = pct_Pnl

        # df.to_csv('EMA_Day_' + str(EMA_period_day) + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins_Signals'+ '.csv')
        # df.to_csv(Supertrend_name1 + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins_Signals'+ '.csv')
                
        # print('Supertrends      :', str(Supertrend_ATR1) + '_' + str(Supertrend_Fact1))
        # print('LAGZ             :', str(LAGZ))
        # print('Window Period    :', str(window_period))
        # print('Hour EMAs        :', str(Supertrend_name1))
        # print('Parameters       :', str(N) + '_' + str(K) + '_' + str(LAGZ))
        print('Parameters       :', str(fast_length) + '_' + str(slow_length) + '_' + str(signal_length))
        print('Initial Capital  :', round(initial_capital, 2))
        print('Total Profit/Loss:', round(profit_loss-initial_capital, 2))
        print('Buy Profit/Loss  :', round(buy_pnl, 2))
        print('Sell Profit/Loss :', round(sell_pnl, 2))
        print('Number of Trades :', trade_count)
        print('Brokerage        :', (trade_count * brokerage_trade))
        print('Net Profit/Loss  :', round((profit_loss - initial_capital - (trade_count * brokerage_trade)), 2))
        print('Profit/Loss %    :', str(round((((profit_loss - (trade_count * brokerage_trade) - initial_capital) / initial_capital) * 100.), 2)) + '%')
        print('Buy PnL %        :', str(round(((buy_pnl / initial_capital) * 100.), 2)) + '%')
        print('Sell PnL %       :', str(round(((sell_pnl  / initial_capital) * 100.), 2)) + '%')
        print('Max Profit       :', round((max_profit - brokerage_trade), 2))
        print('Max Loss         :', round((max_loss - brokerage_trade), 2))
        print('Max Drawdown     :', str(round(MDD(trade_df), 2)) + '%')
        print('Max Buy Days     :', str(max_buydays + 1) + ' Days')
        print('Max Sell Days    :', str(max_selldays + 1 ) + ' Days')

        # trade_df.to_csv(Supertrend_name1 + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins' + '.csv')
        # trade_df.to_csv('EMA_Day_' + str(EMA_period_day) + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins' + '.csv')

    i += 1

    # Plot MACD
    # plot_macd(macd, signal, histogram, hist_color_change=True, macd_color_change=True)


# 359.94
# Parameters       : 17_100_9
# Parameters       : 17_38_26
# Initial Capital  : 110000.0
# Total Profit/Loss: 486465.0
# Buy Profit/Loss  : 328750.0
# Sell Profit/Loss : 157715.0
# Number of Trades : 823
# Brokerage        : 90530.0
# Net Profit/Loss  : 395935.0
# Profit/Loss %    : 359.94%
# Buy PnL %        : 298.86%
# Sell PnL %       : 143.38%
# Max Profit       : 50350.0
# Max Loss         : -8760.0
# Max Drawdown     : -20.36%
# Max Buy Days     : 8 Days
# Max Sell Days    : 9 Days

# 1 Hour

# 445.96
# Parameters       : 9_24_8
# Initial Capital  : 110000.0
# Total Profit/Loss: 585705.0
# Buy Profit/Loss  : 444985.0
# Sell Profit/Loss : 140720.0
# Number of Trades : 865
# Brokerage        : 95150.0
# Net Profit/Loss  : 490555.0
# Profit/Loss %    : 445.96%
# Buy PnL %        : 404.53%
# Sell PnL %       : 127.93%
# Max Profit       : 51900.0
# Max Loss         : -7100.0
# Max Drawdown     : -22.63%
# Max Buy Days     : 8 Days
# Max Sell Days    : 9 Days
