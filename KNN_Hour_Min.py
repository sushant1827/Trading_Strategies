import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta
import datetime as dt
import math
import random

pd.set_option('mode.chained_assignment', None)

# Inputs
# TF = '5'          # Resolution
# N       = 10        # # of Data Points [2:n]
# K       = 126       # # of Nearest Neighbors (K) [1:252]
ADJ     = True      # Adjust Prediction
REP     = False     # Non-Repainting
ADDON   = 'Z-Score' # Add-On
LAGP    = 5         # Pivot Point Lag [2:n] if selected
# LAGZ    = 10        # Z-Score Lag [2:n] if selected
# DISP = 'Both'     # Show Outcomes
# ODS = hlcc4       # Projection Base - Assuming hlcc4 is a valid data source

# ticker   = 'NIFTY50'
# ticker   = 'NIFTYBANK'
# type1    = 'INDEX'
interval        = 15
candles_in_day  = 25
# initial_capital = 170000.
initial_capital = 110000.
# position_size   = 25
position_size   = 50
brokerage_trade = 110.0

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

    if ADJ:
        reference = data.iloc[-1]
    else:
        reference = data.iloc[-2]

    if prediction < reference:
        direction = 1
    elif prediction > reference:
        direction = -1
    else:
        direction = 0

    # direction = 1 if prediction < data[0] if adjust_prediction else data[1] else -1 if prediction > data[0] if adjust_prediction else data[1] else 0

    return prediction, direction

# -----------------------------------------------------------------------------

def ordinary_color(direction):
    if direction == 1:
        return 'Buy' #"#006400" # Green
    elif direction == -1:
        return 'Sell' #"#E00000" # Red
    else:
        return None
    
# -----------------------------------------------------------------------------

def pivot_color(high, low, direction):
    ph = high[-LAGP] if np.argmax(high) == 0 else None
    pl = low[-LAGP] if np.argmin(low) == 0 else None
    if ph and direction == 1:
        return 'Buy' #"#006400" # Green
    elif pl and direction == -1:
        return 'Sell' #"#E00000" # Red
    else:
        return None
    
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

def generate_signals(df):
    
    for i in range(N-1, len(df)):

        # open_min  = df.loc[i-N-1:i, 'Open_15min']
        # high_min  = df.loc[i-N-1:i, 'High_15min']
        # low_min   = df.loc[i-N-1:i, 'Low_15min']
        # close_min = df.loc[i-N-1:i, 'Close_15min']
        # hlc3_min = df.loc[i-(N-1):i, 'Close_15min']
        hlc3_min  = df.loc[i-(N-1):i, 'HLC3_EMA_Min']

        nn_min = knn(hlc3_min, N, K)
        pred_min, dir_min = predict(nn_min, hlc3_min)
        df.loc[i, 'Signal_Min'] = zscore_color(hlc3_min, LAGZ, dir_min)
        # df.loc[i, 'Signal_Day'] = zscore_color(hlc3_min, LAGZ, dir_min)

        # if df['Close_15min'][i] > df['ST1'][i]: df['Signal_Day'][i] = 'Buy'
        # if df['Close_15min'][i] < df['ST1'][i]: df['Signal_Day'][i] = 'Sell'

        # ---------------------------------------------------------------

        hlc3_day  = df.loc[i-(N-1):i, 'HLC3_EMA']

        nn_day = knn(hlc3_day, N, K)
        pred_day, dir_day = predict(nn_day, hlc3_day)
        df.loc[i, 'Signal_Day'] = zscore_color(hlc3_day, LAGZ, dir_day)

        
        # if df['EMA_Fast_Day'][i-1] > df['EMA_Slow_Day'][i-1]: df['Signal_Day'][i] = 'Buy'
        # if df['EMA_Fast_Day'][i-1] < df['EMA_Slow_Day'][i-1]: df['Signal_Day'][i] = 'Sell'
            
        # if df['EMA_Fast_Min'][i-1] > df['EMA_Slow_Min'][i-1]: df['Signal_Min'][i] = 'Buy'
        # if df['EMA_Fast_Min'][i-1] < df['EMA_Slow_Min'][i-1]: df['Signal_Min'][i] = 'Sell'

    # df['Signal_Day'].fillna(method='ffill', inplace=True)
    # df['Signal_Min'].fillna(method='ffill', inplace=True)

    # df['Signal'] = df.apply(get_third_column_value, axis=1)    

# ---------------------------------------------------------------------------------------

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

            prev_low    = df.iloc[i-1, 2]

            if prev_low <= sl_buy_value :

                buy_sell_value      = sl_buy_value #df.iloc[i, 0] - slippage
                profit_loss         = (buy_sell_value - buy_value) * position_size
                total_profit_loss  += profit_loss
                buy_profit_loss    += profit_loss
                trade_count        += 1
                Buy_ON              = False

                j += 1

                buy_date_end  = df.iloc[i, 6]
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

            prev_high    = df.iloc[i-1, 1]

            if prev_high >= sl_sell_value :
                
                sell_buy_value      = sl_sell_value #df.iloc[i, 0] + slippage
                profit_loss         = (sell_value - sell_buy_value) * position_size
                total_profit_loss  += profit_loss
                sell_profit_loss   += profit_loss
                trade_count        += 1
                Sell_ON             = False

                j += 1

                sell_date_end  = df.iloc[i, 6]
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
            buy_sell_value      = df.iloc[i, 0] - slippage
            profit_loss         = (buy_sell_value - buy_value) * position_size
            total_profit_loss  += profit_loss
            buy_profit_loss    += profit_loss
            trade_count        += 1
            Buy_ON              = False

            j += 1

            buy_date_end  = df.iloc[i, 6]
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

            sell_buy_value      = df.iloc[i, 0] + slippage
            # sell_buy_value      = df.iloc[i, 0] * (1 + (slippage / 100.0))
            profit_loss         = (sell_value - sell_buy_value) * position_size
            total_profit_loss  += profit_loss
            sell_profit_loss   += profit_loss
            trade_count        += 1
            Sell_ON             = False

            j += 1

            sell_date_end  = df.iloc[i, 6]
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
            
            buy_value       = df.iloc[i, 0] + slippage
            sl_buy_value    = buy_value - sl_points
            # buy_value       = df.iloc[i, 0] * (1 + (slippage / 100.0))
            trade_count    += 1
            Buy_ON          = True
            buy_date_start  = df.iloc[i, 6]
            
            j += 1

            trade_df.loc[j] = (['Buy', str(df.index[i]), str(buy_value), '', ''])

        
        # if signal == 'Sell' and Sell_ON == False:
        if signalD == 'Sell' and signalM == 'Sell' and Sell_ON == False:

            sell_value       = df.iloc[i, 0] - slippage
            sl_sell_value    = sell_value + sl_points
            # sell_value      = df.iloc[i, 0] * (1 - (slippage / 100.0))
            trade_count    += 1
            Sell_ON         = True
            sell_date_start = df.iloc[i, 6]            

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

# def find_lowest(a, b, c, a1, b1, c1):    
#     lst = []
#     if a < b or a < c:
#         lst.append(a1)
#         # return a, a1
    
#     if b < a or b < c:
#         lst.append(b1)
#         # return b, b1
    
#     if c < a or c < b:
#         lst.append(c1)
#         # return c, c1
#     # else:
    
#     return 0, lst

# def calculate_supertrend_15min(high_60min, low_60min, close_60min, multiplier_15min, length_15min):
#     # Step 1: Calculate the ATR for the 15-minute timeframe using the 60-minute data
#     atr_15min = ta.ATR(high_60min, low_60min, close_60min, timeperiod=length_15min)

#     # Step 2: Calculate the Basic Upper and Basic Lower values for 15-minute Supertrend
#     basic_upper = (close_60min.values[-1] + (multiplier_15min * atr_15min.values[-1]))
#     basic_lower = (close_60min.values[-1] - (multiplier_15min * atr_15min.values[-1]))

#     # Step 3: Calculate the 15-minute Supertrend value based on the previous 15-minute Supertrend value
#     # and the 15-minute close price
#     previous_supertrend_15min = close_60min.values[-1]  # Use the 60-minute close as the starting value
#     today_close_15min = close_60min.values[-1]  # Replace this with the actual 15-minute close value for the current day

#     if previous_supertrend_15min == basic_lower:
#         today_supertrend_15min = max(basic_upper, today_close_15min)
#     elif previous_supertrend_15min == basic_upper:
#         today_supertrend_15min = min(basic_lower, today_close_15min)
#     else:
#         today_supertrend_15min = 0  # No valid Supertrend value (usually the first day with no previous Supertrend)

#     return today_supertrend_15min


# def get_curr_ST(df, in_high_col_min, in_low_col_min, in_close_col_min, 
#                     in_high_col_day, in_low_col_day, in_close_col_day, 
#                     Supertrend_ATR1, Supertrend_Fact1, Supertrend_name1, 
#                     out_col_name):

#     i = 0

#     while i < len(df):

#         day_high_unique_values  = df.loc[df.index <= i, in_high_col_day].drop_duplicates().tolist()
#         day_low_unique_values   = df.loc[df.index <= i, in_low_col_day].drop_duplicates().tolist()
#         day_close_unique_values = df.loc[df.index <= i, in_close_col_day].drop_duplicates().tolist()

#         if len(day_high_unique_values) >= Supertrend_ATR1*2:

#             high_series     = pd.Series(day_high_unique_values[-Supertrend_ATR1*2:])
#             low_series      = pd.Series(day_low_unique_values[-Supertrend_ATR1*2:])
#             close_series    = pd.Series(day_close_unique_values[-Supertrend_ATR1*2:])

#             supertrend_15min_value = calculate_supertrend_15min(high_series, low_series, close_series, 
#                                                                 Supertrend_ATR1, Supertrend_Fact1)

#             df[out_col_name][i] = supertrend_15min_value

            # day_high_unique_values[-1]  = df[in_high_col_min][i]
            # day_low_unique_values[-1]   = df[in_low_col_min][i]
            # day_close_unique_values[-1] = df[in_close_col_min][i]

            # high_series     = pd.Series(day_high_unique_values)
            # low_series      = pd.Series(day_low_unique_values)
            # close_series    = pd.Series(day_close_unique_values)

            # # val = 1
            # col = ['1']

            # while len(col):
            
            #     if 'high' in col:
            #         # val1 = val
            #         # while val1 < len(day_high_unique_values)
            #         new_value = high_series.iloc[0] + 0.1
            #         high_series = pd.concat([pd.Series([new_value]), high_series])
            #         # high_series = pd.concat([high_series.iloc[[0]], high_series.shift().iloc[1:]])
            #         high_series = high_series.reset_index(drop=True)
            #     if 'low' in col: #col == 'low'  : 
            #         new_value = low_series.iloc[0] + 0.1
            #         low_series = pd.concat([pd.Series([new_value]), low_series])
            #         # low_series = pd.concat([low_series.iloc[[0]], low_series.shift().iloc[1:]])
            #         low_series = low_series.reset_index(drop=True)
            #     if 'close' in col: #col == 'close': 
            #         new_value = close_series.iloc[0] + 0.1
            #         close_series = pd.concat([pd.Series([new_value]), close_series])
            #         # close_series = pd.concat([close_series.iloc[[0]], close_series.shift().iloc[1:]])
            #         close_series = close_series.reset_index(drop=True)

            #     val, col = find_lowest(len(high_series), len(low_series), len(close_series),
            #                             'high', 'low', 'close') 
                
            # # print(len(high_series))
            # # print(len(low_series))
            # # print(len(close_series))

            # value = pta.supertrend(high_series, low_series, close_series, 
            #                        length=Supertrend_ATR1, multiplier=Supertrend_Fact1)[Supertrend_name1]    

            # # value = pta.ema(series, period) 
            # # mean_value = pta.ema(series, period).mean()
            # # value1 = value / mean_value
            # # value = pta.ema(day_unique_values, length=period)[-1]
            # df[out_col_name][i] = value.iloc[-1]
            # # print(value.iloc[-1])

        # i += 1

# ---------------------------------------------------------------------------------------

def get_curr_EMA(df, in_col_name1, in_col_name2, out_col_name, period):

    # df, 'HLC3_day', 'HLC3_15min', 'HLC3_EMA', EMA_period_day

    i = 0
    while i < len(df):

        day_unique_values = df.loc[df.index <= i, in_col_name1].drop_duplicates().tolist()        

        if len(day_unique_values) >= period*2:
            # day_unique_values.append(df['Close_15min'][i])
            day_unique_values[-1] = df[in_col_name2][i]
            # day_unique_values.append(df[in_col_name2][i])
            series = pd.Series(day_unique_values)
            value = pta.ema(series, period) 
            mean_value = pta.ema(series, period).mean()
            value1 = value / mean_value
            # value = pta.ema(day_unique_values, length=period)[-1]
            df[out_col_name][i] = value1.iloc[-1]
            # print(value.iloc[-1])

        i += 1

# ---------------------------------------------------------------------------------------

# Read the OHLC data from CSV file

# df = pd.read_csv('NIFTY50_INDEX_15_Min.csv')
# df1 = pd.read_csv(r'D:\AlgoTrade\Fyers_AlgoTrade\BankNifty_Index\NIFTYBANK_INDEX_15_Min.csv')
df1 = pd.read_csv(r'D:\AlgoTrade\Fyers_AlgoTrade\Nifty50_Index\NIFTY50_INDEX_15_Min.csv')

df1 = df1[21391:] # 15 Mins
# print(df1.head())

# ---------------------------------------------------------------------------------------

df1['HLC3'] = (df1['High'] + df1['Low'] + df1['Close']) / 3.0
# df1['HLC3'] = df1['HLC3'].shift(1)
df1['Date'] = pd.to_datetime(df1['Date'])
df1['time'] = df1['Date'].dt.time
df1['date'] = df1['Date'].dt.date
df1 = df1[df1['time'] != dt.time(9,00,00)]
df1.drop(['Unnamed: 0', 'Volume'], inplace=True, axis=1)
# print(df1.head())

# ---------------------------------------------------------------------------------------

day_df = pd.read_csv(r'D:\AlgoTrade\Fyers_AlgoTrade\Nifty50_Index\NIFTY50_INDEX_60_Min.csv')

day_df = day_df[5951:] # 60 Mins

day_df['HLC3'] = (day_df['High'] + day_df['Low'] + day_df['Close']) / 3.0
# day_df['HLC3'] = day_df['HLC3'].shift(1)
day_df['Date'] = pd.to_datetime(day_df['Date'])
day_df['date'] = day_df['Date'].dt.date
day_df['time'] = day_df['Date'].dt.time
day_df = day_df[day_df['time'] != dt.time(8,15,00)]

day_df.drop(['Unnamed: 0', 'Volume'], inplace=True, axis=1)
# print(day_df.head())

# ---------------------------------------------------------------------------------------

# def get_third_column_value(row):
#     if row['Signal_Day'] == 'Buy' and row['Signal_Min'] == 'Buy':
#         return 'Buy'
#     elif row['Signal_Day'] == 'Buy' and row['Signal_Min'] == 'Sell':
#         return 'Sell'
#     elif row['Signal_Day'] == 'Sell' and row['Signal_Min'] == 'Sell':
#         return 'Sell'
#     elif row['Signal_Day'] == 'Sell' and row['Signal_Min'] == 'Buy':
#         return 'Buy'

# ---------------------------------------------------------------------------------------

i = 0
max_pnl = 0.0

while i <= 0:

    N       = 13 #random.randrange(3, 30) #10        # # of Data Points [2:n]
    K       = 100 #random.randrange(50, 30) #126       # # of Nearest Neighbors (K) [1:252]
    LAGZ    = 10 #random.randrange(5, 20) #10        # Z-Score Lag [2:n] if selected

    print(i)

    EMA_period_min = 17 #random.randrange(5, 30)

    df1['HLC3_EMA_Min'] = ta.EMA(df1['HLC3'], EMA_period_min) / ta.EMA(df1['HLC3'], EMA_period_min).mean()
    df1.dropna(axis=0, inplace=True)

    # Supertrend_ATR1  = random.randrange(2, 20)
    # Supertrend_Fact1 = np.random.choice(np.arange(1, 5.25, 0.5), size=1)[0] #float(random.randrange(1, 5))
    # Supertrend_name1 = 'SUPERT_' + str(Supertrend_ATR1) + '_' + str(Supertrend_Fact1) #+ '.0'

    # df['ST1'] = pta.supertrend(df['HA_High'], df['HA_Low'], df['HA_Close'], length=Supertrend_ATR1, multiplier=Supertrend_Fact1)[Supertrend_name1]
    # day_df['ST1'] = pta.supertrend(day_df['High'], day_df['Low'], day_df['Close'], length=Supertrend_ATR1, multiplier=Supertrend_Fact1)[Supertrend_name1]
    # df1['ST1'] = pta.supertrend(df1['High'], df1['Low'], df1['Close'], length=Supertrend_ATR1, multiplier=Supertrend_Fact1)[Supertrend_name1]
    
    # df = df[1:]

    EMA_period_day = EMA_period_min #random.randrange(5, 30)
    # day_df['HLC3_EMA'] = ta.EMA(day_df['HLC3'], EMA_period_day) / ta.EMA(day_df['HLC3'], EMA_period_day).mean()
    day_df['HLC3_EMA'] = 0.0
    
    #--------------------------------------------------------------------------
    
    df = pd.merge_asof(df1, day_df, on='Date', direction='backward', suffixes=('_15min', '_day'))
    df.dropna(axis=0, inplace=True) 
    df.reset_index(inplace=True)
    # df.set_index('Date', inplace=True)
    # print(df.head(10))

    get_curr_EMA(df, 'HLC3_day', 'HLC3_15min', 'HLC3_EMA', EMA_period_day)
    # get_curr_EMA(df, 'EMA_Slow_Day', slow_EMA_period_day)

    # df['ST1'] = 0.0

    df.to_csv('111.csv')

    # get_curr_ST(df, 'High_15min', 'Low_15min', 'Close_15min', 
    #                 'High_day', 'Low_day', 'Close_day',
    #                 Supertrend_ATR1, Supertrend_Fact1, Supertrend_name1, 'ST1')


    df['Signal_Day'] = np.nan
    df['Signal_Min'] = np.nan

    generate_signals(df)

    df['Signal_Day'] = df['Signal_Day'].shift(1)
    df['Signal_Min'] = df['Signal_Min'].shift(1)

    df.dropna(axis=0, inplace=True)
    df.drop('index', inplace=True, axis=1) 
    df.set_index('Date', inplace=True)

    df.to_csv('222.csv')
    # print(df.head(20))

    profit_loss, trade_count, max_profit, max_loss, trade_df, max_buydays, max_selldays, buy_pnl, sell_pnl = backtest(df)

    trade_df.to_csv('EMA_Day_' + str(EMA_period_day) + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins'+ '.csv')
    # trade_df.to_csv(Supertrend_name1 + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins'+ '.csv')

    pct_Pnl = round((((profit_loss - (trade_count * brokerage_trade) - initial_capital) / initial_capital) * 100.), 2)
    print(pct_Pnl)

    if pct_Pnl > max_pnl :
        
        max_pnl = pct_Pnl

        df.to_csv('EMA_Day_' + str(EMA_period_day) + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins_Signals'+ '.csv')
        # df.to_csv(Supertrend_name1 + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins_Signals'+ '.csv')
                
        # print('Supertrends      :', str(Supertrend_ATR1) + '_' + str(Supertrend_Fact1))
        # print('LAGZ             :', str(LAGZ))
        print('Hour EMAs        :', str(EMA_period_day))
        # print('Hour EMAs        :', str(Supertrend_name1))
        print('Minute EMAs      :', str(EMA_period_min))
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
        trade_df.to_csv('EMA_Day_' + str(EMA_period_day) + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins' + '.csv')

    i += 1

# 322.12
# Hour EMAs        : 10
# Minute EMAs      : 15
# Initial Capital  : 110000.0
# Total Profit/Loss: 497000.0
# Buy Profit/Loss  : 354675.0
# Sell Profit/Loss : 142325.0
# Number of Trades : 1297
# Brokerage        : 142670.0
# Net Profit/Loss  : 354330.0
# Profit/Loss %    : 322.12%
# Buy PnL %        : 322.43%
# Sell PnL %       : 129.39%
# Max Profit       : 44905.0
# Max Loss         : -9500.0
# Max Drawdown     : -13.18%
# Max Buy Days     : 8 Days
# Max Sell Days    : 9 Days

# 323.08
# Hour EMAs        : 16
# Minute EMAs      : 14
# Initial Capital  : 110000.0
# Total Profit/Loss: 487935.0
# Buy Profit/Loss  : 374055.0
# Sell Profit/Loss : 113880.0
# Number of Trades : 1205
# Brokerage        : 132550.0
# Net Profit/Loss  : 355385.0
# Profit/Loss %    : 323.08%
# Buy PnL %        : 340.05%
# Sell PnL %       : 103.53%
# Max Profit       : 44905.0
# Max Loss         : -8685.0
# Max Drawdown     : -13.82%
# Max Buy Days     : 8 Days
# Max Sell Days    : 9 Days

# 333.96
# Hour EMAs        : 18
# Minute EMAs      : 18
# Initial Capital  : 110000.0
# Total Profit/Loss: 488030.0
# Buy Profit/Loss  : 347530.0
# Sell Profit/Loss : 140500.0
# Number of Trades : 1097
# Brokerage        : 120670.0
# Net Profit/Loss  : 367360.0
# Profit/Loss %    : 333.96%
# Buy PnL %        : 315.94%
# Sell PnL %       : 127.73%
# Max Profit       : 53025.0
# Max Loss         : -8995.0
# Max Drawdown     : -19.57%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days

# 370.6
# LAGZ             : 10
# Hour EMAs        : 17
# Minute EMAs      : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 516670.0
# Buy Profit/Loss  : 389715.0
# Sell Profit/Loss : 126955.0
# Number of Trades : 991
# Brokerage        : 109010.0
# Net Profit/Loss  : 407660.0
# Profit/Loss %    : 370.6%
# Buy PnL %        : 354.29%
# Sell PnL %       : 115.41%
# Max Profit       : 53025.0
# Max Loss         : -8995.0
# Max Drawdown     : -24.97%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days

# 417.86
# Hour EMAs        : SUPERT_9_1.0
# Minute EMAs      : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 583395.0
# Buy Profit/Loss  : 386185.0
# Sell Profit/Loss : 197210.0
# Number of Trades : 1125
# Brokerage        : 123750.0
# Net Profit/Loss  : 459645.0
# Profit/Loss %    : 417.86%
# Buy PnL %        : 351.08%
# Sell PnL %       : 179.28%
# Max Profit       : 53025.0
# Max Loss         : -8995.0
# Max Drawdown     : -17.85%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days
# 1
# 426.16
# Hour EMAs        : SUPERT_9_1.0
# Minute EMAs      : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 591870.0
# Buy Profit/Loss  : 391150.0
# Sell Profit/Loss : 200720.0
# Number of Trades : 1119
# Brokerage        : 123090.0
# Net Profit/Loss  : 468780.0
# Profit/Loss %    : 426.16%
# Buy PnL %        : 355.59%
# Sell PnL %       : 182.47%
# Max Profit       : 53025.0
# Max Loss         : -8995.0
# Max Drawdown     : -17.26%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days
# 2
# 460.0
# Hour EMAs        : SUPERT_17_2.0
# Minute EMAs      : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 608850.0
# Buy Profit/Loss  : 430250.0
# Sell Profit/Loss : 178600.0
# Number of Trades : 935
# Brokerage        : 102850.0
# Net Profit/Loss  : 506000.0
# Profit/Loss %    : 460.0%
# Buy PnL %        : 391.14%
# Sell PnL %       : 162.36%
# Max Profit       : 53025.0
# Max Loss         : -9345.0
# Max Drawdown     : -16.93%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days
# 3
# 400.42
# 4
# 474.33
# Hour EMAs        : SUPERT_9_2.0
# Minute EMAs      : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 625050.0
# Buy Profit/Loss  : 426655.0
# Sell Profit/Loss : 198395.0
# Number of Trades : 939
# Brokerage        : 103290.0
# Net Profit/Loss  : 521760.0
# Profit/Loss %    : 474.33%
# Buy PnL %        : 387.87%
# Sell PnL %       : 180.36%
# Max Profit       : 53025.0
# Max Loss         : -9345.0
# Max Drawdown     : -13.78%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days


# 309.85
# Hour EMAs        : SUPERT_14_2.0
# Minute EMAs      : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 470750.0
# Buy Profit/Loss  : 348000.0
# Sell Profit/Loss : 122750.0
# Number of Trades : 1181
# Brokerage        : 129910.0
# Net Profit/Loss  : 340840.0
# Profit/Loss %    : 309.85%
# Buy PnL %        : 316.36%
# Sell PnL %       : 111.59%
# Max Profit       : 53025.0
# Max Loss         : -8985.0
# Max Drawdown     : -27.4%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days

# 326.42
# Hour EMAs        : SUPERT_19_1.0
# Minute EMAs      : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 490735.0
# Buy Profit/Loss  : 357930.0
# Sell Profit/Loss : 132805.0
# Number of Trades : 1197
# Brokerage        : 131670.0
# Net Profit/Loss  : 359065.0
# Profit/Loss %    : 326.42%
# Buy PnL %        : 325.39%
# Sell PnL %       : 120.73%
# Max Profit       : 53025.0
# Max Loss         : -10745.0
# Max Drawdown     : -21.62%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days

# 328.25
# Hour EMAs        : SUPERT_11_2.5
# Minute EMAs      : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 485700.0
# Buy Profit/Loss  : 332390.0
# Sell Profit/Loss : 153310.0
# Number of Trades : 1133
# Brokerage        : 124630.0
# Net Profit/Loss  : 361070.0
# Profit/Loss %    : 328.25%
# Buy PnL %        : 302.17%
# Sell PnL %       : 139.37%
# Max Profit       : 53025.0
# Max Loss         : -21320.0
# Max Drawdown     : -19.56%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days
