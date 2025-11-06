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

def calculate_supertrend(prev_supertrend, prev_high, prev_low, prev_close, atr_multiplier):
    """
    Function to calculate the SuperTrend value for the current interval.

    Args:
    prev_supertrend (float): The SuperTrend value from the previous interval.
    prev_high (float): The highest price during the previous interval.
    prev_low (float): The lowest price during the previous interval.
    prev_close (float): The closing price during the previous interval.
    atr_multiplier (float): The ATR (Average True Range) multiplier.

    Returns:
    float: The SuperTrend value for the current interval.
    """
    atr = max(prev_high - prev_low, abs(prev_high - prev_close), abs(prev_low - prev_close))
    return prev_supertrend + atr_multiplier * atr


def generate_signals(df, atr_multiplier):
    
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

        index = 0
        if str(df['time_15min'][i]) == '09:15:00': 
            index = i - 1
        else:
            index = i - 4

        # initial_supertrend  = df.loc[index-1, 'ST1']
        # current_high        = df.loc[i-1, 'High_15min']
        # current_low         = df.loc[i-1, 'Low_15min']
        # current_close       = df.loc[i-1, 'Close_15min']

        # initial_supertrend = calculate_supertrend(initial_supertrend, current_high, current_low, current_close, atr_multiplier)

        if df['Close_15min'][i] > df['ST1'][index]: df['Signal_Day'][i] = 'Buy'
        if df['Close_15min'][i] < df['ST1'][index]: df['Signal_Day'][i] = 'Sell'

        # if df.loc[i, 'Close_15min'] > initial_supertrend: df.loc[i, 'Signal_Day'] = 'Buy'
        # if df.loc[i, 'Close_15min'] < initial_supertrend: df.loc[i, 'Signal_Day'] = 'Sell'

        # ---------------------------------------------------------------

        # hlc3_day  = df.loc[i-(N-1):i, 'HLC3_EMA']

        # nn_day = knn(hlc3_day, N, K)
        # pred_day, dir_day = predict(nn_day, hlc3_day)
        # df.loc[i, 'Signal_Day'] = zscore_color(hlc3_day, LAGZ, dir_day)

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

# day_df['HLC3'] = (day_df['High'] + day_df['Low'] + day_df['Close']) / 3.0
# day_df['HLC3'] = day_df['HLC3'].shift(1)
day_df['Date'] = pd.to_datetime(day_df['Date'])
day_df['date'] = day_df['Date'].dt.date
day_df['time'] = day_df['Date'].dt.time
day_df = day_df[day_df['time'] != dt.time(8,15,00)]

day_df.drop(['Unnamed: 0', 'Volume'], inplace=True, axis=1)
# print(day_df.head())

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

    Supertrend_ATR1  = 2 #random.randrange(2, 13)
    Supertrend_Fact1 = 2.5 #np.random.choice(np.arange(1, 5.25, 0.5), size=1)[0] #float(random.randrange(1, 5))
    Supertrend_name1 = 'SUPERT_' + str(Supertrend_ATR1) + '_' + str(Supertrend_Fact1) #+ '.0'

    day_df['ST1'] = pta.supertrend(day_df['High'], day_df['Low'], day_df['Close'], length=Supertrend_ATR1, 
                                   multiplier=Supertrend_Fact1)[Supertrend_name1]
        
    # EMA_period_day = EMA_period_min #random.randrange(5, 30)
    # day_df['HLC3_EMA'] = ta.EMA(day_df['HLC3'], EMA_period_day) / ta.EMA(day_df['HLC3'], EMA_period_day).mean()
    # day_df['HLC3_EMA'] = 0.0
    
    #--------------------------------------------------------------------------
    
    df = pd.merge_asof(df1, day_df, on='Date', direction='backward', suffixes=('_15min', '_day'))
    df.dropna(axis=0, inplace=True)
    df.reset_index(inplace=True)
    # df.set_index('Date', inplace=True)
    # print(df.head(10))

    # get_curr_EMA(df, 'HLC3_day', 'HLC3_15min', 'HLC3_EMA', EMA_period_day)
    # get_curr_EMA(df, 'EMA_Slow_Day', slow_EMA_period_day)

    # df['ST1'] = 0.0

    df.to_csv('111.csv')

    # get_curr_ST(df, 'High_15min', 'Low_15min', 'Close_15min', 
    #                 'High_day', 'Low_day', 'Close_day',
    #                 Supertrend_ATR1, Supertrend_Fact1, Supertrend_name1, 'ST1')

    df['Signal_Day'] = np.nan
    df['Signal_Min'] = np.nan

    generate_signals(df, Supertrend_Fact1)

    # df['Signal_Day'] = df['Signal_Day'].shift(1)
    df['Signal_Min'] = df['Signal_Min'].shift(1)

    df.dropna(axis=0, inplace=True)
    df.drop('index', inplace=True, axis=1) 
    df.set_index('Date', inplace=True)

    df.to_csv('222.csv')
    # print(df.head(20))

    profit_loss, trade_count, max_profit, max_loss, trade_df, max_buydays, max_selldays, buy_pnl, sell_pnl = backtest(df)

    # trade_df.to_csv('EMA_Day_' + str(EMA_period_day) + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins'+ '.csv')
    trade_df.to_csv(Supertrend_name1 + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins'+ '.csv')

    pct_Pnl = round((((profit_loss - (trade_count * brokerage_trade) - initial_capital) / initial_capital) * 100.), 2)
    print(pct_Pnl)

    if pct_Pnl > max_pnl :
        
        max_pnl = pct_Pnl

        # df.to_csv('EMA_Day_' + str(EMA_period_day) + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins_Signals'+ '.csv')
        df.to_csv(Supertrend_name1 + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins_Signals'+ '.csv')
                
        # print('Supertrends      :', str(Supertrend_ATR1) + '_' + str(Supertrend_Fact1))
        # print('LAGZ             :', str(LAGZ))
        # print('Hour EMAs        :', str(EMA_period_day))
        print('Hour EMAs        :', str(Supertrend_name1))
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

        trade_df.to_csv(Supertrend_name1 + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins' + '.csv')
        # trade_df.to_csv('EMA_Day_' + str(EMA_period_day) + '_' + 'EMA_Mins_' + str(EMA_period_min) + '-' + str(interval) + '_Mins' + '.csv')

    i += 1


# 576.46
# Hour EMAs        : SUPERT_2_2.5
# Minute EMAs      : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 733660.0
# Buy Profit/Loss  : 486330.0
# Sell Profit/Loss : 247330.0
# Number of Trades : 905
# Brokerage        : 99550.0
# Net Profit/Loss  : 634110.0
# Profit/Loss %    : 576.46%
# Buy PnL %        : 442.12%
# Sell PnL %       : 224.85%
# Max Profit       : 53025.0
# Max Loss         : -8995.0
# Max Drawdown     : -16.22%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days

# 531.84
# Hour EMAs        : SUPERT_3_2.5
# Minute EMAs      : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 683035.0
# Buy Profit/Loss  : 443765.0
# Sell Profit/Loss : 239270.0
# Number of Trades : 891
# Brokerage        : 98010.0
# Net Profit/Loss  : 585025.0
# Profit/Loss %    : 531.84%
# Buy PnL %        : 403.42%
# Sell PnL %       : 217.52%
# Max Profit       : 53025.0
# Max Loss         : -8995.0
# Max Drawdown     : -19.75%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days

# 522.9
# Hour EMAs        : SUPERT_4_2.5
# Minute EMAs      : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 672320.0
# Buy Profit/Loss  : 424475.0
# Sell Profit/Loss : 247845.0
# Number of Trades : 883
# Brokerage        : 97130.0
# Net Profit/Loss  : 575190.0
# Profit/Loss %    : 522.9%
# Buy PnL %        : 385.89%
# Sell PnL %       : 225.31%
# Max Profit       : 53025.0
# Max Loss         : -9345.0
# Max Drawdown     : -15.57%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days

# 516.61
# Hour EMAs        : SUPERT_10_2.5
# Minute EMAs      : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 663205.0
# Buy Profit/Loss  : 400220.0
# Sell Profit/Loss : 262985.0
# Number of Trades : 863
# Brokerage        : 94930.0
# Net Profit/Loss  : 568275.0
# Profit/Loss %    : 516.61%
# Buy PnL %        : 363.84%
# Sell PnL %       : 239.08%
# Max Profit       : 53025.0
# Max Loss         : -8995.0
# Max Drawdown     : -14.73%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days


# 516.27
# Hour EMAs        : SUPERT_3_1.5
# Minute EMAs      : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 681970.0
# Buy Profit/Loss  : 451490.0
# Sell Profit/Loss : 230480.0
# Number of Trades : 1037
# Brokerage        : 114070.0
# Net Profit/Loss  : 567900.0
# Profit/Loss %    : 516.27%
# Buy PnL %        : 410.45%
# Sell PnL %       : 209.53%
# Max Profit       : 53025.0
# Max Loss         : -8995.0
# Max Drawdown     : -17.58%
# Max Buy Days     : 13 Days
# Max Sell Days    : 9 Days

