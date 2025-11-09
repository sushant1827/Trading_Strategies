import pandas as pd
import numpy as np
import pandas_ta as pta
import datetime as dt
import math
import random

pd.set_option('mode.chained_assignment', None)

# ---------------------------------------------------------------------------------------

# Read the OHLC data from CSV file

# df = pd.read_csv('NIFTY50_INDEX_15_Min.csv')
# df1 = pd.read_csv(r'D:\AlgoTrade\Fyers_AlgoTrade\BankNifty_Index\NIFTYBANK_INDEX_15_Min.csv')
df1 = pd.read_csv(r'D:\AlgoTrade\Fyers_AlgoTrade\Nifty50_Index\NIFTY50_INDEX_15_Min.csv')

df1 = df1[20791:] # 15 Mins
# df1 = df1[62589:]

# print(df1.head())

# ---------------------------------------------------------------------------------------

# ticker   = 'NIFTY50'
# ticker   = 'NIFTYBANK'
# type1    = 'INDEX'
interval = 15

candles_in_day  = 25
# initial_capital = 170000.
initial_capital = 110000.
# position_size   = 25
position_size   = 50
brokerage_trade = 125.0

df1['Date'] = pd.to_datetime(df1['Date'])
df1['date'] = df1['Date'].dt.date
df1['time'] = df1['Date'].dt.time
df1 = df1[df1['time'] != dt.time(9,00,00)]

df1.drop(['Unnamed: 0', 'Volume'], inplace=True, axis=1)
# print(df1.head())

# ---------------------------------------------------------------------------------------

day_df = pd.read_csv(r'D:\AlgoTrade\Fyers_AlgoTrade\Nifty50_Index\NIFTY50_INDEX_60_Min.csv')

day_df = day_df[5874:] # 15 Mins
# day_df = day_df[10910:] # 15 Mins
day_df['Date'] = pd.to_datetime(day_df['Date'])
day_df['date'] = day_df['Date'].dt.date
day_df['time'] = day_df['Date'].dt.time
day_df = day_df[day_df['time'] != dt.time(8,15,00)]
# print(day_df.head())
day_df.drop(['Unnamed: 0', 'Volume'], inplace=True, axis=1)
# print(day_df.head())

# ---------------------------------------------------------------------------------------

def generate_signals(df):
    
    for i in range(1, len(df)):
        
        if df['plus_di'][i-1] > df['minus_di'][i-1]: 
            df['Signal_Mins'][i] = 'Buy'

        if df['K'][i-1] > df['D'][i-1]: 
            df['Signal_Hour'][i] = 'Buy'

        if df['plus_di'][i-1] < df['minus_di'][i-1]: 
            df['Signal_Mins'][i] = 'Sell'

        if df['K'][i-1] < df['D'][i-1]: 
            df['Signal_Hour'][i] = 'Sell'

    df['Signal_Hour'].fillna(method='ffill', inplace=True)
    df['Signal_Mins'].fillna(method='ffill', inplace=True)

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

                buy_date_end  = df.iloc[i, 4]
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

                sell_date_end  = df.iloc[i, 4]
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


        if signalD == 'Sell' and signalM == 'Sell' and Buy_ON == True:

            # buy_sell_value      = df.iloc[i, 0] * (1 - (slippage / 100.0))
            buy_sell_value      = df.iloc[i, 0] - slippage
            profit_loss         = (buy_sell_value - buy_value) * position_size
            total_profit_loss  += profit_loss
            buy_profit_loss    += profit_loss
            trade_count        += 1
            Buy_ON              = False

            j += 1

            buy_date_end  = df.iloc[i, 4]
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


        if signalD == 'Buy' and signalM == 'Buy' and Sell_ON == True:

            sell_buy_value      = df.iloc[i, 0] + slippage
            # sell_buy_value      = df.iloc[i, 0] * (1 + (slippage / 100.0))
            profit_loss         = (sell_value - sell_buy_value) * position_size
            total_profit_loss  += profit_loss
            sell_profit_loss   += profit_loss
            trade_count        += 1
            Sell_ON             = False

            j += 1

            sell_date_end  = df.iloc[i, 4]
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


        if signalD == 'Buy' and signalM == 'Buy' and Buy_ON == False:
            
            buy_value       = df.iloc[i, 0] + slippage
            sl_buy_value    = buy_value - sl_points
            # buy_value       = df.iloc[i, 0] * (1 + (slippage / 100.0))
            trade_count    += 1
            Buy_ON          = True
            buy_date_start  = df.iloc[i, 4]
            
            j += 1

            trade_df.loc[j] = (['Buy', str(df.index[i]), str(buy_value), '', ''])

        
        if signalD == 'Sell' and signalM == 'Sell' and Sell_ON == False:

            sell_value       = df.iloc[i, 0] - slippage
            sl_sell_value    = sell_value + sl_points
            # sell_value      = df.iloc[i, 0] * (1 - (slippage / 100.0))
            trade_count    += 1
            Sell_ON         = True
            sell_date_start = df.iloc[i, 4]            

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

def get_curr_stoch_rsi(df, stoch_length, rsi_length):

    i = 0

    while i < len(df):

        day_unique_values = df.loc[df.index <= i, 'Close_Hour'].drop_duplicates().tolist()

        if len(day_unique_values) >= (stoch_length + rsi_length + 4):

            day_unique_values[-1] = df['Close_Mins'][i]
            series = pd.Series(day_unique_values)

            stoch_rsi = pta.stochrsi(close=series, length=stoch_length, rsi_length=rsi_length)
            # print(stoch_rsi)

            k_stoch_rsi = 'STOCHRSIk_' + str(stoch_length) + '_' + str(rsi_length) + '_3_3'
            d_stoch_rsi = 'STOCHRSId_' + str(stoch_length) + '_' + str(rsi_length) + '_3_3'

            df['K'][i] = stoch_rsi.loc[len(stoch_rsi)-1, k_stoch_rsi]
            df['D'][i] = stoch_rsi.loc[len(stoch_rsi)-1, d_stoch_rsi]

        i += 1

# ---------------------------------------------------------------------------------------

i = 0
max_pnl = 0.0

while i <= 0:

    print(i)

    di_length = 28 #random.randrange(25, 33) #10

    adx = pta.adx(high=df1['High'], low=df1['Low'], close=df1['Close'], length=di_length)    
    
    df1['plus_di']  = adx['DMP_' + str(di_length)].round(2)
    df1['minus_di'] = adx['DMN_' + str(di_length)].round(2)

    df1.dropna(axis=0, inplace=True)

    df = pd.merge_asof(df1, day_df, on='Date', direction='backward', suffixes=('_Mins', '_Hour')) # , how='left',
    # df.set_index('Date', inplace=True)
    # df.head()

    stoch_length    = 17 #random.randrange(15, 25) #10
    rsi_length      = stoch_length #random.randrange(7, 15) #10

    df['K'] = np.nan
    df['D'] = np.nan

    get_curr_stoch_rsi(df, stoch_length, rsi_length)
    
    df.dropna(axis=0, inplace=True) 
    df.reset_index(inplace=True)
    df.drop('index', inplace=True, axis=1) 
    df.set_index('Date', inplace=True)
    # print(df.head())    

    df['K'] = df['K'].round(2)
    df['D'] = df['D'].round(2)

    df['Signal_Hour'] = np.nan
    df['Signal_Mins'] = np.nan

    generate_signals(df)

    df.dropna(axis=0, inplace=True)

    # df.to_csv('999.csv')

    profit_loss, trade_count, max_profit, max_loss, trade_df, max_buydays, max_selldays, buy_pnl, sell_pnl = backtest(df)

    # trade_df.to_csv('EMA_Day_' + str(fast_EMA_period_day) + '_' + str(slow_EMA_period_day) + '-' 
    #                 + 'EMA_Mins_' + str(fast_EMA_period_min) + '_' + str(slow_EMA_period_min) + '-' 
    #                 + str(interval) + '_Mins'+ '.csv')

    pct_Pnl = round((((profit_loss - (trade_count * brokerage_trade) - initial_capital) / initial_capital) * 100.), 2)
    print(pct_Pnl)

    if pct_Pnl > max_pnl :
        
        max_pnl = pct_Pnl

        df.to_csv('DI_Len_' + str(di_length) + '-' + 'STOCH_Len_'  + str(stoch_length) + '-' 
                   + 'RSI_Len_' + str(rsi_length) + '-' +  str(interval) + '_Mins_Signals' + '.csv')
                
        print('DI Length        :', str(di_length))
        print('STOCH Length     :', str(stoch_length))
        print('RSI Length       :', str(rsi_length))
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

        trade_df.to_csv('DI_Len_' + str(di_length) + '-' + 'STOCH_Len_'  + str(stoch_length) + '-' 
                        + 'RSI_Len_' + str(rsi_length) + '-' +  str(interval) + '_Mins' + '.csv')

    i += 1


# 288.8
# DI Length        : 23
# STOCH Length     : 19
# RSI Length       : 19
# Initial Capital  : 110000.0
# Total Profit/Loss: 466060.0
# Buy Profit/Loss  : 353555.0
# Sell Profit/Loss : 112505.0
# Number of Trades : 1187
# Brokerage        : 148375.0
# Net Profit/Loss  : 317685.0
# Profit/Loss %    : 288.8%
# Buy PnL %        : 321.41%
# Sell PnL %       : 102.28%
# Max Profit       : 54640.0
# Max Loss         : -9235.0
# Max Drawdown     : -20.81%
# Max Buy Days     : 14 Days
# Max Sell Days    : 14 Days


# 356.73
# DI Length        : 27
# STOCH Length     : 19
# RSI Length       : 19
# Initial Capital  : 110000.0
# Total Profit/Loss: 525280.0
# Buy Profit/Loss  : 374395.0
# Sell Profit/Loss : 150885.0
# Number of Trades : 1063
# Brokerage        : 132875.0
# Net Profit/Loss  : 392405.0
# Profit/Loss %    : 356.73%
# Buy PnL %        : 340.36%
# Sell PnL %       : 137.17%
# Max Profit       : 62595.0
# Max Loss         : -9225.0
# Max Drawdown     : -23.09%
# Max Buy Days     : 16 Days
# Max Sell Days    : 15 Days

# DI Length        : 28
# STOCH Length     : 18
# RSI Length       : 18
# Initial Capital  : 110000.0
# Total Profit/Loss: 548360.0
# Buy Profit/Loss  : 396210.0
# Sell Profit/Loss : 152150.0
# Number of Trades : 1033
# Brokerage        : 129125.0
# Net Profit/Loss  : 419235.0
# Profit/Loss %    : 381.12%
# Buy PnL %        : 360.19%
# Sell PnL %       : 138.32%
# Max Profit       : 61530.0
# Max Loss         : -5625.0
# Max Drawdown     : -19.96%
# Max Buy Days     : 16 Days
# Max Sell Days    : 15 Days

# 1
# 332.26
# DI Length        : 27
# STOCH Length     : 20
# RSI Length       : 20
# Initial Capital  : 110000.0
# Total Profit/Loss: 501360.0
# Buy Profit/Loss  : 361005.0
# Sell Profit/Loss : 140355.0
# Number of Trades : 1087
# Brokerage        : 135875.0
# Net Profit/Loss  : 365485.0
# Profit/Loss %    : 332.26%
# Buy PnL %        : 328.19%
# Sell PnL %       : 127.6%
# Max Profit       : 62595.0
# Max Loss         : -5625.0
# Max Drawdown     : -24.09%
# Max Buy Days     : 16 Days
# Max Sell Days    : 15 Days

# 374.95
# DI Length        : 28
# STOCH Length     : 15
# RSI Length       : 15
# Initial Capital  : 110000.0
# Total Profit/Loss: 540320.0
# Buy Profit/Loss  : 389005.0
# Sell Profit/Loss : 151315.0
# Number of Trades : 1023
# Brokerage        : 127875.0
# Net Profit/Loss  : 412445.0
# Profit/Loss %    : 374.95%
# Buy PnL %        : 353.64%
# Sell PnL %       : 137.56%
# Max Profit       : 61530.0
# Max Loss         : -5465.0
# Max Drawdown     : -35.19%
# Max Buy Days     : 16 Days
# Max Sell Days    : 15 Days


# 354.48
# DI Length        : 29
# STOCH Length     : 19
# RSI Length       : 19
# Initial Capital  : 110000.0
# Total Profit/Loss: 516305.0
# Buy Profit/Loss  : 377360.0
# Sell Profit/Loss : 138945.0
# Number of Trades : 1011
# Brokerage        : 126375.0
# Net Profit/Loss  : 389930.0
# Profit/Loss %    : 354.48%
# Buy PnL %        : 343.05%
# Sell PnL %       : 126.31%
# Max Profit       : 61530.0
# Max Loss         : -9225.0
# Max Drawdown     : -21.7%
# Max Buy Days     : 16 Days
# Max Sell Days    : 15 Days

# 326.96
# 5
# 363.99
# DI Length        : 28
# STOCH Length     : 16
# RSI Length       : 16
# Initial Capital  : 110000.0
# Total Profit/Loss: 528010.0
# Buy Profit/Loss  : 386130.0
# Sell Profit/Loss : 141880.0
# Number of Trades : 1021
# Brokerage        : 127625.0
# Net Profit/Loss  : 400385.0
# Profit/Loss %    : 363.99%
# Buy PnL %        : 351.03%
# Sell PnL %       : 128.98%
# Max Profit       : 61530.0
# Max Loss         : -5465.0
# Max Drawdown     : -34.03%
# Max Buy Days     : 16 Days
# Max Sell Days    : 15 Days

# 384.94
# DI Length        : 28
# STOCH Length     : 19
# RSI Length       : 19
# Initial Capital  : 110000.0
# Total Profit/Loss: 549805.0
# Buy Profit/Loss  : 390980.0
# Sell Profit/Loss : 158825.0
# Number of Trades : 1011
# Brokerage        : 126375.0
# Net Profit/Loss  : 423430.0
# Profit/Loss %    : 384.94%
# Buy PnL %        : 355.44%
# Sell PnL %       : 144.39%
# Max Profit       : 61530.0
# Max Loss         : -9225.0
# Max Drawdown     : -20.99%
# Max Buy Days     : 16 Days
# Max Sell Days    : 15 Days


# 360.9
# DI Length        : 28
# STOCH Length     : 15
# RSI Length       : 15
# Initial Capital  : 110000.0
# Total Profit/Loss: 526365.0
# Buy Profit/Loss  : 391775.0
# Sell Profit/Loss : 134590.0
# Number of Trades : 1035
# Brokerage        : 129375.0
# Net Profit/Loss  : 396990.0
# Profit/Loss %    : 360.9%
# Buy PnL %        : 356.16%
# Sell PnL %       : 122.35%
# Max Profit       : 61530.0
# Max Loss         : -8815.0
# Max Drawdown     : -37.58%
# Max Buy Days     : 16 Days
# Max Sell Days    : 15 Days
# 1
# 365.92
# DI Length        : 28
# STOCH Length     : 16
# RSI Length       : 16
# Initial Capital  : 110000.0
# Total Profit/Loss: 530890.0
# Buy Profit/Loss  : 393625.0
# Sell Profit/Loss : 137265.0
# Number of Trades : 1027
# Brokerage        : 128375.0
# Net Profit/Loss  : 402515.0
# Profit/Loss %    : 365.92%
# Buy PnL %        : 357.84%
# Sell PnL %       : 124.79%
# Max Profit       : 61530.0
# Max Loss         : -5465.0
# Max Drawdown     : -33.58%
# Max Buy Days     : 16 Days
# Max Sell Days    : 15 Days
# 2
# 412.1
# DI Length        : 28
# STOCH Length     : 17
# RSI Length       : 17
# Initial Capital  : 110000.0
# Total Profit/Loss: 581180.0
# Buy Profit/Loss  : 409640.0
# Sell Profit/Loss : 171540.0
# Number of Trades : 1023
# Brokerage        : 127875.0
# Net Profit/Loss  : 453305.0
# Profit/Loss %    : 412.1%
# Buy PnL %        : 372.4%
# Sell PnL %       : 155.95%
# Max Profit       : 61530.0
# Max Loss         : -5625.0
# Max Drawdown     : -20.5%
# Max Buy Days     : 16 Days
# Max Sell Days    : 15 Days
