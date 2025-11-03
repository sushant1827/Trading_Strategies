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

df1 = df1[21391:] # 15 Mins
# df1 = df1[64164:]

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
df1['time'] = df1['Date'].dt.time
df1['date'] = df1['Date'].dt.date
df1 = df1[df1['time'] != dt.time(9,00,00)]

df1.drop(['Unnamed: 0', 'Volume'], inplace=True, axis=1)
# print(df1.head())

# ---------------------------------------------------------------------------------------

day_df = pd.read_csv(r'D:\AlgoTrade\Fyers_AlgoTrade\Nifty50_Index\NIFTY50_INDEX_D_Min.csv')

day_df = day_df[1133:] # 15 Mins
day_df['Date'] = pd.to_datetime(day_df['Date'])
day_df['date'] = day_df['Date'].dt.date

day_df.drop(['Date', 'Unnamed: 0', 'Volume'], inplace=True, axis=1)
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

# Generate signals for Buy and Sell
# df['Signal_Day'] = np.nan
# df['Signal_Min'] = np.nan

def generate_signals(df):
    
    for i in range(1, len(df)):
        
        if df['EMA_Fast_Day'][i-1] > df['EMA_Slow_Day'][i-1]: df['Signal_Day'][i] = 'Buy'
        if df['EMA_Fast_Day'][i-1] < df['EMA_Slow_Day'][i-1]: df['Signal_Day'][i] = 'Sell'
            
        if df['EMA_Fast_Min'][i-1] > df['EMA_Slow_Min'][i-1]: df['Signal_Min'][i] = 'Buy'
        if df['EMA_Fast_Min'][i-1] < df['EMA_Slow_Min'][i-1]: df['Signal_Min'][i] = 'Sell'

    df['Signal_Day'].fillna(method='ffill', inplace=True)
    df['Signal_Min'].fillna(method='ffill', inplace=True)

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

                buy_date_end  = df.iloc[i, 5]
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

                sell_date_end  = df.iloc[i, 5]
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

            buy_date_end  = df.iloc[i, 5]
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

            sell_date_end  = df.iloc[i, 5]
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
            buy_date_start  = df.iloc[i, 5]
            
            j += 1

            trade_df.loc[j] = (['Buy', str(df.index[i]), str(buy_value), '', ''])

        
        # if signal == 'Sell' and Sell_ON == False:
        if signalD == 'Sell' and signalM == 'Sell' and Sell_ON == False:

            sell_value       = df.iloc[i, 0] - slippage
            sl_sell_value    = sell_value + sl_points
            # sell_value      = df.iloc[i, 0] * (1 - (slippage / 100.0))
            trade_count    += 1
            Sell_ON         = True
            sell_date_start = df.iloc[i, 5]            

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

def get_curr_EMA(df, col_name, period):

    i = 0
    while i < len(df):

        day_unique_values = df.loc[df.index <= i, 'Close_day'].drop_duplicates().tail(period).tolist()        

        if len(day_unique_values) == period:
            # day_unique_values.append(df['Close_15min'][i])
            day_unique_values[-1] = df['Close_15min'][i]
            series = pd.Series(day_unique_values)
            value = pta.ema(series,  length=period)
            # value = pta.ema(day_unique_values, length=period)[-1]
            df[col_name][i] = value.iloc[-1]
            # print(value.iloc[-1])

        i += 1

# ---------------------------------------------------------------------------------------

i = 0
max_pnl = 0.0

while i <= 0:

    print(i)

    fast_EMA_period_min = 11 #random.randrange(8, 13)
    slow_EMA_period_min = 24 #random.randrange(24, 26)

    df1['EMA_Fast_Min'] = pta.ema(df1['Close'],  length=fast_EMA_period_min) 
    df1['EMA_Slow_Min'] = pta.ema(df1['Close'],  length=slow_EMA_period_min) 

    df1.dropna(axis=0, inplace=True)

    fast_EMA_period_day = 5 #random.randrange(2, 10)
    slow_EMA_period_day = 10 #random.randrange(6, 12)

    # ema_fast_Day = get_curr_EMA(day_df, fast_EMA_period_day, cur_close, date, time) 

    # day_df['EMA_Fast_Day'] = pta.ema(day_df['Close'],  length=fast_EMA_period_day) 
    # day_df['EMA_Slow_Day'] = pta.ema(day_df['Close'],  length=slow_EMA_period_day) 

    # day_df['EMA_Fast_Day'] = day_df['EMA_Fast_Day'].shift(1)
    # day_df['EMA_Slow_Day'] = day_df['EMA_Slow_Day'].shift(1)

    # day_df.dropna(axis=0, inplace=True) 

    df = pd.merge(df1, day_df, on='date', suffixes=('_15min', '_day'))
    # df.set_index('Date', inplace=True)
    # df.head()

    df['EMA_Fast_Day'] = np.nan
    df['EMA_Slow_Day'] = np.nan

    get_curr_EMA(df, 'EMA_Fast_Day', fast_EMA_period_day)
    get_curr_EMA(df, 'EMA_Slow_Day', slow_EMA_period_day)

    df.dropna(axis=0, inplace=True) 
    df.reset_index(inplace=True)
    df.drop('index', inplace=True, axis=1) 
    df.set_index('Date', inplace=True)

    # df = get_curr_EMA(df1, day_df, fast_EMA_period_day, slow_EMA_period_day)

    df['Signal_Day'] = np.nan
    df['Signal_Min'] = np.nan
    # df['Signal']     = np.nan    

    generate_signals(df)

    df.dropna(axis=0, inplace=True)

    # df.to_csv('987.csv')

    profit_loss, trade_count, max_profit, max_loss, trade_df, max_buydays, max_selldays, buy_pnl, sell_pnl = backtest(df)

    # trade_df.to_csv('EMA_Day_' + str(fast_EMA_period_day) + '_' + str(slow_EMA_period_day) + '-' 
    #                 + 'EMA_Mins_' + str(fast_EMA_period_min) + '_' + str(slow_EMA_period_min) + '-' 
    #                 + str(interval) + '_Mins'+ '.csv')

    pct_Pnl = round((((profit_loss - (trade_count * brokerage_trade) - initial_capital) / initial_capital) * 100.), 2)
    print(pct_Pnl)

    if pct_Pnl > max_pnl :
        
        max_pnl = pct_Pnl

        df.to_csv('EMA_Day_' + str(fast_EMA_period_day) + '_' + str(slow_EMA_period_day) + '-' 
                    + 'EMA_Mins_' + str(fast_EMA_period_min) + '_' + str(slow_EMA_period_min) + '-'
                    +  str(interval) + '_Mins_Signals'+ '.csv')
                
        # print('Supertrends      :', str(Supertrend_ATR1) + '_' + str(Supertrend_Fact1))
        print('Day EMAs         :', str(fast_EMA_period_day) + '_' + str(slow_EMA_period_day))
        print('Minute EMAs      :', str(fast_EMA_period_min) + '_' + str(slow_EMA_period_min))        
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

        trade_df.to_csv('EMA_Day_' + str(fast_EMA_period_day) + '_' + str(slow_EMA_period_day) + '-' 
                        + 'EMA_Mins_' + str(fast_EMA_period_min) + '_' + str(slow_EMA_period_min) + '-' 
                        + str(interval) + '_Mins'+ '.csv')

    i += 1



# Day EMAs         : 2_3
# Minute EMAs      : 8_22
# Initial Capital  : 110000.0
# Total Profit/Loss: 957810.0
# Buy Profit/Loss  : 613225.0
# Sell Profit/Loss : 344585.0
# Number of Trades : 737
# Brokerage        : 92125.0
# Net Profit/Loss  : 865685.0
# Profit/Loss %    : 786.99%
# Buy PnL %        : 557.48%
# Sell PnL %       : 313.26%
# Max Profit       : 51160.0
# Max Loss         : -17910.0
# Max Drawdown     : -18.31%
# Max Buy Days     : 13 Days
# Max Sell Days    : 8 Days

# Day EMAs         : 5_10
# Minute EMAs      : 11_24
# Initial Capital  : 110000.0
# Total Profit/Loss: 418840.0
# Buy Profit/Loss  : 274295.0
# Sell Profit/Loss : 144545.0
# Number of Trades : 603
# Brokerage        : 75375.0
# Net Profit/Loss  : 343465.0
# Profit/Loss %    : 312.24%
# Buy PnL %        : 249.36%
# Sell PnL %       : 131.4%
# Max Profit       : 42240.0
# Max Loss         : -8565.0
# Max Drawdown     : -31.27%
# Max Buy Days     : 13 Days
# Max Sell Days    : 12 Days

# Day EMAs         : 5_10
# Minute EMAs      : 11_24
# Initial Capital  : 110000.0
# Total Profit/Loss: 390655.0
# Buy Profit/Loss  : 272280.0
# Sell Profit/Loss : 118375.0
# Number of Trades : 530
# Brokerage        : 66250.0
# Net Profit/Loss  : 324405.0
# Profit/Loss %    : 294.91%
# Buy PnL %        : 247.53%
# Sell PnL %       : 107.61%
# Max Profit       : 42240.0
# Max Loss         : -15505.0
# Max Drawdown     : -31.85%
# Max Buy Days     : 13 Days
# Max Sell Days    : 12 Days

# Day EMAs         : 5_7
# Minute EMAs      : 7_24
# Initial Capital  : 110000.0
# Total Profit/Loss: 391215.0
# Buy Profit/Loss  : 233930.0
# Sell Profit/Loss : 157285.0
# Number of Trades : 690
# Brokerage        : 86250.0
# Net Profit/Loss  : 304965.0
# Profit/Loss %    : 277.24%
# Buy PnL %        : 212.66%
# Sell PnL %       : 142.99%
# Max Profit       : 39190.0
# Max Loss         : -16050.0
# Max Drawdown     : -17.15%
# Max Buy Days     : 9 Days
# Max Sell Days    : 9 Days

# Day EMAs         : 5_10
# Minute EMAs      : 11_28
# Initial Capital  : 110000.0
# Total Profit/Loss: 353670.0
# Buy Profit/Loss  : 252520.0
# Sell Profit/Loss : 101150.0
# Number of Trades : 486
# Brokerage        : 60750.0
# Net Profit/Loss  : 292920.0
# Profit/Loss %    : 266.29%
# Buy PnL %        : 229.56%
# Sell PnL %       : 91.95%
# Max Profit       : 42320.0
# Max Loss         : -15505.0
# Max Drawdown     : -37.41%
# Max Buy Days     : 13 Days
# Max Sell Days    : 12 Days

# Day EMAs         : 5_10
# Minute EMAs      : 10_25
# Initial Capital  : 110000.0
# Total Profit/Loss: 358535.0
# Buy Profit/Loss  : 261490.0
# Sell Profit/Loss : 97045.0
# Number of Trades : 542
# Brokerage        : 67750.0
# Net Profit/Loss  : 290785.0
# Profit/Loss %    : 264.35%
# Buy PnL %        : 237.72%
# Sell PnL %       : 88.22%
# Max Profit       : 42240.0
# Max Loss         : -15505.0
# Max Drawdown     : -39.86%
# Max Buy Days     : 13 Days
# Max Sell Days    : 12 Days

# Day EMAs         : 6_10
# Minute EMAs      : 7_28
# Initial Capital  : 110000.0
# Total Profit/Loss: 313097.5
# Buy Profit/Loss  : 231542.5
# Sell Profit/Loss : 81555.0
# Number of Trades : 592
# Brokerage        : 74000.0
# Net Profit/Loss  : 239097.5
# Profit/Loss %    : 217.36%
# Buy PnL %        : 210.49%
# Sell PnL %       : 74.14%
# Max Profit       : 36265.0
# Max Loss         : -16050.0
# Max Drawdown     : -27.66%
# Max Buy Days     : 11 Days
# Max Sell Days    : 11 Days

# Day EMAs         : 2_3
# Minute EMAs      : 8_22
# Initial Capital  : 110000.0
# Total Profit/Loss: 344020.0
# Buy Profit/Loss  : 280320.0
# Sell Profit/Loss : 63700.0
# Number of Trades : 867
# Brokerage        : 108375.0
# Net Profit/Loss  : 235645.0
# Profit/Loss %    : 214.22%
# Buy PnL %        : 254.84%
# Sell PnL %       : 57.91%
# Max Profit       : 45485.0
# Max Loss         : -12460.0
# Max Drawdown     : -21.36%
# Max Buy Days     : 13 Days
# Max Sell Days    : 8 Days
