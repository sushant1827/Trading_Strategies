import pandas as pd
import numpy as np
import pandas_ta as pta
import datetime as dt

import glob
import os

# Read the OHLC data from CSV file

# df = pd.read_csv('NIFTY50_INDEX_15_Min.csv')
df = pd.read_csv(r'D:\AlgoTrade\Fyers_AlgoTrade\BankNifty_Index\NIFTYBANK_INDEX_15_Min.csv')

df = df[21467:] # 15 Mins
# df = df[-100:] # 15 Mins
df.head()


# ticker   = 'NIFTY50'
# ticker   = 'NIFTYBANK'
# type1    = 'INDEX'
interval = 15

candles_in_day  = 25
initial_capital = 150000.
position_size   = 25
brokerage_trade = 20.0

df['Date'] = pd.to_datetime(df['Date'])
df['time'] = df['Date'].dt.time
df['date'] = df['Date'].dt.date
df = df[df['time'] != dt.time(9,00,00)]
df.set_index('Date', inplace=True)
# df.drop(['Volume', 'time'], inplace=True, axis=1)
df.drop(['Unnamed: 0', 'Volume'], inplace=True, axis=1)
df.head()
# print(len(df))


# Calculate the Heikin Ashi (HA) candles

df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
df['HA_Open'] = (df['Open'] + df['Close']) / 2.0

for i in range(1, len(df)):
    df['HA_Open'][i] = (df.HA_Open[i-1] + df.HA_Close[i-1]) / 2.0

df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

df.dropna(axis=0, inplace=True)
df.tail(10)


# Generate signals for Buy and Sell

def generate_signals(df):
    
    buy_signal = False
    sell_signal = False    

    for i in range(len(df)):

        if not buy_signal and df['Close'][i-1] > df['ST1'][i-1]: # \
        # if not buy_signal and df['HA_Close'][i-1] > df['ST1'][i-1]: # \            

            buy_signal  = True
            sell_signal = False
            df['Signal'][i] = 'Buy'            


        if not sell_signal and df['Close'][i-1] < df['ST1'][i-1]: # \
        # if not sell_signal and df['HA_Close'][i-1] < df['ST1'][i-1]: # \

            sell_signal = True
            buy_signal  = False
            df['Signal'][i] = 'Sell'            


path = r'D:\AlgoTrade\Fyers_AlgoTrade\New_Options_Data_BnF\Jan_2021-May_2023'
csv_files = glob.glob(path + "/*.csv")

def get_value(signal, date, time, spot_price, init=False):

    right = 'CE'

    if signal == 'Buy':
        right = 'PE'

    order_break = False

    strike = int(round(spot_price / 100, 0) * 100)

    datetime_str = str(date) + " " + str(time)
    datetime = dt.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

    df = pd.DataFrame()

    for file in csv_files:

        list_file_name = os.path.basename(file)
        search_file_name = 'banknifty_option_' + str(date) + '.csv'

        open_price = 0.0

        if list_file_name == search_file_name:

            df = pd.read_csv(file)

            values_to_remove = ['BANKNIFTY', 'BANKNIFTY-I']
            df = df[~df['symbol'].isin(values_to_remove)]

            df[['symbol1', 'expiry', 'strike', 'type']] = df['symbol'].str.extract(r'^([A-Z]+)(\d{2}\w{3}\d{2})(\d+)([A-Z]+)$')

            search_result = df[
                (df['date'] == str(date)) &
                (df['time'] == str(time)) &
                (df['strike'] == str(strike)) &
                (df['type'] == right)
            ]

            expiry_date = search_result['expiry'].values[0] if not search_result.empty else None

            if init:

                if expiry_date:
                    expiry_time = dt.time(11, 30, 0)
                    expiry_datetime = dt.datetime.combine(dt.datetime.strptime(expiry_date, "%d%b%y").date(), expiry_time)

                    if datetime <= expiry_datetime:
                        open_price = search_result['open'].values[0] if not search_result.empty else -999
                        break
            else:

                if expiry_date:
                    expiry_time = dt.time(15, 00, 0)
                    expiry_datetime = dt.datetime.combine(dt.datetime.strptime(expiry_date, "%d%b%y").date(), expiry_time)

                    if datetime > expiry_datetime:

                        search_result1 = df[
                            (df['date'] == str(expiry_date)) &
                            (df['time'] == str(expiry_time)) &
                            (df['strike'] == str(strike)) &
                            (df['type'] == right)
                        ]

                        open_price = search_result1['open'].values[0] if not search_result1.empty else -999
                        order_break = True
                        break

                    else:
                        open_price = search_result['open'].values[0] if not search_result.empty else -999
                        break
    
    return open_price, strike, right, order_break



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
    slippage            = 5.0
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
    strike_price        = 0
    option_type         = ''

    trade_df = pd.DataFrame(columns=['Signal','DateTime', 'Strike', 'Option', 'Price','Profit','Cum Profit'])

    convert_dict = {'Price': float,
                    'Profit': float,
                    'Cum Profit': float,
                    }
 
    trade_df = trade_df.astype(convert_dict)

    j = 0
    
    for i in range(len(df)):

        signal      = df.iloc[i, -1]
        date        = df.iloc[i, -7]
        time        = df.iloc[i, -8]
        spot_price  = df.iloc[i, -12]
        order_break = False

        # print(signal, date, time, spot_price)

        if signal == 'Sell' and Buy_ON == True:
            
            sell_premium        = get_value('Buy', date, time, strike_price)[0]
            order_break         = get_value('Buy', date, time, strike_price)[3]

            if not order_break:
                buy_sell_value      = sell_premium #- slippage
                profit_loss         = (buy_value - buy_sell_value) * position_size
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

                trade_df.loc[j] = (['Closed Buy', str(df.index[i]), str(strike_price), option_type, str(buy_sell_value), 
                                    str(profit_loss), str(total_profit_loss)])
            
            else:
                Buy_ON          = False
                buy_value       = 0.0
                buy_sell_value  = 0.0
                buy_date_start  = None
                buy_date_end    = None
                max_buy_count   = 0


        if signal == 'Buy' and Sell_ON == True:
            
            buy_premium         = get_value('Sell', date, time, strike_price)[0]
            order_break         = get_value('Buy', date, time, strike_price)[3]

            if not order_break:
                sell_buy_value      = buy_premium #+ slippage
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

                trade_df.loc[j] = (['Closed Sell', str(df.index[i]), str(strike_price), option_type, str(sell_buy_value), 
                                    str(profit_loss), str(total_profit_loss)])
                
            else:
                Sell_ON         = False
                sell_value      = 0.0
                sell_buy_value  = 0.0
                sell_date_start = None
                sell_date_end   = None
                max_sell_count  = 0


        if signal == 'Buy' and Buy_ON == False:

            day_name = df['date'][i].strftime("%A")

            if  day_name == 'Thursday' and df['time'][i] > dt.time(11, 30):
                continue 

            buy_premium, strike_price, option_type, order_break = get_value(signal, date, time, spot_price, init=True)

            if buy_premium > 0.0:
                buy_value       = buy_premium #+ slippage
                trade_count    += 1
                Buy_ON          = True
                buy_date_start  = df.iloc[i, 5]
                
                j += 1

                trade_df.loc[j] = (['Buy', str(df.index[i]), str(strike_price), option_type, str(buy_value), '', ''])

        
        if signal == 'Sell' and Sell_ON == False:

            day_name = df['date'][i].strftime("%A")

            if  day_name == 'Thursday' and df['time'][i] > dt.time(11, 30):
                continue

            sell_premium, strike_price, option_type, order_break = get_value(signal, date, time, spot_price, init=True)

            if sell_premium > 0.0:
                sell_value      = sell_premium #- slippage
                trade_count    += 1
                Sell_ON         = True
                sell_date_start = df.iloc[i, 5]

                j += 1

                trade_df.loc[j] = (['Sell', str(df.index[i]), str(strike_price), option_type, str(sell_value), '', ''])


        if profit_loss > max_profit:
            max_profit = profit_loss

        if profit_loss < max_loss:
            max_loss = profit_loss


    return total_profit_loss, trade_count, max_profit, max_loss, trade_df,  max_buy, max_sell, buy_profit_loss, sell_profit_loss



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



i = 0
max_pnl = 0.0

while i <= 0:

    print(i)

    Supertrend_ATR1  = 6 #random.randrange(2, 20)
    Supertrend_Fact1 = 2.0 #random.randrange(1, 5)
    Supertrend_name1 = 'SUPERT_' + str(Supertrend_ATR1) + '_' + str(Supertrend_Fact1) #+ '.5'

    df['ST1'] = pta.supertrend(df['HA_High'], df['HA_Low'], df['HA_Close'], length=Supertrend_ATR1, multiplier=Supertrend_Fact1)[Supertrend_name1]

    df = df[1:]
    
    df['Signal'] = 'Hold'

    generate_signals(df)

    df.dropna(axis=0, inplace=True)
    # df.head()

    # filtered_df = df[df['Signal'] != 'Hold']

    df.to_csv(str(Supertrend_ATR1) + '_' + str(Supertrend_Fact1) + '-' + str(interval) + '_Mins_Signals'+ '.csv')

    profit_loss, trade_count, max_profit, max_loss, trade_df, max_buydays, max_selldays, buy_pnl, sell_pnl = backtest(df)  

    # pct_Pnl = round((((profit_loss - (trade_count * brokerage_trade) - initial_capital) / initial_capital) * 100.), 2)

    # # if pct_Pnl > max_pnl :
        
    # max_pnl = pct_Pnl
    
    # print('Supertrends      :', str(Supertrend_ATR1) + '_' + str(Supertrend_Fact1))
    # print('Initial Capital  :', round(initial_capital, 2))
    # print('Total Profit/Loss:', round(profit_loss-initial_capital, 2))
    # print('Buy Profit/Loss  :', round(buy_pnl, 2))
    # print('Sell Profit/Loss :', round(sell_pnl, 2))
    # print('Number of Trades :', trade_count)
    # print('Brokerage        :', (trade_count * brokerage_trade))
    # print('Net Profit/Loss  :', round((profit_loss - initial_capital - (trade_count * brokerage_trade)), 2))
    # print('Profit/Loss %    :', str(round((((profit_loss - (trade_count * brokerage_trade) - initial_capital) / initial_capital) * 100.), 2)) + '%')
    # print('Buy PnL %        :', str(round(((buy_pnl / initial_capital) * 100.), 2)) + '%')
    # print('Sell PnL %       :', str(round(((sell_pnl  / initial_capital) * 100.), 2)) + '%')
    # print('Max Profit       :', round((max_profit - brokerage_trade), 2))
    # print('Max Loss         :', round((max_loss - brokerage_trade), 2))
    # print('Max Drawdown     :', str(round(MDD(trade_df), 2)) + '%')
    # print('Max Buy Days     :', str(max_buydays + 1) + ' Days')
    # print('Max Sell Days    :', str(max_selldays + 1 ) + ' Days')

    trade_df.to_csv(str(Supertrend_ATR1) + '_' + str(Supertrend_Fact1) + '-' + str(interval) + '_Mins'+ '.csv')

    i += 1


