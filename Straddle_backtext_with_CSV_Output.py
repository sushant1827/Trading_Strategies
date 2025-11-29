# Strategy Details

# Time 9:20 AM -> Short CE & PE -> Spot Strike
# Stoploss  -> By point and By PCT (%) -> 25% on both leg
# Exit on SL Hit OR EOD - 03:20 PM 
# Exit only one leg on SL Hit and bring other leg SL to Cost * SL_Factor (0.95) i.e. Sell price * 0.95

# -------------------------------------------------------------------------------------------------
import pandas as pd
import datetime as dt
import glob
import random
import ffn
# import os
# -------------------------------------------------------------------------------------------------

# Get CSV files list from a folder
# path = os.getcwd()
path = r'D:\AlgoTrade\Fyers_AlgoTrade\Options_Weekly'
csv_files = glob.glob(path + "/*.csv")

# This creates a list of dataframes
df_list = (pd.read_csv(file) for file in csv_files)
# print(df_list)

n = 0
m = 0
minutesInDay = 375
max_profit = 180000.0

# trade_df = pd.DataFrame(columns=['Straddle_Open_DateTime', 'Day', 'Price', 'Profit','Cum Profit'])
# 'Lot_Size', 'Slippage', 'Brokerage', 
trade_df = pd.DataFrame(columns=['Start_DateTime', 'Day', 'Spot_Price', 'Strike_Price', 'CE_Sell_Price', 'PE_Sell_Price',
                                 'SL_by_Point', 'SL_by_PCT', 'SL_Pt_Pct', 'PE_Pt_Pct', 'CE_SL_Price', 'PE_SL_Price', 
                                 'CE_SL_Hit', 'CE_SL_Hit_DateTime', 'CE_SL_Hit_Price', 'CE_Loss', 'New_PE_SL_Price',
                                 'PE_SL_Hit', 'PE_SL_Hit_DateTime', 'PE_SL_Hit_Price', 'PE_Loss', 'New_CE_SL_Price',
                                 'EOD_DateTime', 'CE_Buy_Price_EOD', 'PE_Buy_Price_EOD', 'CE_Profit_EOD', 'PE_Profit_EOD',
                                 'Profit', 'Profit_PCT', 'Cum_Profit'
                                 ])

convert_dict = {
                'Spot_Price'        : float,
                'CE_Sell_Price'     : float,
                'PE_Sell_Price'     : float,
                'CE_SL_Price'       : float,
                'PE_SL_Price'       : float,
                'CE_SL_Hit_Price'   : float,
                'New_PE_SL_Price'   : float,
                'PE_SL_Hit_Price'   : float,
                'New_CE_SL_Price'   : float,
                'CE_Buy_Price_EOD'  : float,
                'PE_Buy_Price_EOD'  : float,
                'CE_Profit_EOD'     : float,
                'PE_Profit_EOD'     : float,
                'Profit'            : float,
                'Profit_PCT'        : float,
                'Cum_Profit'        : float,
               }
 
trade_df = trade_df.astype(convert_dict)

while n <= 20:
    
    print(n)
    
    start_H = 9 #random.randint(9, 10) #12
    start_M = random.randint(27, 54) #30
    end_H   = 14 #random.randint(14, 15)
    end_M   = random.randint(27, 54)

    # Parameters to be Optimized

    # start_time     = dt.time(start_H,start_M,00)
    # end_time       = dt.time(end_H,end_M,00)

    lot_size       = 25

    SL_by_point    = False
    CE_stoploss_pt = 150.0
    PE_stoploss_pt = 150.0

    SL_by_pct      = True
    CE_SL_pct      = random.randint(24, 30) #25.0
    PE_SL_pct      = random.randint(24, 30) #25.0

    SL_Factor      = 0.95

    init_capital   = 150000
    total_profit   = init_capital

    profit_Monday   = 0.0
    profit_Tuesday  = 0.0
    profit_Wednesday= 0.0
    profit_Thursday = 0.0
    profit_Friday   = 0.0

    slippage       = 0.01 # 0.5%
    brokerage      = 25.0

    # Run_on_Days    = ['Friday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']
    Run_on_Days    = ['Tuesday']

    days_data = {

                'Monday' : {
                            'start_H'   : 9,
                            'start_M'   : 18,
                            'end_H'     : 14,
                            'end_M'     : 52,
                            'CE_SL_pct' : 25.0,
                            'PE_SL_pct' : 30.0
                        },  

                'Tuesday' : {
                            'start_H'   : 10,
                            'start_M'   : 21,
                            'end_H'     : 14,
                            'end_M'     : 44,
                            'CE_SL_pct' : 26.0,
                            'PE_SL_pct' : 28.0
                        },

                'Wednesday' : {
                            'start_H'   : 9,  
                            'start_M'   : 45,
                            'end_H'     : 14,
                            'end_M'     : 44,
                            'CE_SL_pct' : 29.0,
                            'PE_SL_pct' : 25.0
                            },

                'Thursday' : {
                            'start_H'   : 9,  
                            'start_M'   : 50,
                            'end_H'     : 14,
                            'end_M'     : 34,
                            'CE_SL_pct' : 28.0,
                            'PE_SL_pct' : 28.0
                            },

                'Friday' : {
                            'start_H'   : 9,  
                            'start_M'   : 27,
                            'end_H'     : 14,
                            'end_M'     : 27,
                            'CE_SL_pct' : 25.0,
                            'PE_SL_pct' : 25.0
                            }
    }

    # w = 0

    # loop over the list of csv files
    for f in csv_files:
        
        # w += 1
        # print(w)

        # read the csv file
        df = pd.read_csv(f)
        df = df.rename(columns={'Unnamed: 0': 'datetime'})
        # print(df.head)

        df.drop(['open_interest', 'volume'], axis=1, inplace=True) # 'exchange_code', 'product_type', 'Unnamed: 0', 

        # -------------------------------------------------------------------------------------------------

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['expiry_date'] = pd.to_datetime(df['expiry_date'])
        df['date'] = df['datetime'].dt.date
        df['day'] = df['datetime'].dt.day_name()
        df['time'] = df['datetime'].dt.time

        df.set_index('datetime', inplace=True)

        # df.head(10)

        # -------------------------------------------------------------------------------------------------

        spot_df = pd.read_csv(r'D:\AlgoTrade\Fyers_AlgoTrade\BankNifty_Index\NIFTYBANK_INDEX_1_Min.csv')
        # spot_df.head()

        spot_df['Date'] = pd.to_datetime(spot_df['Date'])
        spot_df.set_index('Date', inplace=True)
        # spot_df.head()

        # -------------------------------------------------------------------------------------------------

        # trade_df.loc[m] = ([str(start_datetime), start_day,                                  
        #         spot_price, strike_price, 
        #         CE_sell_price, PE_sell_price,
        #         str(SL_by_point), str(SL_by_pct), CE_stoploss_ptc, PE_stoploss_ptc, xCE_SL_price, xPE_SL_price, 
        #         str(CE_stoploss), CE_SL_time, CE_SL_hit_price, CE_loss, nCE_SL_price,
        #         str(PE_stoploss), PE_SL_time, PE_SL_hit_price, PE_loss, nPE_SL_price,
        #         str(end_datetime), CE_buy_price, PE_buy_price, 
        #         CE_profit, PE_profit,
        #         profit, profit_pct, total_profit
        #         ])

        # def clear_all():        

        dates = df.date.unique()
        days  = df.day.unique()

        for j in range(len(dates)):

            start_date    = dates[j]

            straddle        = False
            start_day       = ''
            spot_price      = 0.0
            strike_price    = 0
            CE_sell_price   = 0.0
            PE_sell_price   = 0.0
            CE_stoploss     = False
            PE_stoploss     = False
            xCE_stoploss     = False
            xPE_stoploss     = False
            CE_stoploss_ptc = 0.0
            PE_stoploss_ptc = 0.0
            xCE_SL_price    = 0.0
            xPE_SL_price    = 0.0
            CE_SL_time      = ''
            PE_SL_time      = ''
            CE_SL_hit_price = 0.0
            PE_SL_hit_price = 0.0
            CE_loss         = 0.0
            PE_loss         = 0.0
            nCE_SL_price    = 0.0
            nPE_SL_price    = 0.0
            CE_buy_price    = 0.0
            PE_buy_price    = 0.0
            CE_profit       = 0.0
            PE_profit       = 0.0
            profit          = 0.0
            profit_pct      = 0.0        
            CE_high_price = 0.0
            PE_high_price = 0.0

            xdf = pd.DataFrame()

            if not start_date.strftime("%A") in Run_on_Days:
                # print(start_date.strftime("%A"))
                # print('*******************')
                continue       

            # day_data = days_data[start_date.strftime("%A")]

            # start_H   = day_data['start_H']
            # start_M   = day_data['start_M']
            # end_H     = day_data['end_H']
            # end_M     = day_data['end_M']
            # CE_SL_pct = day_data['CE_SL_pct']
            # PE_SL_pct = day_data['PE_SL_pct']

            start_time     = dt.time(start_H,start_M,00)
            end_time       = dt.time(end_H,end_M,00)

            start_datetime = dt.datetime.combine(start_date, start_time)
            end_datetime   = dt.datetime.combine(start_date, end_time)

            if not (start_datetime in spot_df.index):
                continue    

            spot_price = spot_df.loc[start_datetime, 'Open']
            # print('Spot Price    = ' + str(spot_price))

            strike_price = int(round(spot_price / 100, 0) * 100)
            # print('Strike Price  = ' + str(strike_price))

            xdf = df[df['date'] == start_date]
            xdf = xdf[(xdf['strike_price'] == strike_price)]
            xdf_CE = xdf[(xdf['right'] == 'Call')]
            xdf_PE = xdf[(xdf['right'] == 'Put')]

            if not len(xdf_CE) == len(xdf_PE):
                continue

            for i in range(len(xdf_CE)):
            # for i in range(minutesInDay):

                # if not start_time in xdf_CE['time'].values or not end_time in xdf_PE['time'].values:
                #     break

                if xdf_CE['time'][i] == start_time and not straddle:

                    start_day = str(xdf_CE.day[i])

                    # print('----------------------------------------')
                    # print('Placing Straddle Order')
                    # print('----------------------------------------')

                    straddle = True

                    # # print('Date          = ' + str(xdf_CE.index[i]))
                    # print('Day           = ' + start_day)
                    # print('Spot Price    = ' + str(spot_price))
                    # print('Strike Price  = ' + str(strike_price))

                    CE_sell_price = xdf_CE['open'][i] * (1 - slippage)
                    # CE_sell_price = xdf.loc[(xdf['right'] == 'Call'), 'open'].values[i]
                    # print('CE Sell Price = ' + str(CE_sell_price))

                    PE_sell_price = xdf_PE['open'][i] * (1 - slippage)
                    # PE_sell_price = xdf.loc[(xdf['right'] == 'Put'), 'open'].values[i]
                    # print('PE Sell Price = ' + str(PE_sell_price))

                    # if SL_by_point:
                    #     CE_SL_price = CE_sell_price + CE_stoploss_pt
                    #     PE_SL_price = PE_sell_price + PE_stoploss_pt
                    #     CE_stoploss_ptc = CE_stoploss_pt
                    #     PE_stoploss_ptc = PE_stoploss_pt
                    
                    if SL_by_pct:
                        CE_SL_price = CE_sell_price * (1 + (CE_SL_pct / 100.0))
                        PE_SL_price = PE_sell_price * (1 + (PE_SL_pct / 100.0))

                        CE_stoploss_ptc = CE_SL_pct
                        PE_stoploss_ptc = PE_SL_pct

                    xCE_SL_price = CE_SL_price
                    xPE_SL_price = PE_SL_price
                    
                    # print('CE SL Price   = ' + str(CE_SL_price))
                    # print('PE SL Price   = ' + str(PE_SL_price))

                if xdf_CE['time'][i] >= start_time and xdf_CE['time'][i] < end_time and straddle and not CE_stoploss: # and not xCE_stoploss:

                    # CE_high_price = xdf.loc[(xdf['right'] == 'Call'), 'high'].values[i]
                    CE_high_price = xdf_CE['high'][i]
                    CE_stoploss   = CE_high_price >= CE_SL_price

                    if CE_stoploss:

                        # CE_stoploss = False
                        # xCE_stoploss = True

                        CE_SL_time = str(xdf_CE.index[i])
                        CE_SL_hit_price = CE_high_price
                        
                        # print('----------------------------------------')
                        # print('CE Stoploss Triggered')
                        # print('----------------------------------------')

                        # # print('Date          = ' + str(xdf.index[i]))
                        # print('Date          = ' + str(xdf_CE.index[i]))
                        # print('CE SL Price   = ' + str(CE_high_price))

                        # if SL_by_point:
                        #     CE_loss = 0.0 - CE_stoploss_pt * lot_size

                        if SL_by_pct:
                            CE_loss = 0.0 - (CE_SL_price - CE_sell_price) * lot_size
                        
                        PE_SL_price = PE_sell_price * SL_Factor
                        nPE_SL_price = PE_SL_price

                        # print('CE Loss       = ' + str(CE_loss))
                        # print('New PE SL     = ' + str(PE_SL_price))

                        # CE_profit   = CE_loss


                if xdf_PE['time'][i] >= start_time and xdf_PE['time'][i] < end_time and straddle and not PE_stoploss:# and not xPE_stoploss:

                    # PE_high_price = xdf.loc[(xdf['right'] == 'Put'), 'high'].values[i]
                    PE_high_price = xdf_PE['high'][i]
                    PE_stoploss   = PE_high_price >= PE_SL_price

                    if PE_stoploss:

                        # PE_stoploss = False
                        # xPE_stoploss = True
                        
                        PE_SL_time  = str(xdf_PE.index[i])
                        PE_SL_hit_price = PE_high_price

                        # print('----------------------------------------')
                        # print('PE Stoploss Triggered')
                        # print('----------------------------------------')

                        # print('Date          = ' + str(xdf.index[i]))
                        # print('Date          = ' + str(xdf_PE.index[i]))
                        # print('PE SL Price   = ' + str(PE_high_price))

                        # if SL_by_point:
                        #     PE_loss = 0.0 - PE_stoploss_pt * lot_size

                        if SL_by_pct:
                            PE_loss = 0.0 - (PE_SL_price - PE_sell_price) * lot_size

                        CE_SL_price = CE_sell_price * SL_Factor
                        nCE_SL_price = CE_SL_price

                        # print('PE Loss       = ' + str(PE_loss))
                        # print('New CE SL     = ' + str(CE_SL_price))

                        # PE_profit   = PE_loss

                
                if xdf_CE['time'][i] == end_time and straddle:

                    # print('----------------------------------------')
                    # print('Closing Straddle Order')
                    # print('----------------------------------------')

                    straddle = False

                    # print('Date          = ' + str(xdf.index[i]))
                    # print('Date          = ' + str(xdf_CE.index[i]))
                    # print('Spot Price    = ' + str(spot_price))
                    # print('Strike Price  = ' + str(strike_price))
                    # print('========================================')

                    # print('Lot Size      = ' + str(lot_size))

                    # CE_buy_price = xdf.loc[(xdf['right'] == 'Call'), 'open'].values[i]
                    CE_buy_price = xdf_CE['open'][i] * (1 + slippage)
                    # PE_buy_price = xdf.loc[(xdf['right'] == 'Put'), 'open'].values[i]
                    PE_buy_price = xdf_PE['open'][i] * (1 + slippage)

                    if CE_stoploss:                
                        CE_profit   = CE_loss
                        # CE_stoploss = False
                    else:                
                        # print('CE Buy Price  = ' + str(CE_buy_price))
                        CE_profit = round(CE_sell_price - CE_buy_price, 2) * lot_size

                    if PE_stoploss:
                        PE_profit   = PE_loss
                        # PE_stoploss = False
                    else:                
                        # print('PE Buy Price  = ' + str(PE_buy_price))
                        PE_profit = round(PE_sell_price - PE_buy_price, 2) * lot_size                        
                        
                    # print('CE Profit/Loss= ' + str(CE_profit))
                    # print('PE Profit/Loss= ' + str(PE_profit))

                    profit = round(CE_profit + PE_profit, 2) - brokerage*2.0
                    # print('Straddle PnL  = ' + str(profit))

                    if start_date.strftime("%A") == 'Monday':
                        profit_Monday   += profit
                    elif start_date.strftime("%A") == 'Tuesday':
                        profit_Tuesday  += profit
                    elif start_date.strftime("%A") == 'Wednesday':
                        profit_Wednesday  += profit
                    elif start_date.strftime("%A") == 'Thursday':
                        profit_Thursday  += profit
                    elif start_date.strftime("%A") == 'Friday':
                        profit_Friday  += profit                  

                    # print('========================================')

                    profit_pct = round(((profit / init_capital) * 100.0), 2)

                    total_profit += profit

                    # print('\n')

                    if not CE_stoploss:
                        CE_SL_time      = ''
                        CE_SL_hit_price = 0.0
                        CE_loss         = 0.0
                        nPE_SL_price    = 0.0
                    else:
                        CE_buy_price    = 0.0

                    if not PE_stoploss:
                        PE_SL_time      = ''
                        PE_SL_hit_price = 0.0
                        PE_loss         = 0.0
                        nCE_SL_price    = 0.0
                    else:
                        PE_buy_price    = 0.0

                    # if m == 0:

                    #     trade_df.loc[m] = ([start_datetime, '', 0.0, 0, 0.0, 0.0,
                    #                         '', '', 0.0, 0.0, 0.0, 0.0, 
                    #                         '', '', 0.0, 0.0, 0.0, 
                    #                         '', '', 0.0, 0.0, 0.0, 
                    #                         '', 0.0, 0.0, 0.0, 0.0,
                    #                         0.0, 0.0, init_capital])

                    #     m += 1

                    # trade_df.loc[m] = ([start_datetime, start_day, spot_price, strike_price, CE_sell_price, PE_sell_price,
                    #                     str(SL_by_point), str(SL_by_pct), CE_stoploss_ptc, PE_stoploss_ptc, xCE_SL_price, xPE_SL_price, 
                    #                     str(CE_stoploss), CE_SL_time, CE_SL_hit_price, CE_loss, nCE_SL_price,
                    #                     str(PE_stoploss), PE_SL_time, PE_SL_hit_price, PE_loss, nPE_SL_price,
                    #                     end_datetime, CE_buy_price, PE_buy_price, CE_profit, PE_profit,
                    #                     profit, profit_pct, total_profit
                    #                     ])
                    
                    # m += 1

                    # clear_all()                 

                    # trade_df.to_csv('Temp'+ '_' + ticker + '_' + type1 + '_' + interval + '_Min' + '.csv')
                    # trade_df.to_csv('Temp1'+ '_' + '.csv')

                    break


    n = n + 1
    
    if total_profit > max_profit:

        max_profit = total_profit

        # Run_on_Days    = ['Friday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']

        print('****************************************')
        print('Start Hour       = ' + str(start_H))
        print('Start Minute     = ' + str(start_M))
        print('End Hour         = ' + str(end_H))
        print('End Minute       = ' + str(end_M))
        print('CE SL %          = ' + str(CE_SL_pct))
        print('PE SL %          = ' + str(PE_SL_pct))
        print('Profit Monday    = ' + str(profit_Monday))
        print('Profit Tuesday   = ' + str(profit_Tuesday))
        print('Profit Wednesday = ' + str(profit_Wednesday))
        print('Profit Thursday  = ' + str(profit_Thursday))
        print('Profit Friday    = ' + str(profit_Friday))
        print('----------------------------------------')
        print('Total Profit     = ' + str(total_profit))
        print('Profit %         = ' + str((round(((total_profit - init_capital) / init_capital) * 100, 2))) + '%')
        print('****************************************')
    
    # trade_df.set_index('Start_DateTime', inplace=True)
    # perf = trade_df['Cum_Profit'].calc_stats()
    # # # perf.plot()    
    # perf.display()
    # print('****************************************')
    # perf.display_monthly_returns()    
    # # # ffn.to_drawdown_series(trade_df['Cum_Profit']).plot(figsize=(15,7),grid=True)
    # perf.stats    
    # perf.display_lookback_returns()
    
