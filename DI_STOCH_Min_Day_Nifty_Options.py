import pandas as pd
import datetime as dt
from datetime import time, timedelta
import os
import glob
import Strategies

# ---------------------------------------------------------------------------------------

def get_atm(symbol, spot_price, offset=0.0):

    if symbol == 'BankNifty':
        base = 100
    if symbol == 'Nifty':
        base = 50

    if spot_price > 0.0:
        strike_price = int(round(spot_price / base, 0) * base)
        offset_price = int(round(offset / base, 0) * base)

    return strike_price + offset_price

# ---------------------------------------------------------------------------------------

def get_options_df(file_path):

    df = pd.read_csv(file_path)

    values_to_remove = ['NIFTY', 'NIFTY-I']
    df = df[~df['symbol'].isin(values_to_remove)]

    df[['symbol1', 'expiry', 'strike', 'type']] = df['symbol'].str.extract(r'^([A-Z]+)(\d{2}\w{3}\d{2})(\d+)([A-Z]+)$')

    return df

# ---------------------------------------------------------------------------------------

def get_options_df_bystrike(df, date, option, strike):

    result = df[
        (df['date'] == str(date)) &
        (df['strike'] == str(strike)) &
        (df['type'] == option)
    ]

    result = result.reset_index(drop=True)
    result['Time'] = pd.to_datetime(result['time'], format='%H:%M:%S')

    if len(result) < 375:

        start_time  = dt.time(9, 15)
        current_time = start_time

        for index, row in result.iterrows():
            if index > 0:
                index_val = index - 1
            else:
                index_val = index

            if row['Time'].time() > current_time:
                previous_row    = result.loc[index_val]
                new_row         = previous_row.copy()
                new_row['Time'] = current_time
                new_row['time'] = str(current_time)

                result = pd.concat([result.loc[:index_val], pd.DataFrame(new_row).transpose(), 
                                    result.loc[index:]]).reset_index(drop=True)
                
                datetime_obj = dt.datetime.combine(dt.datetime.today(), current_time)
                updated_datetime_obj = datetime_obj + timedelta(minutes=1)
                current_time = updated_datetime_obj.time()
            
            datetime_obj = dt.datetime.combine(dt.datetime.today(), current_time)
            updated_datetime_obj = datetime_obj + timedelta(minutes=1)
            current_time = updated_datetime_obj.time()

    return result

# ---------------------------------------------------------------------------------------

def check_sl_hit(position):

    start_date  = position.entry_date
    start_time  = position.entry_time

    end_date    = position.exit_date
    end_time    = position.exit_time

    while start_date <= end_date:
        
        filename    = 'nifty_option_' + str(start_date) + '.csv'
        file_path   = os.path.join(options_data_files, filename)

        options_df  = get_options_df(file_path)

        values_to_remove = ['NIFTY', 'NIFTY-I']
        options_df = options_df[~options_df['symbol'].isin(values_to_remove)]

        options_df[['symbol1', 'expiry', 'strike', 'type']] = options_df['symbol'].str.extract(r'^([A-Z]+)(\d{2}\w{3}\d{2})(\d+)([A-Z]+)$')

        filtered_df = options_df[
                      (options_df['strike']   == str(position.strike)) &
                      (options_df['type']     == position.option)
                      ]
        
        reset_df    = filtered_df.reset_index(drop=True)

        index   = 0
        new_df  = pd.DataFrame()

        if start_date == end_date:
            if start_time:
                index   = reset_df[reset_df['time'] == start_time].index
                if not index.empty:
                    index_val = index[0]
                    reset_df= reset_df.iloc[index_val:] 
                else:
                    new_df = reset_df
            else:
                new_df = reset_df

            if not str(end_time) == '15:15:00':
                index   = reset_df[reset_df['time'] == end_time].index
                if not index.empty:
                    index_val = index[0]
                    new_df  = reset_df.iloc[:index_val]
            else:
                new_df = reset_df
        else:
            if start_time:
                index   = reset_df[reset_df['time'] == start_time].index
                if not index.empty:
                    index_val = index[0]
                    new_df  = reset_df.iloc[index_val:]
                else:
                    new_df = reset_df
            else:
                new_df = reset_df

        filtered_rows  = pd.DataFrame()
        if len(new_df):
            new_df = new_df.reset_index(drop=True)
            filtered_rows = new_df[new_df['high'] >= position.sl_price]

        if len(filtered_rows):
            final_df = filtered_rows.reset_index(drop=True)

            # if not filtered_rows.empty:
            position.sl_hit     = True
            position.sl_date    = str(start_date)
            position.sl_time    = final_df.loc[0, 'time']
            position.sl_hit_at  = final_df.loc[0, 'high']
            break
        else:
            prev_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
            next_date = prev_date + timedelta(days=1)

            while True:
                date       = next_date.strftime("%Y-%m-%d")
                filename   = 'nifty_option_' + str(date) + '.csv'
                file_path  = os.path.join(options_data_files, filename)

                if os.path.exists(file_path):
                    break
                else:
                    if str(date) == '9999-12-31':
                        break
                    next_date = next_date + timedelta(days=1)

            start_date = date
            start_time = None

    return position.sl_hit

# ---------------------------------------------------------------------------------------

symbol = 'Nifty'
signal_filename = r'D:\AlgoTrade\Fyers_AlgoTrade\DI_Len_28-STOCH_Len_17-RSI_Len_17-15_Mins_Signals.csv'

df = pd.read_csv(signal_filename)
# df.set_index('Date', inplace=True)
# df = df[df['Signal'] != 'Hold']
# print(df.head())


for column in df.columns:
    # Check if the column contains decimal values
    if df[column].dtype == float:
        # Round the column values to two decimal places
        df[column] = df[column].round(2)

# df.head()


if symbol == 'BankNifty':
    options_data_files = r'D:\AlgoTrade\Fyers_AlgoTrade\New_Options_Data_BnF\Jan_2021-May_2023'
    lot_size = 25
if symbol == 'Nifty':
    options_data_files = r'D:\AlgoTrade\Fyers_AlgoTrade\New_Options_Data_Nfty\Jan_2021-May_2023'
    lot_size = 50

csv_files = glob.glob(options_data_files + "/*.csv")

# ---------------------------------------------------------------------------------------

j = 0
init_invest   = 110000
total_profit  = 110000
brokerage_val = 40.0
brokerage_pct = 0.0016
stop_loss_pts = 100
# stop_loss_pct = 75

# ---------------------------------------------------------------------------------------

trade_df = pd.DataFrame(columns=['Entry_Date', 'Entry_Time', 'Entry_Day', 'Exit_Date', 'Exit_Time', 'Exit_Day', 
                                 'Strike', 'Option', 'Lots', 'Entry_Price', 'Exit_Price', 
                                 'StopLoss_%', 'StopLoss_Price', 'StopLoss_Hit', 'SL_Hit_Time', 'SL_Hit_Price',
                                 'Profit', 'Brokerage', 'Net_Profit', 'Cum_Profit'])

convert_dict = {'Entry_Price'   : float,
                'Exit_Price'    : float,
                'StopLoss_%'    : float,
                'StopLoss_Price': float,
                'SL_Hit_Price'  : float,
                'Profit'        : float,
                'Brokerage'     : float,
                'Net_Profit'    : float,
                'Cum_Profit'    : float,
                }

trade_df = trade_df.astype(convert_dict)

trade_df.loc[j] = (['', '', '', '', '', '', '', '', '', '', '', '', 
                    '', '', '', '', '', '', '', str(init_invest)])

j += 1

# ---------------------------------------------------------------------------------------

positions   = []
utility     = Strategies.Utility()
slippage    = 1
start_time  = dt.time(9, 15)
expiry_time = dt.time(15, 15)
prev_signal = ''

Buy_ON      = False
Sell_ON     = False

for index, row in df.iterrows():

    # if row['Signal'] == 'Hold': continue
    # if row['Signal'] == prev_signal: continue
    # if row['time'] == '09:15:00' or row['time'] == '09:30:00': continue    

    if row['Signal_Day'] == 'Buy' and row['Signal_Min'] == 'Buy' and (Buy_ON == False or Sell_ON == True):
        current_signal = 'Buy'
        Buy_ON         = True
        Sell_ON        = False
    elif row['Signal_Day'] == 'Sell' and row['Signal_Min'] == 'Sell' and (Sell_ON == False or Buy_ON == True):
        current_signal = 'Sell'
        Sell_ON        = True
        Buy_ON         = False
    elif row['Signal_Day'] == 'Buy' and row['Signal_Min'] == 'Sell' and Buy_ON == True:
        current_signal = 'Sell'
        Buy_ON         = False
    elif row['Signal_Day'] == 'Sell' and row['Signal_Min'] == 'Buy' and Sell_ON == True:
        current_signal = 'Buy'
        Sell_ON        = False
    else:
        continue

    # current_signal = row['Signal']
    # current_date   = row['date']
    # current_time   = row['time']
    current_date   = row['date_Mins']
    current_time   = row['time_Mins']
    current_spot   = row['Open_Mins']
    # current_spot   = row['Open']

    # date_object = dt.datetime.strptime(current_date, '%Y-%m-%d')
    # day_name = date_object.strftime('%A')

    # if day_name == 'Friday': continue

    strike_offset = 0
    if current_signal == 'Buy' : strike_offset = 200
    if current_signal == 'Sell': strike_offset = -200

    # banknifty_option_2021-01-01
    filename    = 'nifty_option_' + str(current_date) + '.csv'
    file_path   = os.path.join(options_data_files, filename)

    options_df  = get_options_df(file_path)
    strike      = get_atm(symbol=symbol, spot_price=current_spot, offset=strike_offset)

    if current_signal == 'Sell': option = 'CE'
    if current_signal == 'Buy' : option = 'PE'

    if len(positions) > 0:
 
        current_position = positions[0]

        pos_expiry_date = current_position.expiry_date

        if current_date <= pos_expiry_date and current_time <= str(current_position.expiry_time):

            options_df1 = get_options_df_bystrike(options_df, current_date, current_position.option, current_position.strike)

            # if current_position.option == 'PE': option1 = 'CE'
            # if current_position.option == 'CE': option1 = 'PE'
 
            exit_price, expiry_date, order_time = utility.get_option_price(df=options_df1, date=current_date, time1=current_time, 
                                                                           option=current_position.option, strike=current_position.strike,
                                                                           slippage=slippage, action='Buy')
            
            current_position.exit_date  = current_date
            current_position.exit_time  = current_time
            current_position.exit_price = exit_price

            if check_sl_hit(current_position):
                if current_position.exit_date > current_position.sl_date \
                    or (current_position.exit_date == current_position.sl_date 
                        and str(current_position.exit_time) > current_position.sl_time):
                    
                    current_position.exit_date  = current_position.sl_date
                    current_position.exit_time  = current_position.sl_time
                    current_position.exit_price = current_position.sl_price
                else:
                    current_position.sl_hit     = False
                    current_position.sl_date    = ''
                    current_position.sl_time    = ''
                    current_position.sl_price   = 0.0
                    current_position.sl_hit_at  = 0.0                    

            entr_date = dt.datetime.strptime(current_position.entry_date, "%Y-%m-%d")
            entr_day  = entr_date.strftime("%A")

            exit_date = dt.datetime.strptime(current_position.exit_date, "%Y-%m-%d")
            exit_day  = exit_date.strftime("%A")

            total_qty   = current_position.lots * lot_size
            profit      = (current_position.entry_price - current_position.exit_price) * total_qty
            turnover    = (current_position.entry_price + current_position.exit_price) * total_qty
            brokerage   = round((brokerage_pct * turnover) + brokerage_val , 2)
            net_profit  = profit - brokerage
            total_profit += net_profit

            trade_df.loc[j] = ([str(current_position.entry_date), str(current_position.entry_time), str(entr_day), 
                                str(current_position.exit_date), str(current_position.exit_time), str(exit_day),
                                str(current_position.strike), current_position.option, str(current_position.lots), 
                                str(current_position.entry_price), str(current_position.exit_price),
                                str(current_position.stop_loss), str(current_position.sl_price), str(current_position.sl_hit), 
                                current_position.sl_time, str(current_position.sl_hit_at), 
                                str(profit), str(brokerage), str(net_profit), str(total_profit)])
        
            j += 1

            positions = []

        else:

            while pos_expiry_date < current_date:
            
                xdate       = pos_expiry_date
                xfilename   = 'nifty_option_' + str(xdate) + '.csv'
                xfile_path  = os.path.join(options_data_files, xfilename)

                xoptions_df = get_options_df(xfile_path)
                xoptions_df = get_options_df_bystrike(xoptions_df, xdate, current_position.option, current_position.strike)

                # if current_position.option == 'PE': option1 = 'CE'
                # if current_position.option == 'CE': option1 = 'PE'

                xexit_price, xexpiry_date, xorder_time = utility.get_option_price(df=xoptions_df, date=pos_expiry_date, 
                                                                                  time1=expiry_time, option=current_position.option,
                                                                                  strike=current_position.strike, slippage=slippage, action='Buy')
                
                current_position.exit_date  = pos_expiry_date
                current_position.exit_time  = expiry_time
                current_position.exit_price = xexit_price

                if check_sl_hit(current_position):
                    if current_position.exit_date > current_position.sl_date \
                       or (current_position.exit_date == current_position.sl_date 
                           and str(current_position.exit_time) > current_position.sl_time):
                        
                        current_position.exit_date  = current_position.sl_date
                        current_position.exit_time  = current_position.sl_time
                        current_position.exit_price = current_position.sl_price
                    else:
                        current_position.sl_hit     = False
                        current_position.sl_date    = ''
                        current_position.sl_time    = ''
                        current_position.sl_price   = 0.0
                        current_position.sl_hit_at  = 0.0  
                    
                entr_date = dt.datetime.strptime(current_position.entry_date, "%Y-%m-%d")
                entr_day  = entr_date.strftime("%A")

                exit_date = dt.datetime.strptime(current_position.exit_date, "%Y-%m-%d")
                exit_day  = exit_date.strftime("%A")

                total_qty   = current_position.lots * lot_size
                profit      = (current_position.entry_price - current_position.exit_price) * total_qty
                turnover    = (current_position.entry_price + current_position.exit_price) * total_qty
                brokerage   = round((brokerage_pct * turnover) + brokerage_val , 2)
                net_profit  = profit - brokerage
                total_profit += net_profit

                trade_df.loc[j] = ([str(current_position.entry_date), str(current_position.entry_time), str(entr_day), 
                                    str(current_position.exit_date), str(current_position.exit_time), str(exit_day),
                                    str(current_position.strike), current_position.option, str(current_position.lots), 
                                    str(current_position.entry_price), str(current_position.exit_price),
                                    str(current_position.stop_loss), str(current_position.sl_price), str(current_position.sl_hit), 
                                    current_position.sl_time, str(current_position.sl_hit_at), 
                                    str(profit), str(brokerage), str(net_profit), str(total_profit)])
            
                j += 1         

                # ----------------------------------------------------------------------------

                prev_date = dt.datetime.strptime(pos_expiry_date, "%Y-%m-%d")
                next_date = prev_date + timedelta(days=1)

                while True:
                    ndate       = next_date.strftime("%Y-%m-%d")  #dt.datetime.strptime(str(next_date), "%Y-%m-%d")
                    nfilename   = 'nifty_option_' + str(ndate) + '.csv'
                    nfile_path  = os.path.join(options_data_files, nfilename)

                    if os.path.exists(nfile_path):
                        break
                    else:
                        next_date = next_date + timedelta(days=1)
                        
                date_format = '%Y-%m-%d'
                date1        = dt.datetime.strptime(ndate, date_format).date()

                strike_offset1 = 0
                if current_position.option == 'PE' : strike_offset1 = 200
                if current_position.option == 'CE': strike_offset1 = -200

                current_position.strike = utility.get_atm_price(symbol=symbol, date=date1, time=start_time, offset=strike_offset1)

                noptions_df = get_options_df(nfile_path)
                options_df2 = get_options_df_bystrike(noptions_df, ndate, current_position.option, current_position.strike)

                entry_price, expiry_date, order_time = utility.get_option_price(df=options_df2, date=ndate, time1=start_time, 
                                                                                option=current_position.option, strike=current_position.strike, 
                                                                                slippage=slippage)
                
                current_position.entry_date = ndate
                current_position.entry_time = str(start_time)
                current_position.entry_price= entry_price
                current_position.expiry_date= expiry_date

                sl_price = entry_price + stop_loss_pts
                current_position.sl_price = sl_price

                pos_expiry_date = expiry_date

            # ---------------------------------------------------------------------------------------

            if pos_expiry_date >= current_date:

                options_df3 = get_options_df_bystrike(options_df, current_date, current_position.option, current_position.strike)

                # if current_position.option == 'PE': option1 = 'CE'
                # if current_position.option == 'CE': option1 = 'PE'

                exit_price, expiry_date, order_time = utility.get_option_price(df=options_df3, date=current_date, time1=current_time, 
                                                                               option=current_position.option, strike=current_position.strike,
                                                                               slippage=slippage, action='Buy')
                
                current_position.exit_date  = current_date
                current_position.exit_time  = current_time
                current_position.exit_price = exit_price

                if current_position.exit_date < current_position.entry_date \
                   or (current_position.exit_date == current_position.entry_date 
                       and current_position.exit_time < current_position.entry_time):

                    continue
                
                if check_sl_hit(current_position):
                    if current_position.exit_date > current_position.sl_date \
                       or (current_position.exit_date == current_position.sl_date 
                           and str(current_position.exit_time) > current_position.sl_time):
                        
                        current_position.exit_date  = current_position.sl_date
                        current_position.exit_time  = current_position.sl_time
                        current_position.exit_price = current_position.sl_price
                    else:
                        current_position.sl_hit     = False
                        current_position.sl_date    = ''
                        current_position.sl_time    = ''
                        current_position.sl_price   = 0.0
                        current_position.sl_hit_at  = 0.0  

                entr_date = dt.datetime.strptime(current_position.entry_date, "%Y-%m-%d")
                entr_day  = entr_date.strftime("%A")

                exit_date = dt.datetime.strptime(current_position.exit_date, "%Y-%m-%d")
                exit_day  = exit_date.strftime("%A")

                total_qty   = current_position.lots * lot_size
                profit      = (current_position.entry_price - current_position.exit_price) * total_qty
                turnover    = (current_position.entry_price + current_position.exit_price) * total_qty
                brokerage   = round((brokerage_pct * turnover) + brokerage_val , 2)
                net_profit  = profit - brokerage
                total_profit += net_profit

                trade_df.loc[j] = ([str(current_position.entry_date), str(current_position.entry_time), str(entr_day), 
                                    str(current_position.exit_date), str(current_position.exit_time), str(exit_day),
                                    str(current_position.strike), current_position.option, str(current_position.lots), 
                                    str(current_position.entry_price), str(current_position.exit_price),
                                    str(current_position.stop_loss), str(current_position.sl_price), str(current_position.sl_hit), 
                                    current_position.sl_time, str(current_position.sl_hit_at), 
                                    str(profit), str(brokerage), str(net_profit), str(total_profit)])
            
                j += 1

                positions = []
            
            else:
                print('Too Many Days')  

    # ---------------------------------------------------------------------------------------

    if len(positions) == 0:

        if Buy_ON or Sell_ON:

            options_df4 = get_options_df_bystrike(options_df, current_date, option, strike)

            entry_price, expiry_date, order_time = utility.get_option_price(df=options_df4, date=current_date, time1=current_time, 
                                                                            option=option, strike=strike, slippage=slippage)
            
            if current_date <= expiry_date and current_time < str(expiry_time):

                position = Strategies.Position(symbol=symbol, option=option, action=current_signal, id=1, strike_offset=strike_offset, 
                                            lots=1, sl_bool=False, stop_loss=0.0, tgt_bool=False, target=0.0, slippage=slippage,
                                            strike=strike, entry_date=current_date, entry_time=current_time, entry_price=entry_price, 
                                            expiry_date=expiry_date, expiry_time=str(expiry_time))
                
                sl_price = entry_price + stop_loss_pts
                # sl_price = entry_price * (1 + (stop_loss_pct / 100.0)) 
                position.sl_price = sl_price

                positions.append(position)

                # prev_signal = current_signal
    
# ---------------------------------------------------------------------------------------

trade_df.to_csv('DI_STOCH_Min_Hour_Nifty_Options' + '.csv') 
