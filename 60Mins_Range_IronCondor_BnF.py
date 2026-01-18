import pandas as pd
import os
import datetime
from datetime import timedelta

import warnings
warnings.filterwarnings('ignore')


def fill_missing_time_intervals(df, time_col='time', start_time='09:15:00', end_time='15:29:00', freq='1T'):
    """
    Fills missing time intervals in a DataFrame with linear interpolation.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing a time column and other data columns.
    time_col (str): The name of the time column in the DataFrame (default is 'time').
    start_time (str): The starting time in the range (default is '09:15:00').
    end_time (str): The ending time in the range (default is '15:29:00').
    freq (str): The frequency of time intervals (default is 1 minute '1T').
    
    Returns:
    pd.DataFrame: A new DataFrame with missing time intervals filled and values interpolated.
    """

    df[time_col] = pd.to_datetime(df[time_col]).dt.time
    
    # Create a complete time range from start_time to end_time with 1-minute intervals
    time_range = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # Create a new DataFrame with the complete time range and convert to time only
    complete_df = pd.DataFrame(time_range, columns=[time_col])
    complete_df[time_col] = complete_df[time_col].dt.time
    
    merged_df = pd.merge(complete_df, df, on=time_col, how='left')
    
    # Fill missing values using linear interpolation
    for col in df.columns:
        if col != time_col:
            merged_df[col] = merged_df[col].interpolate(method='linear')

    merged_df.drop(columns=['Unnamed: 0'], inplace=True)

    # print(merged_df.head())
    # print(merged_df.tail())

    # Return the new DataFrame
    return merged_df

# =========================================================

def calculate_brokerage(total_sell_price, total_buy_price):

    brokerage       = 40
    # STT_pct         = 0.0625 / 100.0
    STT_pct         = 0.1 / 100.0 # 0.1 Oct 2024
    transaction_pct = 0.0495 / 100.0
    GST_pct         = 18 / 100.0
    sebi_charge_pct = 10 / 10000000
    stamp_duty_pct  = 0.003 / 100
    
    STT                 = round((total_sell_price * 2 * STT_pct), 2)
    transaction_charges = round(((total_sell_price + total_buy_price) * 2 * transaction_pct), 2)
    sebi_charges        = round(((total_sell_price + total_buy_price) * 2 * sebi_charge_pct), 2)
    GST                 = round(((brokerage + sebi_charges + transaction_charges) * GST_pct), 2)
    stamp_duty          = round((total_buy_price * 2 * stamp_duty_pct), 2)

    total_brokerage = round((brokerage + STT + transaction_charges + sebi_charges + GST + stamp_duty), 2)

    return total_brokerage

# =========================================================

# Load index spot price data (1-minute OHLC data)

# spot_file = 'NIFTY50_INDEX_60_Min.csv'
# spot_path = 'D:/AlgoTrade/Fyers_AlgoTrade/Fyers_Data/Nifty50_Index'

spot_file = 'NIFTYBANK_INDEX_60_Min.csv'
spot_path = 'D:/AlgoTrade/Fyers_AlgoTrade/Fyers_Data/BankNifty_Index'

spot_df = pd.read_csv(os.path.join(spot_path, spot_file))
spot_df = spot_df.iloc[-1203:-42]
# spot_df = spot_df.iloc[-350:-42]
spot_df.drop(columns=['Unnamed: 0', 'Volume'], inplace=True)
spot_df.reset_index(inplace=True, drop=True)

vix_file = 'INDIAVIX_INDEX_60_Min.csv'
vix_path = 'D:/AlgoTrade/Fyers_AlgoTrade/Fyers_Data/INDIAVIX_Index'
vix_df = pd.read_csv(os.path.join(vix_path, vix_file))
vix_df = vix_df.iloc[-1203:-42]
# vix_df = vix_df.iloc[-350:-42]
vix_df.drop(columns=['Unnamed: 0', 'Volume'], inplace=True)
vix_df.reset_index(inplace=True, drop=True)

# print(spot_df.head())
# print(spot_df.tail())

# Convert 'Date' column to datetime format
spot_df['Date'] = pd.to_datetime(spot_df['Date'])
vix_df['Date'] = pd.to_datetime(vix_df['Date'])

# Load options daily data (1-minute OHLC data)
options_path = 'D:/AlgoTrade/Fyers_AlgoTrade/Fyers_Data/New_Options_Data_BnF/2024'
# options_path = 'D:/AlgoTrade/Fyers_AlgoTrade/Fyers_Data/New_Options_Data_Nfty/2024'

# Set start and end date based on the index spot data
start_date = spot_df['Date'].min().floor('D')  # Earliest available date (start of the day)
end_date = spot_df['Date'].max().floor('D')  # Latest available date (end of the day)

# Define variables for strategy
initial_amount      = 300000
total_brokerage     = 0
net_profit          = 0
max_profit          = 0
max_loss            = 0
profit_count        = 0
loss_count          = 0
consecutive_wins    = 0
consecutive_losses  = 0
max_drawdown        = 0
win_streak          = 0
loss_streak         = 0
lot_size            = 90
stop_loss           = 1.40 # by Percentage
# stop_loss           = 30 # by Point
slippage            = 1.005
slippage_H          = 0.995
MTM_loss            = -14500
max_vix             = 17

trade_time   = datetime.time(10, 15) #'09:30:00'
closing_time = datetime.time(14, 45) #'15:29:00'

# Define the columns for your CSV file
columns = [
    'Date', 'Day_Name', 'Start_Time', 'Spot_Strike', 'CE_Strike', 'PE_Strike', 
    'CE_Buy_Strike', 'PE_Buy_Strike', 'Lot_Size', 'CE_Sell_Price', 'PE_Sell_Price', 
    'CE_Buy_Price_H', 'PE_Buy_Price_H', 
    'Margin', 'CE_SL_Price', 'PE_SL_Price', 'CE_SL_Diff', 'PE_SL_Diff', 'CE_SL_Hit', 'PE_SL_Hit', 
    'CE_SL_Time', 'PE_SL_Time', 'End_Time', 'CE_Buy_Price', 'PE_Buy_Price', 
    'CE_Sell_Price_H', 'PE_Sell_Price_H',
    'Gross_Profit', 'Day_Brokerage', 'Day_Profit', 'Net_Profit'
]

df = pd.DataFrame(columns=columns)

# Define buy/sell logic based on the strategy
for current_day in pd.date_range(start=start_date, end=end_date):

    day_name = current_day.day_name()

    # if day_name == 'Thursday':
    #     stop_loss = 1.4
    # else:
    #     stop_loss = stop_loss + 0.025        

    time_delta = timedelta(hours=trade_time.hour-1, minutes=trade_time.minute, seconds=trade_time.second) # removed 1 hr to get spot time candle
    current_day1 = current_day + time_delta

    spot_day = spot_df[(spot_df['Date'] == current_day1)]

    if spot_day.empty:
        continue
    
    # Sell Strikes

    CE_spot_price = spot_day.iloc[0]['High']    
    CE_strike = round(CE_spot_price / 100) * 100

    PE_spot_price = spot_day.iloc[0]['Low']    
    PE_strike = round(PE_spot_price / 100) * 100

    # Buy Strikes

    CE_Buy_strike_H = CE_strike + 500
    PE_Buy_strike_H = PE_strike - 500

    # Spot Strike

    spot_time_delta = timedelta(hours=trade_time.hour, minutes=trade_time.minute, seconds=trade_time.second) # removed 1 hr to get spot time candle
    spot_current_day = current_day + spot_time_delta

    vix_day = vix_df[(vix_df['Date'] == spot_current_day)]

    if not vix_day.empty:
        if vix_day.iloc[0]['Open'] > max_vix:
            continue

    day_spot = spot_df[(spot_df['Date'] == spot_current_day)]    

    if day_spot.empty:
        spot_price = 0.0
    else:
        spot_price = day_spot.iloc[0]['Open']

    spot_strike = round(spot_price / 100) * 100
   

    # CE_strike = spot_strike + 200
    # PE_strike = spot_strike - 200

    # if (CE_strike - spot_strike) > (spot_strike - PE_strike):
    #     PE_strike = spot_strike - (CE_strike - spot_strike)

    # if (CE_strike - spot_strike) < (spot_strike - PE_strike):
    #     CE_strike = spot_strike + (spot_strike - PE_strike)

    # print(current_day)
    # print(f'CE: {CE_strike} PE: {PE_strike}')

    options_file = f'banknifty_option_{current_day.strftime("%Y-%m-%d")}.csv'
    # options_file = f'nifty_option_{current_day.strftime("%Y-%m-%d")}.csv'
    options_file_path = os.path.join(options_path, options_file)
    
    if os.path.exists(options_file_path):

        options_df = pd.read_csv(options_file_path)

        # Sell Legs

        call_sell_data = options_df[options_df['symbol'].str.contains(f'{CE_strike}CE')]
        put_sell_data  = options_df[options_df['symbol'].str.contains(f'{PE_strike}PE')]

        call_sell_data = fill_missing_time_intervals(call_sell_data)
        put_sell_data  = fill_missing_time_intervals(put_sell_data)

        call_sell_data['time'] = pd.to_datetime(call_sell_data['time'], format='%H:%M:%S').dt.time
        put_sell_data['time']  = pd.to_datetime(put_sell_data['time'], format='%H:%M:%S').dt.time

        # Buy Legs

        call_buy_data_H = options_df[options_df['symbol'].str.contains(f'{CE_Buy_strike_H}CE')]
        put_buy_data_H  = options_df[options_df['symbol'].str.contains(f'{PE_Buy_strike_H}PE')]

        call_buy_data_H = fill_missing_time_intervals(call_buy_data_H)
        put_buy_data_H  = fill_missing_time_intervals(put_buy_data_H)

        call_buy_data_H['time'] = pd.to_datetime(call_buy_data_H['time'], format='%H:%M:%S').dt.time
        put_buy_data_H['time']  = pd.to_datetime(put_buy_data_H['time'], format='%H:%M:%S').dt.time

        call_sell_price     = 0.0
        put_sell_price      = 0.0
        call_sell_SL_price  = 0.0
        put_sell_SL_price   = 0.0
        call_buy_price      = 0.0
        put_buy_price       = 0.0

        call_buy_price_H    = 0.0
        put_buy_price_H     = 0.0
        call_sell_price_H   = 0.0
        put_sell_price_H    = 0.0

        total_buy_price     = 0.0
        total_sell_price    = 0.0
        day_brokerage       = 0.0

        original_call_sell_price = 0.0
        original_put_sell_price  = 0.0

        position   = False
        CE_SL_Hit  = False
        PE_SL_Hit  = False
        H_SL_Hit   = False
        MTM_loss_hit = False

        CE_SL_Time = datetime.time(0, 0)
        PE_SL_Time = datetime.time(0, 0)

        # print(call_sell_data.head())
        # print(call_sell_data.tail())

        for i in range(len(call_sell_data)):

            if position == True:

                if call_sell_data['time'].iloc[i] > trade_time and call_sell_data['time'].iloc[i] < closing_time:

                    if MTM_loss_hit == False:

                        if CE_SL_Hit == False:
                            call_buy_price    = call_sell_data['high'].iloc[i]

                            if not H_SL_Hit:
                                put_sell_price_H  = put_buy_data_H['low'].iloc[i]

                        if PE_SL_Hit == False:
                            put_buy_price  = put_sell_data['high'].iloc[i]

                            if not H_SL_Hit:
                                call_sell_price_H = call_buy_data_H['low'].iloc[i]

                        # mtm_value = ((call_sell_price - call_buy_price) + (put_sell_price - put_buy_price)) * lot_size

                        mtm_value = (((call_sell_price - call_buy_price) + (put_sell_price - put_buy_price)) 
                                     + ((put_sell_price_H - put_buy_price_H) + (call_sell_price_H - call_buy_price_H))) * lot_size

                        if mtm_value <= MTM_loss:
                            CE_SL_Hit = True
                            PE_SL_Hit = True
                            MTM_loss_hit = True
                            H_SL_Hit = True

                            call_sell_price_H = call_buy_data_H['low'].iloc[i]
                            put_sell_price_H  = put_buy_data_H['low'].iloc[i]

                    if CE_SL_Hit == False:

                        call_buy_price = call_sell_data['high'].iloc[i]

                        if call_buy_price >= call_sell_SL_price:
                            call_buy_price = call_sell_SL_price * slippage
                            CE_SL_Time = call_sell_data['time'].iloc[i]
                            CE_SL_Hit = True
                            put_sell_SL_price = original_put_sell_price
                            
                            put_sell_price_H  = put_buy_data_H['low'].iloc[i]

                            if not H_SL_Hit:
                                call_sell_price_H = call_buy_data_H['low'].iloc[i]
                                H_SL_Hit = True


                    if PE_SL_Hit == False:

                        put_buy_price  = put_sell_data['high'].iloc[i]

                        if put_buy_price >= put_sell_SL_price:
                            put_buy_price = put_sell_SL_price * slippage
                            PE_SL_Time = call_sell_data['time'].iloc[i]
                            PE_SL_Hit = True
                            call_sell_SL_price = original_call_sell_price

                            call_sell_price_H = call_buy_data_H['low'].iloc[i]                            

                            if not H_SL_Hit:
                                put_sell_price_H  = put_buy_data_H['low'].iloc[i]
                                H_SL_Hit = True



                if call_sell_data['time'].iloc[i] == closing_time:

                    if CE_SL_Hit == False:
                        call_buy_price = call_sell_data['open'].iloc[i]

                        # if not H_SL_Hit:
                        put_sell_price_H  = put_buy_data_H['low'].iloc[i]

                        position = False

                        # if not H_SL_Hit:
                            # call_sell_price_H = call_buy_data_H['low'].iloc[i]

                    if PE_SL_Hit == False:
                        put_buy_price  = put_sell_data['open'].iloc[i]

                        # if not H_SL_Hit:
                        call_sell_price_H = call_buy_data_H['low'].iloc[i]

                        position = False

                        # if not H_SL_Hit:
                            # put_sell_price_H  = put_buy_data_H['low'].iloc[i]
                
                    # print(current_day)
                    # print(f'CE_Price@Buy: {call_buy_price} PE_Price@Buy: {put_buy_price}')

                    total_buy_price = ((call_buy_price + put_buy_price) + (call_sell_price_H + put_sell_price_H)) * lot_size

                    # if position == False:
                    final_trade_value = (((call_sell_price - call_buy_price) + (put_sell_price - put_buy_price)) 
                                         + ((put_sell_price_H - put_buy_price_H) + (call_sell_price_H - call_buy_price_H))) * lot_size
                    # print(f'Final trade value: {final_trade_value}')

                    # if MTM_loss_hit == True:
                        # final_trade_value = MTM_loss

                    day_brokerage = calculate_brokerage(total_sell_price, total_buy_price)

                    # Calculate net profit after brokerage
                    day_profit = final_trade_value - day_brokerage

                    # Update total net profit
                    net_profit += day_profit

                    total_brokerage += day_brokerage
                    
                    # Track max profit and max loss
                    max_profit = max(max_profit, day_profit)
                    max_loss = min(max_loss, day_profit)
                    
                    # Track consecutive wins and losses
                    if day_profit > 0:
                        profit_count += 1
                        win_streak += 1
                        loss_streak = 0
                        consecutive_wins = max(consecutive_wins, win_streak)
                    else:
                        loss_count += 1
                        loss_streak += 1
                        win_streak = 0
                        consecutive_losses = max(consecutive_losses, loss_streak)
                    
                    # Track drawdown (difference between max profit and net profit)
                    drawdown = max_profit - net_profit
                    max_drawdown = max(max_drawdown, drawdown)

                    position = False

                    # print('**************************')

                    new_row  = {
                        'Date'          : call_sell_data['date'].iloc[0], #'2024-09-14',
                        'Day_Name'      : day_name, #'Wednesday',
                        'Start_Time'    : trade_time, #'09:15',
                        'Spot_Strike'   : spot_strike, #50000,
                        'CE_Strike'     : CE_strike, #51000,
                        'PE_Strike'     : PE_strike, #50000,
                        'CE_Buy_Strike' : CE_Buy_strike_H, 
                        'PE_Buy_Strike' : PE_Buy_strike_H,
                        'Lot_Size'      : lot_size, #30
                        'CE_Sell_Price' : call_sell_price, #350,
                        'PE_Sell_Price' : put_sell_price, #280,
                        'CE_Buy_Price_H': call_buy_price_H, 
                        'PE_Buy_Price_H': put_buy_price_H,
                        'Margin'        : initial_amount,
                        'CE_SL_Price'   : call_sell_SL_price, #400,
                        'PE_SL_Price'   : put_sell_SL_price, #230,
                        'CE_SL_Diff'    : call_sell_SL_diff, #400,
                        'PE_SL_Diff'    : put_sell_SL_diff, #230,
                        'CE_SL_Hit'     : CE_SL_Hit,
                        'PE_SL_Hit'     : PE_SL_Hit,
                        'CE_SL_Time'    : CE_SL_Time,
                        'PE_SL_Time'    : PE_SL_Time,
                        'End_Time'      : closing_time, #'15:15',
                        'CE_Buy_Price'  : call_buy_price,
                        'PE_Buy_Price'  : put_buy_price, #200,
                        'CE_Sell_Price_H' : call_sell_price_H, 
                        'PE_Sell_Price_H' : put_sell_price_H,
                        'Gross_Profit'  : final_trade_value,
                        'Day_Brokerage' : day_brokerage,
                        'Day_Profit'    : day_profit, #5000,                        
                        'Net_Profit'    : initial_amount + net_profit                        
                    }

                    new_row_df = pd.DataFrame([new_row])
                    df = pd.concat([df, new_row_df], ignore_index=True)

                    break

            
            if position == False and call_sell_data['time'].iloc[i] == trade_time:

                call_sell_price     = call_sell_data['open'].iloc[i]
                call_sell_price     = round((call_sell_price * slippage), 2)
                call_sell_SL_price  = call_sell_price * stop_loss # by percentage
                # call_sell_SL_price  = call_sell_price + stop_loss # by points
                call_sell_SL_diff   = call_sell_SL_price - call_sell_price
                original_call_sell_price = call_sell_price

                call_buy_price_H    = call_buy_data_H['open'].iloc[i]
                call_buy_price_H    = round((call_buy_price_H * slippage_H), 2)

                put_sell_price      = put_sell_data['open'].iloc[i]
                put_sell_price      = round((put_sell_price * slippage), 2)
                put_sell_SL_price   = put_sell_price * stop_loss # by percentage
                # put_sell_SL_price   = put_sell_price + stop_loss # by points
                put_sell_SL_diff    = put_sell_SL_price - put_sell_price
                original_put_sell_price = put_sell_price

                put_buy_price_H     = put_buy_data_H['open'].iloc[i]
                put_buy_price_H     = round((put_buy_price_H * slippage_H), 2)
                
                # print(current_day)
                # print(f'CE_Price@Sell: {call_sell_price} PE_Price@Sell: {put_sell_price}')

                total_sell_price  = ((call_sell_price + put_sell_price) + (call_buy_price_H + put_buy_price_H)) * lot_size
                # total_buy_price_H = ((call_buy_price_H + put_buy_price_H) * lot_size)

                position = True

# =========================================================

# Output the final results

print(f"Initial amount              : {initial_amount}")
print(f"Total profit                : {net_profit:.2f}")
print(f"Total brokerage             : {total_brokerage:.2f}")
print(f"Net profit (after brokerage): {net_profit - total_brokerage:.2f}")
print(f"Net profit in percent       : {(net_profit - total_brokerage) / initial_amount * 100:.2f}%")
print(f"Max profit                  : {max_profit:.2f}")
print(f"Max loss                    : {max_loss:.2f}")
print(f"Profit count                : {profit_count}")
print(f"Loss count                  : {loss_count}")
print(f"Profit to loss ratio        : {profit_count / loss_count if loss_count > 0 else 'N/A':.2f}")
print(f"Max drawdown                : {max_drawdown:.2f}")
print(f"Consecutive wins            : {consecutive_wins}")
print(f"Consecutive losses          : {consecutive_losses}")
print(f"Win percentage              : {profit_count / (profit_count + loss_count) * 100:.2f}%")
print(f"Loss percentage             : {loss_count / (profit_count + loss_count) * 100:.2f}%")

# =========================================================

# Save the DataFrame to a CSV file

df.to_csv('BNF_trading_data_IronCondor.csv', index=False)
print("CSV file created and data saved successfully.")

# =========================================================

# Try different timing for end - Done
# Try different stop_loss values - Done
# Try different lot_size values - Done
# Try Reducing stop_loss percentage - Day till expiry - Done
# Try Nifty - Done
# Try closer strike - same as other strike - Done
# Try open and close instead high and low - Done

# Try 30 and 45 mins range
# Calculate Margin
# Try using buy legs also - Iron Condor - along with SL
# No positions when Vix is above 15-30
# Check for Strike difference Spot to CE and PE
# Try using premium close to value or delta value
# Check for BankNifty move from 10.15 - up and down on historical data

# =========================================================

# BankNifty Results:

# vix = 17 / SL 40 / MTM = -4000 / slippage = 1.005 / 
# lot_size = 30 / start_time = 10:15 / end_time = 14:45

# Initial amount              : 300000
# Total profit                : 91261.12
# Total brokerage             : 15345.12
# Net profit (after brokerage): 75916.00
# Net profit in percent       : 25.31%
# Max profit                  : 8126.73
# Max loss                    : -4901.60
# Profit count                : 82
# Loss count                  : 60
# Profit to loss ratio        : 1.37
# Max drawdown                : 8225.38
# Consecutive wins            : 6
# Consecutive losses          : 5
# Win percentage              : 57.75%
# Loss percentage             : 42.25%

# =========================================================