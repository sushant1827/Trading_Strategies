# Features to Support 
# 1.  Best time to Entry
# 2.  Best time to Exit
# 3.  Best SL for CE and PE
# 4.  Slippage and Brokerage 
# 5.  Total + Days-wise Profit
# 6.  Winning ratio - Win / Loss
# 7.  Max Draw Down - MDD %
# 8.  Consecutive Wins / Losses
# 9.  Best time to Trade - VIX effect'

import pandas as pd
import datetime
import os
import random

folder_path  = r'D:/AlgoTrade/Fyers_AlgoTrade/New_Options_Data_BnF/2021-2024'
files = os.listdir(folder_path)
csv_files = [file for file in files if file.endswith('.csv')] 

k = 0

while k < 10:

    print(k)

    # start_times = [datetime.time(9,30), datetime.time(9,40), datetime.time(9,50), datetime.time(10,00), datetime.time(10,10)]
    # start_times = [datetime.time(9,40), datetime.time(9,45), datetime.time(9,50)]
    # end_times   = [datetime.time(14,40), datetime.time(14,50), datetime.time(15,00), datetime.time(15,10), datetime.time(15,20)]
    # end_times   = [datetime.time(14,40), datetime.time(14,45), datetime.time(14,50), datetime.time(15,00), datetime.time(15,10)]
    # # ce_sl_pcts  = [0.12, 0.17, 0.22, 0.27, 0.32]
    # # pe_sl_pcts  = [0.12, 0.17, 0.22, 0.27, 0.32]
    # ce_sl_pcts  = [0.15, 0.18, 0.22, 0.25, 0.28, 0.30, 0.32]
    # pe_sl_pcts  = [0.15, 0.18, 0.22, 0.25, 0.28, 0.30, 0.32]
    # start_time  = random.choice(start_times)
    # end_time    = random.choice(end_times)
    # ce_sl_pct   = random.choice(ce_sl_pcts) # 25%
    # pe_sl_pct   = random.choice(pe_sl_pcts) # 25%
    # ce_sl_ptss  = [50, 55, 60, 65, 70, 75, 80, 85, 90]
    ce_sl_ptss  = [50, 55, 60, 65, 70]
    # pe_sl_ptss  = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    pe_sl_ptss  = [10, 15, 20, 25, 30]
    ce_sl_pts   = random.choice(ce_sl_ptss) # 25%
    pe_sl_pts   = random.choice(pe_sl_ptss) # 25%
    # start_time  = datetime.time(10,10)
    start_time  = datetime.time(9,50)
    end_time    = datetime.time(14,50)
    # ce_sl_pct   = 0.32 # 25%
    # pe_sl_pct   = 0.17 # 25%
    # ce_sl_pts   = 80 # 25%
    # pe_sl_pts   = 30 # 25%
    lot_size    = 2
    qty_per_lot = 25
    total_qty   = lot_size * qty_per_lot
    slippage    = 0.01 # 1%
    # slippage    = 2 # 1%
    brokerage   = 85
    vix_val     = 17
    strike_off  = 0
    pos_day_cnt = 0
    neg_day_cnt = 0
    mon_profit  = 0.0
    tue_profit  = 0.0
    wed_profit  = 0.0
    thu_profit  = 0.0
    fri_profit  = 0.0
    total_profit= 0.0
    pnl_ratio   = 0.0
    init_amount = 300000
    curr_tpnl   = init_amount
    peak_amount = init_amount
    draw_down   = 0.0
    max_drw_dwn = 0.0
    win_cnt     = 0
    los_cnt     = 0
    max_seq_win = 0
    max_seq_loss= 0
    day_pnl_pct = 0.0
    pnl_pct     = 0.0
    total_brokr = 0.0
    # mtm_loss_val= 5500

    # day_names   = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    # day_names   = ['Tuesday', 'Wednesday', 'Thursday']
    day_names   = ['Tuesday']
    lot_flag    = False

    trades = pd.DataFrame()

    vix_file_path  = r'D:/AlgoTrade/Fyers_AlgoTrade/INDIAVIX_Index/INDIAVIX_INDEX_5_Min.csv'
    df_vix = pd.read_csv(vix_file_path)
    
    df_vix.drop(['Unnamed: 0', 'Volume'], inplace=True, axis=1)
    df_vix['Date'] = pd.to_datetime(df_vix['Date'])
    df_vix.set_index('Date', inplace=True)

    for csv_file in csv_files:

        file_path = os.path.join(folder_path, csv_file)

        df = pd.read_csv(file_path)
        df.drop(['Unnamed: 0', 'volume'], inplace=True, axis=1)

        current_date  = datetime.datetime.strptime(df.iloc[0]['date'], "%Y-%m-%d")
        current_day   = current_date.strftime('%A')
        current_month = int(current_date.strftime('%m'))
        current_year  = int(current_date.strftime('%Y'))

        if current_month >= 9 and current_year >= 2023:
            day_names   = ['Monday', 'Tuesday', 'Wednesday']

        if (current_month >= 7 and current_year >= 2023) and not lot_flag:
            qty_per_lot = 15
            total_qty   = lot_size * qty_per_lot
            brokerage   = 70
            lot_flag    = True

        if current_day not in day_names:
            continue

        df['datetime'] = df['date'] + " " + df['time']
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)

        spotdata = df[df['symbol'] == 'BANKNIFTY']
        exp      = df.iloc[0]['symbol'][9:16]
        td       = {}
        straddle = False
        day_pnl  = 0.0
        # sl_hit   = False
        # curr_mtm_loss = 0.0

        for s in range(len(spotdata)):
            currentcandle = spotdata.iloc[s]

            if currentcandle.name.time() == start_time and not straddle:

                combined_datetime = datetime.datetime.combine(currentcandle.name.date(), start_time)
                formatted_result  = combined_datetime.strftime('%Y-%m-%d %H:%M:%S')
                curr_vix_open     = df_vix.loc[pd.to_datetime(formatted_result), 'Open']

                if curr_vix_open > vix_val:
                    break

                # print('Placing Straddle!')
                cst = currentcandle['open']
                cst = int(round(cst/100, 0) * 100) + strike_off
                opdata = {'CE' : df[df['symbol'] == 'BANKNIFTY' + exp + str(cst) + 'CE'],
                          'PE' : df[df['symbol'] == 'BANKNIFTY' + exp + str(cst) + 'PE']}
                
                ce_sellprice = opdata['CE'].loc[currentcandle.name]
                pe_sellprice = opdata['PE'].loc[currentcandle.name]

                # td = {'CE' : {'sellprice' : ce_sellprice['open'] - (ce_sellprice['open'] * slippage), 'symbol' : ce_sellprice['symbol'],
                #               'selltime' : currentcandle.name.time(), 'date' : currentcandle.name.date(),
                #               'sl' : ce_sellprice['open'] + (ce_sellprice['open'] * ce_sl_pct),
                #               'pos' : True, 'strike' : cst, 'qty' : total_qty},
                #       'PE' : {'sellprice' : pe_sellprice['open'] - (pe_sellprice['open'] * slippage), 'symbol' : pe_sellprice['symbol'],
                #               'selltime' : currentcandle.name.time(), 'date' : currentcandle.name.date(),
                #               'sl' : pe_sellprice['open'] + (pe_sellprice['open'] * pe_sl_pct),
                #               'pos' : True, 'strike' : cst, 'qty' : total_qty}
                #      }
                
                td = {'CE' : {'sellprice' : ce_sellprice['open'] - (ce_sellprice['open'] * slippage), 'symbol' : ce_sellprice['symbol'],
                              'selltime' : currentcandle.name.time(), 'date' : currentcandle.name.date(),
                              'sl' : ce_sellprice['open'] - (ce_sellprice['open'] * slippage) + ce_sl_pts,
                              'pos' : True, 'strike' : cst, 'qty' : total_qty},
                      'PE' : {'sellprice' : pe_sellprice['open'] - (pe_sellprice['open'] * slippage), 'symbol' : pe_sellprice['symbol'],
                              'selltime' : currentcandle.name.time(), 'date' : currentcandle.name.date(),
                              'sl' : pe_sellprice['open'] - (pe_sellprice['open'] * slippage) + pe_sl_pts,
                              'pos' : True, 'strike' : cst, 'qty' : total_qty}
                     }
                
                straddle = True

            if straddle:

                # if currentcandle.name in opdata['CE'].index and currentcandle.name in opdata['PE'].index:

                #     if opdata['CE'].loc[currentcandle.name]['high'] >= td['CE']['sl'] and td['CE']['pos']:
                #         td['PE']['sl'] = td['PE']['sellprice']

                #     if opdata['PE'].loc[currentcandle.name]['high'] >= td['PE']['sl'] and td['PE']['pos']:
                #         td['CE']['sl'] = td['CE']['sl']

                for t in ['CE', 'PE']:

                    if currentcandle.name in opdata['CE'].index and currentcandle.name in opdata['PE'].index:

                        if opdata[t].loc[currentcandle.name]['high'] >= td[t]['sl'] and td[t]['pos']:

                            td[t]['buyprice']   = td[t]['sl']
                            td[t]['buyday']     = currentcandle.name.date().strftime('%A')
                            td[t]['buytime']    = currentcandle.name.time()
                            td[t]['reason']     = 'SL Hit'
                            td[t]['pnl']        = ((td[t]['sellprice'] - td[t]['buyprice']) * td[t]['qty']) - brokerage
                            td[t]['pos']        = False

                            day_pnl     += td[t]['pnl']
                            total_brokr += brokerage

                            new_row_df  = pd.DataFrame([td[t]])
                            trades      = pd.concat([trades, new_row_df], ignore_index=True)

                            # sl_hit      = True
                            # curr_mtm_loss += td[t]['pnl'] + brokerage

                        if currentcandle.name.time() == end_time and td[t]['pos']:

                            td[t]['buyprice']   = opdata[t].loc[currentcandle.name]['open'] + (opdata[t].loc[currentcandle.name]['open'] * slippage)
                            td[t]['buyday']     = currentcandle.name.date().strftime('%A')
                            td[t]['buytime']    = currentcandle.name.time()
                            td[t]['reason']     = 'Time Up'
                            td[t]['pnl']        = ((td[t]['sellprice'] - td[t]['buyprice']) * td[t]['qty']) - brokerage
                            td[t]['pos']        = False

                            day_pnl     += td[t]['pnl']
                            total_brokr += brokerage

                            new_row_df  = pd.DataFrame([td[t]])
                            trades      = pd.concat([trades, new_row_df], ignore_index=True)

                        # if (sl_hit and td[t]['pos']) and currentcandle.name.time() < end_time:

                        #     curr_loss = ((td[t]['sellprice'] - opdata[t].loc[currentcandle.name]['open']) * td[t]['qty'])

                        #     if (curr_mtm_loss + curr_loss) < (0.0 - mtm_loss_val):
                                
                        #         td[t]['buyprice']   = opdata[t].loc[currentcandle.name]['open'] + (opdata[t].loc[currentcandle.name]['open'] * slippage)
                        #         td[t]['buyday']     = currentcandle.name.date().strftime('%A')
                        #         td[t]['buytime']    = currentcandle.name.time()
                        #         td[t]['reason']     = 'MTM SL Hit'
                        #         td[t]['pnl']        = ((td[t]['sellprice'] - td[t]['buyprice']) * td[t]['qty']) - brokerage
                        #         td[t]['pos']        = False

                        #         day_pnl += td[t]['pnl']
                        #         total_brokr += brokerage

                        #         new_row_df = pd.DataFrame([td[t]])
                        #         trades = pd.concat([trades, new_row_df], ignore_index=True)


        if straddle:
            curr_tpnl += day_pnl

            if curr_tpnl < peak_amount:
                draw_down = curr_tpnl - peak_amount

            if draw_down < max_drw_dwn:
                max_drw_dwn = round(draw_down, 2)
                mdd_pct     = round(((max_drw_dwn / peak_amount) * 100), 2)

            if curr_tpnl > peak_amount:
                peak_amount = curr_tpnl

            if day_pnl >= 0.0:
                pos_day_cnt += 1
                los_cnt = 0
                win_cnt += 1
                if win_cnt > max_seq_win:
                    max_seq_win = win_cnt

            if day_pnl < 0.0:
                neg_day_cnt += 1
                win_cnt = 0
                los_cnt +=1
                if los_cnt > max_seq_loss:
                    max_seq_loss = los_cnt


    proft_ratio = round(((pos_day_cnt / (pos_day_cnt + neg_day_cnt)) * 100), 2)
    total_pnl   = round(trades['pnl'].sum(), 2)

    # pnl_pct     = round(((total_pnl / init_amount) * 100), 2)
    # pnl_pct     = round((pnl_pct * 100), 2)
    result      = trades.groupby('buyday')['pnl'].sum()

    # mon_profit  = round(result['Monday'], 2)
    tue_profit  = round(result['Tuesday'], 2)
    # wed_profit  = round(result['Wednesday'], 2)
    # thu_profit  = round(result['Thursday'], 2)
    # fri_profit  = round(result['Friday'], 2)

    # total_pnl   = round(tue_profit + wed_profit + thu_profit, 2)

    avg_profit  = round((total_pnl / (pos_day_cnt + neg_day_cnt)), 2)

    if total_pnl > 0.0:# total_profit: # and proft_ratio > pnl_ratio:

        print('==================================')
        # print(f"Initial Amnt    : {init_amount}")
        print(f"India VIX       : {vix_val}")
        print(f"Start Time      : {start_time}")
        print(f"End Time        : {end_time}")
        # print(f"CE SL %      : {ce_sl_pct}")
        # print(f"PE SL %      : {pe_sl_pct}")
        print(f"CE SL Pts       : {ce_sl_pts}")
        print(f"PE SL Pts       : {pe_sl_pts}")
        print('----------------------------------')
        print(f"Total Days      : {pos_day_cnt + neg_day_cnt}")
        print(f"Profit Days     : {pos_day_cnt}")
        print(f"Loss Days       : {neg_day_cnt}")
        print('----------------------------------')
        print(f"Win Ratio       : {proft_ratio}")
        print(f"Total Profit    : {total_pnl}")
        # print(f"Profit Pct      : {pnl_pct}%")
        print(f"Average Profit  : {avg_profit}")
        print('----------------------------------')
        print(f"Max Win Streak  : {max_seq_win}")
        print(f"Max Loss Streak : {max_seq_loss}")
        print('----------------------------------')
        print(f"Total Brokerage : {total_brokr}")
        print(f"Max Draw Down   : {max_drw_dwn}")
        print(f"MDD Pct         : {mdd_pct}%")
        print('----------------------------------')
        print(f"Monday Pnl      : {mon_profit}")
        print(f"Tuesday Pnl     : {tue_profit}")
        print(f"Wednesday Pnl   : {wed_profit}")
        print(f"Thursday Pnl    : {thu_profit}")
        print(f"Friday Pnl      : {fri_profit}")
        print('==================================')

    # trades.to_csv('Straddle_950_1450_BnF.csv')

    k += 1

# ==================================
# 2021-2023
# 2 lot - 15/25
# ==================================
# India VIX       : 17
# Start Time      : 09:50:00
# End Time        : 14:50:00
# CE SL Pts       : 80
# PE SL Pts       : 30
# Slippage        : 1%
# ----------------------------------
# Total Days      : 220
# Profit Days     : 165
# Loss Days       : 55
# ----------------------------------
# Win Ratio       : 75.0
# Total Profit    : 263883.59
# Average Profit  : 1199.47
# ----------------------------------
# Max Win Streak  : 21
# Max Loss Streak : 3
# ----------------------------------
# Total Brokerage : 35930.0
# Max Draw Down   : -15621.25
# MDD Pct         : -2.86%
# ----------------------------------
# Monday Pnl      : -1542.0
# Tuesday Pnl     : 62579.8
# Wednesday Pnl   : 85230.71
# Thursday Pnl    : 117615.08
# Friday Pnl      : 0.0
# ==================================

# ==================================
# 2021
# 2 lot - 15/25
# ==================================
# India VIX       : 17
# Start Time      : 09:50:00
# End Time        : 14:50:00
# CE SL Pts       : 80
# PE SL Pts       : 30
# Slippage        : 1%
# ----------------------------------
# Total Days      : 69
# Profit Days     : 54
# Loss Days       : 15
# ----------------------------------
# Win Ratio       : 78.26
# Total Profit    : 112551.48
# Average Profit  : 1631.18
# ----------------------------------
# Max Win Streak  : 21
# Max Loss Streak : 2
# ----------------------------------
# Total Brokerage : 11730.0
# Max Draw Down   : -11416.28
# MDD Pct         : -2.74%
# ----------------------------------
# Monday Pnl      : 0.0
# Tuesday Pnl     : 39100.05
# Wednesday Pnl   : 41916.93
# Thursday Pnl    : 31534.5
# Friday Pnl      : 0.0
# ==================================

# ==================================
# 2022
# 2 lot - 15/25
# ==================================
# India VIX       : 17
# Start Time      : 09:50:00
# End Time        : 14:50:00
# CE SL Pts       : 80
# PE SL Pts       : 30
# Slippage        : 1%
# ----------------------------------
# Total Days      : 31
# Profit Days     : 24
# Loss Days       : 7
# ----------------------------------
# Win Ratio       : 77.42
# Total Profit    : 52418.27
# Average Profit  : 1690.91
# ----------------------------------
# Max Win Streak  : 9
# Max Loss Streak : 3
# ----------------------------------
# Total Brokerage : 5270.0
# Max Draw Down   : -5670.0
# MDD Pct         : -1.89%
# ----------------------------------
# Monday Pnl      : 0.0
# Tuesday Pnl     : 3442.25
# Wednesday Pnl   : 18057.0
# Thursday Pnl    : 30919.02
# Friday Pnl      : 0.0
# ==================================
    
# ==================================
# 2023
# 2 lot - 15/25
# ==================================
# India VIX       : 17
# Start Time      : 09:50:00
# End Time        : 14:50:00
# CE SL Pts       : 80
# PE SL Pts       : 30
# Slippage        : 1%
# ----------------------------------
# Total Days      : 120
# Profit Days     : 87
# Loss Days       : 33
# ----------------------------------
# Win Ratio       : 72.5
# Total Profit    : 98913.84
# Average Profit  : 824.28
# ----------------------------------
# Max Win Streak  : 13
# Max Loss Streak : 3
# ----------------------------------
# Total Brokerage : 18930.0
# Max Draw Down   : -15621.25
# MDD Pct         : -4.11%
# ----------------------------------
# Monday Pnl      : 0.0
# Tuesday Pnl     : 20037.5
# Wednesday Pnl   : 25256.79
# Thursday Pnl    : 55161.55
# Friday Pnl      : 0.0
# ==================================