import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#define starting variables
STARTING_BALANCE = 100000
down_days = 1

# Data Configuration
csv_path = 'Nifty50_Index/NIFTY50_INDEX_30_Min.csv'

price = pd.read_csv(csv_path)
price = price.drop(columns=['Unnamed: 0', 'Volume'], errors='ignore')
price = price.rename(columns={'Date': 'DateTime'})
price = price.set_index('DateTime')
price.index = pd.to_datetime(price.index)
price = price[~price.index.duplicated(keep='first')]


#calculate return and balance
price['oc'] = price.Close / price.Open
price['cc'] = price.Close / price.Close.shift(1)
price.cc.iat[0] = 1
price['Bench_Bal'] = STARTING_BALANCE * price.cc.cumprod()


#calculate benchmark drawdown
price['Bench_Peak'] = price.Bench_Bal.cummax()
price['Bench_DD'] = price.Bench_Bal - price.Bench_Peak

bench_dd = round(((price.Bench_DD / price.Bench_Peak).min() * 100), 2)

print(bench_dd)

#calculate additional columns for strategy

#check if today is a down day
price['Down'] = price.oc < 1

#count consecutive down days
#https://stackoverflow.com/questions/27626542/counting-consecutive-positive-value-in-python-array
down = price['Down']
price['Consecutive'] = down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)

# print(price.tail())

#identify entries and allocate trading fees
price['Long'] = price.Consecutive >= down_days

#calculate system return and balance
price['Sys_Ret'] = np.where(price.Long.shift(1) == True, price.cc, 1)
price['Sys_Bal'] = STARTING_BALANCE * price.Sys_Ret.cumprod()

print(price.tail())

#plot results
plt.plot(price.Bench_Bal)
plt.plot(price.Sys_Bal)

plt.show()


# print(price.head(10))
# print(price.shape)

#plot chart
# plt.plot(df_original.Close)
# plt.show()

# #calculate benchmark return and balance
# price['Return'] = price.Close / price.Close.shift(1)
# price.Return.iat[0] = 1
# price['Bench_Bal'] = STARTING_BALANCE * price.Return.cumprod()

# # print(price.tail())

# #calculate benchmark drawdown
# price['Bench_Peak'] = price.Bench_Bal.cummax()
# price['Bench_DD'] = price.Bench_Bal - price.Bench_Peak

# bench_dd = round(((price.Bench_DD / price.Bench_Peak).min() * 100), 2)

# print(bench_dd)


# #calculate additional columns for strategy

# #daily range
# price['Range'] = price.High - price.Low
# #distance between close and daily low
# price['Dist'] = abs(price.Close - price.Low)
# #distance as % of range
# price['Pct'] = (price.Dist / price.Range) * 100

# print(price.tail())


# #identify entries and allocate trading fees
# price['Long'] = price.Pct < PCT_THRESH

# #calculate system return and balance
# price['Sys_Ret'] = np.where(price.Long.shift(1) == True, price.Return, 1)
# price['Sys_Bal'] = (STARTING_BALANCE * price.Sys_Ret.cumprod())

# print(price.tail())

# plt.plot(price.Bench_Bal)
# plt.plot(price.Sys_Bal)

# plt.show()