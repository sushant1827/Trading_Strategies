# Import libraries
import pandas as pd
import talib as ta

# Define parameters
rsiLength = 9
stochLength = 9
k = 3
d = 3
diLength = 10
adxSmoothing = 14

# Load data
data15m = pd.read_csv('NIFTY50_INDEX_15_Min.csv')
data15m['Date'] = pd.to_datetime(data15m['Date'])
data15m.set_index('Date', inplace=True)

data3m = pd.read_csv('NIFTY50_INDEX_5_Min.csv')
data3m['Date'] = pd.to_datetime(data3m['Date'])
data3m.set_index('Date', inplace=True)

# Calculate Stochastic RSI indicator
rsiValue = ta.RSI(data15m['Close'], timeperiod=rsiLength)
rsiMax = ta.MAX(rsiValue, timeperiod=stochLength)
rsiMin = ta.MIN(rsiValue, timeperiod=stochLength)
stochRsiK = ((rsiValue - rsiMin) / (rsiMax - rsiMin)) * 100
stochRsiD = ta.SMA(stochRsiK, timeperiod=d)

# Calculate DMI indicator
data3m['dmiPlus'] = ta.PLUS_DI(data3m['High'], data3m['Low'], data3m['Close'], timeperiod=diLength)
data3m['dmiMinus'] = ta.MINUS_DI(data3m['High'], data3m['Low'], data3m['Close'], timeperiod=diLength)
data3m['adx'] = ta.ADX(data3m['High'], data3m['Low'], data3m['Close'], timeperiod=adxSmoothing)

# Define buy and sell signals
buySignal = (stochRsiK > stochRsiD) #&
            #  (data3m['dmiPlus'] > data3m['dmiMinus']) &
            #  (data3m['adx'] > 25))
sellSignal = (stochRsiK < stochRsiD) #&
            #   (data3m['dmiMinus'] > data3m['dmiPlus']) &
            #   (data3m['adx'] > 25))

buyFlag = False
sellFlag = False
buyprice = 0.
sellprice = 0.
unitPnl = 0.
totalPnl = 0.
qty = 50

# Execute trades
for i in range(len(data3m)):
    if buySignal[i]:
        buyFlag = True
        buyprice = data3m['Close'][i]
        if sellFlag:
            unitPnl = sellprice - buyprice
            # print(unitPnl)
        totalPnl += unitPnl * qty
        
    elif sellSignal[i]:
        sellFlag = True
        sellprice = data3m['Close'][i]
        if buyFlag:
            unitPnl = sellprice - buyprice
            # print(unitPnl)

        totalPnl += unitPnl * qty

    print("Total Pnl : " + str(totalPnl))