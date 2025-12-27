import pandas as pd
import talib

# Load data
file_path = r"D:\AlgoTrade\Fyers_AlgoTrade\Fyers_Data\BankNifty_Index\NIFTYBANK_INDEX_1_Min.csv"
data = pd.read_csv(file_path, parse_dates=['Date'])

# Rename columns to lowercase to match the rest of the code
data.rename(columns={'Date': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
data.set_index('datetime', inplace=True)

# Resample the data to 15-minute intervals

data = data.resample('15T').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last'
    # 'volume': 'sum'  # if you have a volume column and want to sum it up
})

# print(data.tail())

# Calculate momentum indicators using TA-Lib
# data['momentum'] = talib.MOM(data['close'], timeperiod=10)
# data['rsi'] = talib.RSI(data['close'], timeperiod=14)
# data['roc'] = talib.ROC(data['close'], timeperiod=10)
# data['ema9'] = talib.EMA(data['close'], timeperiod=9) # ROC(data['close'], timeperiod=10)
# data['ema15'] = talib.EMA(data['close'], timeperiod=15) # ROC(data['close'], timeperiod=10)

data = data.tail(10000)

# Initialize variables
initial_cash = 100000
cash = initial_cash
position = 0
trade_qty = 100
brokerage = 100
trades = []
position_entry_price = 0
stop_loss = 0

target_profit = 60
sl_value      = 10
candle_body   = 100
sl_tolerance  = 3
candle_count  = 0

# Iterate through the data
for i in range(2, len(data)):
    # Get current and previous candles
    current = data.iloc[i]
    prev = data.iloc[i - 1]
    # pprev = data.iloc[i - 2]
    # ppprev = data.iloc[i - 3]
    # pppprev = data.iloc[i - 4]

     # Check the strategy conditions
    if (
        # prev['high'] < pprev['high'] and
        # prev['close'] < pprev['low'] and
        # current['open'] > prev['low'] and
        # prev['open'] > prev['close'] and (prev['open'] - prev['close']) >= candle_body and
        # pprev['open'] > pprev['close'] and (pprev['open'] - pprev['close']) >= candle_body and
        prev['open'] > prev['close'] and (prev['open'] - prev['close']) >= candle_body and
        # pprev['open'] > pprev['close'] and #abs(pprev['close'] - pprev['open']) >= candle_body and
        # prev['close'] < pprev['low'] and
        current['low'] <= prev['low'] - sl_tolerance and
        # current['high'] < prev['high'] and
        # current['momentum'] > max(prev['momentum'], pprev['momentum']) and
        # current['rsi'] > max(prev['rsi'], pprev['rsi']) and
        # current['roc'] > max(prev['roc'], pprev['roc']) and 
        # prev['close'] < prev['ema9'] and 
        # prev['ema9'] < prev['ema15'] and 
        # ppprev['open'] < ppprev['close'] and (ppprev['close'] - ppprev['open']) >= candle_body and
        # pppprev['open'] < pppprev['close'] and (pppprev['close'] - pppprev['open']) >= candle_body and
        # ppprev['close'] > pppprev['high'] and
        position == 0):

        # Sell condition
        position_entry_price = prev['low'] - sl_tolerance #current['close']
        position = 1
        # stop_loss = prev['high'] + sl_tolerance
        stop_loss = position_entry_price + sl_value

        trades.append(('sell  ', current.name, round(position_entry_price, 1), 0.0, cash))

    # Check for stop loss or target
    if position > 0:
        # if (current['high'] >= stop_loss and candle_count > 0):
        if ((current['close'] > current['open'] 
            and current['open'] - current['low'] < target_profit
            and current['high'] >= stop_loss)
            or 
            (current['close'] < current['open'] 
            and current['close'] - current['low'] < target_profit
            # and current['open'] - current['low'] < target_profit
            and current['high'] >= stop_loss)
            ):

            cash += (position_entry_price - stop_loss) * trade_qty
            cash -= brokerage
            
            if position_entry_price > stop_loss:
                trades.append(('profit', current.name, stop_loss, position_entry_price - stop_loss, cash))
            else:
                trades.append(('loss  ', current.name, stop_loss, position_entry_price - stop_loss, cash))

            position = 0
            position_entry_price = 0
            stop_loss = 0
            candle_count = 0

        elif position_entry_price - current['low'] >= target_profit:

            cash +=  target_profit * trade_qty
            cash -= brokerage
            trades.append(('target', current.name, position_entry_price - target_profit, target_profit, cash))

            position = 0
            position_entry_price = 0
            stop_loss = 0
            candle_count = 0

        else:

            val = round(position_entry_price - current['close'], 2)
            cash +=  val * trade_qty
            cash -= brokerage

            if position_entry_price > current['close']:
                trades.append(('profit', current.name, round(position_entry_price - val, 1), val, cash))
            else:
                trades.append(('loss  ', current.name, round(position_entry_price - val, 1), val, cash))

            position = 0
            position_entry_price = 0
            stop_loss = 0
            candle_count = 0


# Calculate performance metrics
total_trades = int(len(trades) / 2)
win_trades = len([trade for trade in trades if trade[0] == 'profit' or trade[0] == 'target'])
lose_trades = int(total_trades - win_trades)
win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
net_profit = cash - initial_cash

# Print trading measures
print(f'Start Portfolio Value: {initial_cash}')
print(f'End Portfolio Value: {cash}')
print(f'Net Profit: {net_profit}')
print(f'Total Trades: {total_trades}')
print(f'Win Trades: {win_trades}')
print(f'Losing Trades: {lose_trades}')
print(f'Win Rate: {win_rate:.2f}%')

# Print detailed trade history
# for trade in trades:
    # print(trade)
