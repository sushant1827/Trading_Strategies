import pandas as pd
from datetime import date, timedelta
import logging
import requests
from io import StringIO
import time
import yfinance as yf

def configure_logging():
    # Configure Application Logging
    log_filepath = "ToolsLog.log"
    format = "%(asctime)s: - %(levelname)s - %(message)s"
    logging.basicConfig(
            filename=log_filepath,
            format=format,
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S")

configure_logging()
logging.info("Tools : MidcapShop Screen extract commenced")

# Get the list of stocks (Nifty Midcap 50)

symbols = []

url = 'https://www.niftyindices.com/IndexConstituent/ind_niftymidcap50list.csv'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
}
response = requests.get(url, headers=headers)

# Check if request was successful
if response.status_code == 200:
    csv_data = StringIO(response.text)
    df = pd.read_csv(csv_data)
    if not df.empty:
        symbols = df["Symbol"].tolist()
else:
    logging.exception("McapShop: Error while fetching stock list from Nifty Mid Cap universe. Skipping strategy")
    quit()

# Function: Calculates price fall% from 20DMA and select top 5 Stocks from that list
if len(symbols) == 0:
    logging.exception("McapShop: Error while fetching stock list from Nifty universe. Skipping strategy")
    quit()

results, end_date = [], date.today()
start_date = end_date - timedelta(days=50)  # To ensure we have at least 20 days of data

# ---------------------------
# Placeholder: implement this
# ---------------------------
def get_historical_data(symbol: str, start_date: date, end_date: date, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol+".NS", start=start_date - timedelta(days=150), end=end_date, auto_adjust=True,
                     interval="1d", progress=False, multi_level_index=None,
                     rounding=True, threads=True)
    time.sleep(0.25)
    if df.empty:
        raise ValueError(f"No data found for {symbol} between {start_date} and {end_date}")

    # Reset index to keep Date as a column
    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.set_index("Date", inplace=True)
    return df

# Find top5 stocks within the Midcap50 universe which is fallen the most from its 20DMA
for symbol in symbols:
    try:
        df = get_historical_data(symbol, start_date, end_date)

        if df is None or df.empty or 'Close' not in df.columns:
            continue

        df = df.sort_index()  # Fix - Sometimes data provided is not properly index on date
        df['20DMA'] = df['Close'].rolling(window=20).mean()

        latest_close = df['Close'].iloc[-1]
        latest_dma = df['20DMA'].iloc[-1]

        if pd.isna(latest_dma):
            continue

        deviation = ((latest_close - latest_dma) / latest_dma) * 100

        if latest_close < latest_dma:
            results.append((symbol, deviation))

    except Exception as e:
        logging.exception("McapShop: Error while fetching eligible stocks")

# Extract just the top 5 symbols that are the farthest below 20DMA
top_5_symbols = [symbol for symbol, _ in sorted(results, key=lambda x: x[1])[:5]]

# Create DataFrame
if not top_5_symbols:
    print("No Eligible Stocks under 20DMA today")
else:
    print(top_5_symbols)
    logging.info("Tools : MidcapShop Screen extract completed")
