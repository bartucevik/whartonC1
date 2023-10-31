import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

def fetch_stock_info(ticker):
    try:
        print(ticker)
        mainval = yf.Ticker(ticker)
        stockinfo = mainval.info
        return stockinfo
    except:
        print(f"Error fetching data for {ticker}")
        return None

# Load your data
path = "C:/Users/bartu/Downloads/wilshire_5000_stocks.xlsx"
datapre = pd.read_excel(path)
ticker = datapre["Ticker"].tolist()
specificSlist = []

# Create a ThreadPoolExecutor to fetch stock info in parallel
max_threads = 15  # You can adjust the number of threads based on your machine's capabilities
results = []
with ThreadPoolExecutor(max_threads) as executor:
    results = list(executor.map(fetch_stock_info, ticker))

with ThreadPoolExecutor(max_threads) as executor:
    specific = list(executor.map(fetch_stock_info, ))

# Filter out None results (errors)
results = [res for res in results if res is not None]

# Create a DataFrame from the results
mainframe = pd.DataFrame(results)

# Save to CSV
mainframe.to_csv('aidata.csv', encoding='utf-8')

