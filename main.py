import yfinance as yf
import pandas as pd
import multiprocessing
from icecream import ic


# Print the ETF info
exchange_mapping = {
    "B3 S.A.": ".sa", #
    "BMV": ".mx", #
    "BSE LTD": ".bo", #
    "Euronext": ".pa", #
    "Frankfurt Stock Exchange": ".F", #
    "Hong Kong Exchanges And Clearing Ltd": ".HK", #
    "Korea Exchange": ".ks", #
    "London Stock Exchange": ".L", #
    "Nasdaq": "", #
    "New York Stock Exchange, Inc.": "", #
    "Shenzhen Stock Exchange": ".ss", #
    "SIX Swiss Exchange": ".SW", #
    "Toronto Stock Exchange": ".TO", #
    "Shanghai Stock Exchange": ".sh", #
}




dataL = "C:/Users/bartu/Downloads/23-24 Competition Stock List-FINAL.xlsx"
dataL = pd.read_excel(dataL)
datal = pd.DataFrame(dataL)
datal["Exchange2"] = datal["Exchange"].map(exchange_mapping)
datal["totaln"] = datal["Ticker"].astype(str)+datal["Exchange2"]
thenewc = datal["totaln"].tolist()
mainframe = pd.DataFrame({})
partialframe = pd.DataFrame({})

for i in thenewc:
    try:
        print(i)
        mainval = yf.Ticker(i)
        stockinfo = mainval.info
        partialframe = pd.DataFrame([stockinfo])
        partialframe = partialframe.applymap(str)
        mainframe = pd.concat([mainframe, partialframe], axis=0)
    except:
        print("error")


pd.options.display.max_colwidth = 1000
mainframe = pd.DataFrame(mainframe.drop_duplicates(keep="last"))
print(mainframe.to_string())
mainframe.to_csv('file_name3.csv', encoding='utf-8')




