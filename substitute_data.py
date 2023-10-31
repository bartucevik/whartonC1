import pandas as pd
import yfinance as yf
tickerslistex = ["IQV","UNH","PSA","MSFT","NVDA","V","RGLD","APD","LULU","HIBB","AEO","URBN","ULTA","TJX","AZO","ARCB"]
tickerslist = ["IRM","DLR","IQV","UNH","PSA","MSFT","NVDA","V","RGLD","APD","LULU","HIBB","AEO","URBN","ULTA","TJX","AZO","ARCB"]
#"LULU","HIBB","AEO","URBN","ULTA","TJX","ASML","TREX"
#"IQV","UNH","PSA","MSFT","NVDA","V","RGLD","APD","RGLD","HIBB"
# ["LULU","HIBB","AEO","URBN","ULTA","TJX","ASML","TREX","ARCB"]
datas = pd.DataFrame({})
ds = 0
dno = 0
dname = []
dval = []
sectoral = []
tdfmain = pd.read_csv("file_name3.csv")
tdfmain = pd.DataFrame(tdfmain)


for i in tickerslist:
    data1 = tdfmain[tdfmain["symbol"] == i]
    data1 = data1.applymap(str)
    if float(data1["targetLowPrice"]) * 11/10 > float(data1["currentPrice"]):
        dno += 1
        d = (float(data1["targetHighPrice"]) - float(data1["targetLowPrice"])) / float(data1["currentPrice"])
        print(d)
        sectoral.append(data1["sector"].to_string())
        ds += d
        dname.append(data1["symbol"].to_string())
        dval.append(d)

        datas = pd.concat([datas, data1], axis=0)
datas = datas.drop_duplicates(keep="last")
for i in datas.columns:
    print(datas[i])

ds = ds/dno
percentagen = []
percentaged = 0
for i in range(dno):
    val = dval[i]
    print(dname[i])
    percentagen.append(val/ds*100)
    percentaged += val/ds*100

percentages = []
for i in range(len(dname)):
    percentages.append(percentagen[i] / percentaged * 100)

dataframepre = {
    "percentages": percentages,
    "sectors": sectoral,
    "ticker": dname,
}

df = pd.DataFrame(dataframepre)
df = df.astype(str)
df = df.applymap(lambda x: x.strip('0'))
# Converting the "percentages" column to float
df['percentages'] = df['percentages'].astype(float)
sum_by_sector = df.groupby('sectors')['percentages'].sum()
print(sum_by_sector)
print(df)






