import pandas as pd
#thetext = open("thecsv.txt","w+")
thecsv = pd.read_csv("../file_name2.csv")
print(pd.DataFrame(thecsv).columns)
"""
df = pd.DataFrame(thecsv)
for i in df.columns:
    print(df[i])
    ask = input(i)
    if ask == " ":
        thetext.write(str(i)+"\n")
"""