import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

dfaidata = pd.read_csv("../aidata.csv")
dfaidata = pd.DataFrame(dfaidata)
dfaidata = dfaidata.drop(columns = ['dateShortInterest'], axis=1)

def preprocessor(df):
    with open("../thecsv.txt", "r") as alpha:
        selected_columns = alpha.read().splitlines()

    df = df[selected_columns]
    dropnacon1 = len(df.columns) * 0.3
    df = df.dropna(thresh=dropnacon1)
    df = df.loc[:, df.isna().mean() < 0.3]

    # Drop columns with more than 95% missing values
    dropnacon1 = len(df.columns) * 0.05
    df = df.dropna(thresh=dropnacon1)

    # Use mean instead of sum for more clarity

    # Convert binary categorical columns to dummies
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() == 2:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col, drop_first=True)], axis=1)
            df.drop([col], axis=1, inplace=True)

    df = df.dropna()
    # Drop non-numeric columns and columns with only one unique value
    aidata = df.select_dtypes(exclude=['object']).loc[:, df.nunique() > 1]

    # Handle missing values by filling with zeros
    with pd.option_context('mode.use_inf_as_na', True):
        aidata.dropna(inplace=True)
    columnslist = aidata.columns

    return aidata, df, columnslist

alphabased = 0.00001
alphalist = []
for i in range(1, 10):
    alphabased *= i
    alphalist.append(alphabased)

def aiprogress(x_data,y_data):
    global y_test, y_pred2, scaler, lmodel, ridge
    scaler = MinMaxScaler()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=101, shuffle=True)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    print(len(x_train))
    x_test = scaler.transform(x_test)
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LinearRegression
    parameters = {'alpha': alphalist}
    ridge = Ridge()
    grid = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=4)
    grid.fit(x_train, y_train)
    best_alpha = grid.best_params_['alpha']
    ridge = Ridge(alpha=best_alpha)
    print(best_alpha)
    ridge.fit(x_train, y_train)
    y_pred2 = ridge.predict(x_test)
    mse = mean_squared_error(y_test, y_pred2)
    lmodel = LinearRegression()
    lmodel.fit(x_train, y_train)
    print("shape", np.arange(x_train.shape[0]))
    predlmodel = lmodel.predict(x_test)
    msel = mean_squared_error(y_test, predlmodel)
    print(y_test.head())
    print(f"Mean Squared Error (MSE)     of n: {mse}")
    print(f"Linear model (MSE): {msel}")

aidata, df, cols1 = preprocessor(dfaidata)
columns1a = aidata.columns
y_data = aidata["currentPrice"]# Using pop to drop the column and retrieve its values
x_data = aidata.drop(columns=["currentPrice"], axis=1)
aiprogress(x_data, y_data)
# sectoral basis
def repeatmodule(selecteddf):
    testmpred = []
    positivesnames = []
    print("y_test",len(y_test   ))
    print(y_test[1])
    for i in range(len(y_test)):
        valuetmp = y_test[i] - y_pred2[i]
        testmpred.append(valuetmp)
        if valuetmp > 0: positivesnames.append(selecteddf["symbol"][i])
    testmpred = pd.DataFrame(testmpred, columns=["data"])
    negatives = testmpred[testmpred < 0].dropna()
    pozitives = testmpred[testmpred > 0].dropna()
    pozitivesdf = {
        "pozitives": pozitives,
        "names": positivesnames
    }
    pozitivesdf = pd.DataFrame(pozitivesdf)
    print(pozitivesdf)
    sectorslist = pd.unique(selecteddf["sector"])
    for i in sectorslist:
        dfnew = selecteddf[selecteddf["sector"] == i]
        dfnew = dfnew.select_dtypes(exclude=['object']).loc[:, dfnew.nunique() > 1]
        # Handle missing values by filling with zeros
        with pd.option_context('mode.use_inf_as_na', True):
            dfnew.dropna(inplace=True)
        y_data = dfnew["currentPrice"].values  # Using pop to drop the column and retrieve its values
        x_data = dfnew.drop(columns=["currentPrice"], axis=1).values
        aiprogress(x_data, y_data)
repeatmodule(df)

"""
data1 = pd.read_csv("file_name2.csv")
data1 = pd.DataFrame(data1)
data1 = data1[data1["country"] == "United States"]
dataA = data1
print("file_name2")
aidatan, dfn, cols2 = preprocessor(dataA)
columns2a = aidatan.columns
cols1 = [col for col in cols1 if col in aidatan.columns]
aidatan = aidatan[cols1]
y_data4 = aidatan["currentPrice"].values  # Using pop to drop the column and retrieve its values
x_data4 = aidatan.drop(columns=["currentPrice"], axis=1).values
xdata_n = scaler.transform(x_data4)
predict_l = lmodel.predict(xdata_n)
predict_r = ridge.predict(xdata_n)

predictionsdf = pd.DataFrame({})
lineardiffL = []
ridgediffL = []
for i in range(len(x_data)):
    lineardiffL.append(predict_l[i] - y_data4[i])
    ridgediffL.append(predict_r[i] - y_data4[i])
predictionsdf["linear"] = lineardiffL
predictionsdf["Ridge"] = ridgediffL
print(predictionsdf)
"""

df3 = df[df["symbol"].isin(["LULU","HIBB","AEO","URBN","ULTA","TJX"])]
df3 = df3.dropna()
df3 = df3[cols1]
print(df3)
y_data = df3["currentPrice"].values  # Using pop to drop the column and retrieve its values
x_data = df3.drop(columns=["currentPrice"], axis=1).values
print("newsample")
print("\n\n\n\n\n")
aiprogress(x_data, y_data)
repeatmodule(df3)



















