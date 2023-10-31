import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import load_model

thecsv = pd.read_csv("file_name3.csv")
df = pd.DataFrame(thecsv)
dropnacon1 = len(df.columns)*0.05
df = df.dropna(thresh = dropnacon1)
with open("thecsv.txt", "r") as alpha:
    selected_columns = alpha.read().splitlines()
df = df[selected_columns]
# df = df[["Unnamed: 0","fullTimeEmployees","auditRisk",]]
df = df.loc[:, df.isna().sum() <  len(df) / 4]

def dummyconv(var1):
    if var1.nunique == 2:
        var1 = pd.get_dummies(var1)
    return var1

df = df.apply(dummyconv)
print(df)
aidata = pd.DataFrame(df.select_dtypes(exclude=['object']))
aidata2 = aidata.filter(like="date")
aidata = aidata.drop(aidata2, axis=1)
aidata = aidata.loc[:, df.nunique() > 1]
aidata = aidata.fillna(0)

y_data = aidata["currentPrice"].values
print(len(y_data))
x_data = aidata.drop(["currentPrice"], axis=1).values
print(len(x_data))
scaler = MinMaxScaler()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=101)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

themodel = load_model("my_model.h5")
print(themodel)
predictions = themodel.predict(x_test)
print("mse")
print(mean_squared_error(y_test, predictions))