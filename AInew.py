import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
datano = 0
particular_columns = []

class aiproject:
    def __init__(self, predf):
        self.predf = predf+".csv"
        self.predf = pd.read_csv(self.predf)
        self.predf = pd.DataFrame(self.predf)
        self.df = self.predf

    def columnselection(self):
        with open("thecsv.txt", "r") as alpha:
            selected_columns = alpha.read().splitlines()
        if datano != 0:
            self.df = self.df[particular_columns]

    def selected_columns(self):
        selected_columns = input("Type the columns, separate with comas (no spaces)")
        selected_columns = selected_columns.split(",")
        self.df = self.df[selected_columns]

    def preliminary(self):
        dropnacon1 = len(self.df.columns) * 0.3
        self.df = self.df.dropna(thresh=dropnacon1)
        self.df = self.df.loc[:, self.df.isna().mean() < 0.3]
        # Drop columns with more than 95% missing values
        dropnacon1 = len(self.df.columns) * 0.05
        self.df = self.df.dropna(thresh=dropnacon1)

    def binatyconverter(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object' and self.df[col].nunique() == 2:
                self.df = pd.concat([self.df, pd.get_dummies(self.df[col], prefix=col, drop_first=True)], axis=1)
                self.df.drop([col], axis=1, inplace=True)

    def lastdropper(self): self.df = self.df.dropna()

    def preoprocessor(self):
        # Drop non-numeric columns and columns with only one unique value
        with pd.option_context('mode.use_inf_as_na', True):
            self.df.dropna(inplace=True)
        self.aidata = self.df.select_dtypes(exclude=['object']).loc[:, self.df.nunique() > 1]
        self.processcolumns = self.aidata.columns
        # Handle missing values by filling with zeros
        self.y_data = self.aidata["currentPrice"]  # Using pop to drop the column and retrieve its values
        self.x_data = self.aidata.drop(columns=["currentPrice"], axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data, test_size=0.2, random_state=101)

    def scaler(self):
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.x_train)
        self.x_train = self.scaler.transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

    def alphalist(self):
        alphabased = 10 ** -int(input("Basic alpha power"))
        self.alphalist = []
        for i in range(1, 10):
            alphabased *= i
            self.alphalist.append(alphabased)

    def models(self):
        parameters = {'alpha': self.alphalist}
        ridge = Ridge()
        self.grid = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
        self.grid.fit(self.x_train, self.y_train)
        best_alpha = self.grid.best_params_['alpha']
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(self.x_train, self.y_train)
        y_pred2 = ridge.predict(self.x_test)
        mse = mean_squared_error(self.y_test, y_pred2)
        self.lmodel = LinearRegression()
        self.lmodel.fit(self.x_train, self.y_train)
        predlmodel = self.lmodel.predict(self.x_test)
        msel = mean_squared_error(self.y_test, predlmodel)
        print(f"Ridge model (MSE): {mse}")
        print(f"Linear model (MSE): {msel}")

    def p_or_n(self):
        for i in self.df["symbol"]:
            thedata = self.df[self.df["symbol"] == i]
            thedata = thedata[self.processcolumns]
            # Drop these columns from the DataFrame
            thedatay = thedata["currentPrice"]
            thedatax = thedata.drop(["currentPrice"], axis=1)
            thedatax = self.scaler.transform(thedatax)
            thedatap = self.lmodel.predict(thedatax)
            result = thedatap - thedatay
            print(f"{i}:{result}")







class specific_analysis(aiproject):
    def __init__(self, dfs):
        self.dfs = dfs

    def selectiveNs(self):
        self.__columnname = input("Which column")

    def selective(self):
        dfuniques = pd.unique(self.dfs[self.__columnname])
        valueslist = []
        print("Press space to add")
        for i in dfuniques:
            asktoaddc = input(f"do you wanna add {i} as a value?")
            if asktoaddc == " ": valueslist.append(i)
        self.dfsnew = self.dfs[self.dfs[self.__columnname].isin(valueslist)]
        print(self.dfsnew)



csvslist = ["aidata"]

for i in csvslist:
    app1 = aiproject(i)
    app1.columnselection()
    app1.preliminary()
    app1.binatyconverter()
    app1.lastdropper()
    app1.preoprocessor()
    app1.scaler()
    app1.alphalist()
    app1.models()
    app1.p_or_n()

print("APP2:")
print("\n"*5)

app2 = specific_analysis(app1.predf)
app2.selectiveNs()
app2.selective()