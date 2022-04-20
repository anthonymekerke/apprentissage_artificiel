#!/usr/bin/python3

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("auto-mpg.csv")

"""

print(df.head())

"""

del df["name"]

train, test = train_test_split(df, test_size=0.3) #split data in 2

Y_train = train["mpg"]
Y_test = test["mpg"]

del train["mpg"]
del test["mpg"]

ss = StandardScaler()
X_train = ss.fit_transform(train)
X_test = ss.fit_transform(test)

model = KNeighborsRegressor(n_neighbors=5)

model.fit(X_train, Y_train)

score_train = model.score(X_train, Y_train) #score() make predict + score
score_test = model.score(X_test, Y_test)

print("score test: " + str(score_test))
print("score train: " + str(score_train))

Y_predict = model.predict(X_test)

print("MAE: " + str(mean_absolute_error(Y_test, Y_predict)))
print("MSE: " + str(mean_squared_error(Y_test, Y_predict)))
