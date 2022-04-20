#!/usr/bin/python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    dataframe = pd.read_csv("data/day.csv")
    print(dataframe.head())

    del dataframe['dteday']

    x_train, x_test = train_test_split(dataframe, test_size=0.3)

    y_train = x_train['cnt']
    del x_train['cnt']
    y_test = x_test['cnt']
    del x_test['cnt']

    model = MLPRegressor()

    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    coeff_det = model.score(x_test, y_test)
    MAE = mean_absolute_error(y_test, y_predict)
    MSE = mean_squared_error(y_test, y_predict)

    print("coefficient of determination: " + str(coeff_det))
    print("MAE: " + str(MAE))
    print("MSE: " + str(MSE))

if __name__ == '__main__':
    main()