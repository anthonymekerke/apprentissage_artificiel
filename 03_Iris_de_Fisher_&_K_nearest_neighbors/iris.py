#!/usr/bin/python3

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

dataframe = pd.read_csv("iris.csv")

""""

print("shape:")
print(dataframe.shape) #print row and column of data
print("infos:")
print(dataframe.info()) #print infos about data
print("description:")
print(dataframe.describe()) #print description of data
print("head:")
print(dataframe.head()) #print first row of data

"""

train, test = train_test_split(dataframe, test_size=0.3) #split data in 2

Y_train = train["Species"]
Y_test = test["Species"]

del test["Species"] #train.drop(['Species'], axis=1) is equivalent
del train["Species"] #test.drop(['Species'], axis=1) is equivalent

model = KNeighborsClassifier(n_neighbors=5)

model.fit(train, Y_train)
y_predict = model.predict(test)

score_train = model.score(train, Y_train)
score_test = model.score(test, Y_test)

print(score_train)
print(score_test)

matrix = confusion_matrix(Y_test, y_predict)

print(matrix)
