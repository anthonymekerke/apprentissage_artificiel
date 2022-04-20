#!/usr/bin/python3

import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
	dataframe = pd.read_csv("data/winequality-red.csv")
	print(dataframe.head())
	
	x_train, x_test = train_test_split(dataframe, test_size=0.33)
	
	y_train = x_train['quality']
	y_test = x_test['quality']
	
	del x_train['quality']
	del x_test['quality']
	
	model = DecisionTreeRegressor()
	model.fit(x_train, y_train)
	
	tree.export_graphviz(model, out_file='tree_red-wine.dot',
						max_depth=5,
						feature_names=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'])
				
	y_predict = model.predict(x_test)
	
	COE = r2_score(y_test, y_predict)
	MAE = mean_absolute_error(y_test, y_predict)
	MSE = mean_squared_error(y_test, y_predict)
	
	print("Coefficient of determination: " + str(COE))
	print("MAE: " + str(MAE))
	print("MSE: " + str(MSE))
	
if __name__ == '__main__':
	main()
