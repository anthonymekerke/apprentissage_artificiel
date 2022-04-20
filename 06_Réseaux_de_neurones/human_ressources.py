#!/usr/bin/python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def label_encode(data, col):
	# Transforme un type catégorie en entier
	le = LabelEncoder()

	# On récupère tous les noms de catégories possibles
	unique_values = list(data[col].unique())
	le_fitted = le.fit(unique_values)

	# On liste l'ensemble des valeurs
	values = list(data[col].values)

	# On transforme les catégories en entier
	values_transformed = le.transform(values)

	# On fait le remplacement de la colonne dans le dataframe d'origine
	data[col] = values_transformed

def analyze_good_employees(data):
	averages = data.mean() # créer un data contenant toutes les moyennes pour tout
	average_last_evaluation = averages['last_evaluation']
	average_project = averages['number_project']
	average_montly_hours = averages['average_montly_hours']
	average_time_spend = averages['time_spend_company']

	good_employees = data[data['last_evaluation'] > average_last_evaluation]
	good_employees = good_employees[good_employees['number_project'] > average_project]
	good_employees = good_employees[good_employees['average_montly_hours'] > average_montly_hours]
	good_employees = good_employees[good_employees['time_spend_company'] > average_time_spend]

	sns.set()
	plt.figure(figsize=(15, 8))
	plt.hist(data['left'])
	print(good_employees.shape)
	sns.heatmap(good_employees.corr(), vmax=0.5, cmap="PiYG")
	plt.title('Correlation matrix')
	plt.show()

def main():
	dataframe = pd.read_csv("data/human_resources.csv")
	for col in dataframe.columns:
		label_encode(dataframe, col)
	
	x_train, x_test = train_test_split(dataframe, test_size=0.3)
	
	y_train = x_train['left']
	del x_train['left']
	y_test = x_test['left']
	del x_test['left']
	
	model = MLPClassifier()
	
	model.fit(x_train, y_train)
	y_predict = model.predict(x_test)
	
	score_train = model.score(x_train, y_train)
	score_test = model.score(x_test, y_test)
	matrix = confusion_matrix(y_test, y_predict)
	
	print("Train score:" + str(score_train) + ", Test score: " + str(score_test))
	print(matrix)

	analyze_good_employees(dataframe)

if __name__ == '__main__':
	main()
