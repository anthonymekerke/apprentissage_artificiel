#!/usr/bin/python3

"""
dot -Tpdf fichier.dot -o fichier.pdf -> 'Pour lire les fichiers *.dot'
"""

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def main():
	dataframe = pd.read_csv("data/glass.csv")
	del dataframe['Id']
	del dataframe['Iron']
	
	x_train, x_test = train_test_split(dataframe, test_size=0.33)
	
	y_train = x_train['Type']
	y_test = x_test['Type']
	
	del x_train['Type']
	del x_test['Type']
	
	classifier = tree.DecisionTreeClassifier(criterion='entropy')
	classifier.fit(x_train, y_train)
	tree.export_graphviz(classifier, out_file='tree_glass.dot',
						max_depth=5,
						feature_names=['refractive index','Sodium','Magnesium','Aluminium','Silicon','Potassium','Calcium','Barium'])
						
	score_train = classifier.score(x_train, y_train)
	score_test = classifier.score(x_test, y_test)
	
	print("train score: " + str(score_train) + ", Test score: " + str(score_test))
	
	y_predict = classifier.predict(x_test)
	matrix = confusion_matrix(y_test, y_predict)
	print(matrix)
	

if __name__ =='__main__':
	main()
