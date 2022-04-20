#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import sklearn.pipeline as pipe
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

fig = plt.figure() #create empty figure

#dataset train
nb_train = 15	
X_train = np.random.uniform(-3, 10, nb_train)
noise_train = np.random.normal(0,1, nb_train)
y_train = 10 * np.sin(X_train) / X_train
y_train += noise_train

#dataset test
nb_test = 50
X_test = np.random.uniform(-3, 10, nb_test)
noise_test = np.random.normal(0,1, nb_test)
y_test = 10 * np.sin(X_test) / X_test
y_test += noise_test

plt.plot(X_test, y_test, 'o', c='black')

k = 0
color_curve = ["green", "red", "blue", "orange", "brown"]
degree = [1,3,6,9,12]

for i in degree:
	model = pipe.make_pipeline(PolynomialFeatures(i), Ridge())
	model.fit(X_train.reshape(-1,1), y_train)

	y_predict = model.predict(X_test.reshape(-1,1))
	
	plt.plot(X_test, y_predict, color=color_curve[k])
	k += 1
	
	RSS = mean_squared_error(y_test, y_predict)
	print("degree " + str(i) + ": Residual mean square= " + str(RSS))
	

plt.axis([-3, 10, -5, 15]) #configure axis : [xmin, xmax, ymin, ymax]
plt.title("plot all models") #add title
plt.grid() #add grid to figure
plt.show() #show figure
