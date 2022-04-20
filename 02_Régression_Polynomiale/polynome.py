#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import sklearn.pipeline as pipe
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

fig = plt.figure() #create empty figure

nb = 15
		
X = np.random.uniform(-3, 10, nb) #uniform points
noise = np.random.normal(0,1, nb) #gaussian noise
y = 10 * np.sin(X) / X #values = 10 * sin(x) / x

y += noise #adding gaussian noise to clean values

plt.plot(X, y, 'o', c='black') #plot points in black color

k = 0
color_curve = ["green", "red", "blue", "orange", "brown"]
x_plot = np.linspace(-3, 10, 100).reshape(-1,1)
RSS = []

for i in [1, 3, 6, 9, 12]:
	model = pipe.make_pipeline(PolynomialFeatures(i), Ridge())
	model.fit(X.reshape(-1,1), y)

	y_plot = model.predict(x_plot)
	
	y_predict = model.predict(X.reshape(-1,1))
	plt.plot(x_plot, y_plot, color=color_curve[k])
	
	RSS.append(mean_squared_error(y, y_predict))
	print("degree " + str(i) + ": Residual mean square= " + str(RSS[k]))
	
	k += 1

plt.axis([-3, 10, -5, 15]) #configure axis : [xmin, xmax, ymin, ymax]
plt.title("plot all models") #add title
plt.grid() #add grid to figure
plt.show() #show figure
