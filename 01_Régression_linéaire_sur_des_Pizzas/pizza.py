#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Génération des points
X = np.array([[6], [8], [10], [14], [18]]) #pizzas' sizes
y = [7, 9, 13, 17.5, 18] #pizzas' prices

# Dessiner le graphes
fig = plt.figure() #create empty figure

plt.plot(X,y, 'o', c='black') #plot points in blue color

plt.axis([0, 25, 0, 25]) #configure axis : [xmin, xmax, ymin, ymax]
plt.xlabel('sizes in cms')
plt.ylabel('Prices in euros')
plt.title('pizza prices plotted against size')
plt.grid() #add the grid on figure

# Définition du Modèles d'apprentissage
regr = LinearRegression() #create an object LinearRegression
regr.fit(X, y) #train the model

x_plot = np.linspace(0, 25, 10).reshape(-1,1)
y_plot = regr.predict(x_plot) #make the linear regression

plt.plot(x_plot, y_plot, color="green") #plot linear regression

plt.show() #show the figure

# Évaluation du Modèle
y_predict = regr.predict(X)
RSS = 0

for i in range(5):
	RSS += (y[i] - y_predict[i])**2
	
print("Residual sum of squares : " + str(RSS))

variance = np.var(X, ddof=1) # variance(X)
covariance = np.cov(X.transpose(), y)[0][1] #covariance(x,y)
alpha = covariance / variance
r_squared = regr.score(x_plot, y_plot)

print("variance: " + str(variance))
print("covariance: " + str(covariance))
print("alpha: " + str(alpha))
print("Rsquared = " + str(r_squared))
