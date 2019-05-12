##################################################################################
# Creator     : Gaurav Roy
# Date        : 12 May 2019
# Description : The code contains the approach for Polynomial Linear Regression on 
#               the Position_Salaries.csv.
##################################################################################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Position_Salaries.csv')

# X = dataset.iloc[:,1].values
# Since X only had 1 feture, it became a vector (1D) instead of matrix
# To avoid any future warnings, we should make X a matrix instead of a vector
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

# Creating MLR using this new X_poly
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

# Visualizing Linear Regression results
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear Regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')

# Visualizing Polynomial Regression results
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='green')
plt.title('Polynomial Regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.grid()
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
