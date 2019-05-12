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
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

# Fitting Regression Model to the dataset
# Create regressor

# Predicting a new result with Regression
Y_pred =regressor.predict([[6.5]])

# Visualizing Regression results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Polynomial Regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.grid()
plt.show()

## Visualizing Regression results (for higher resolution and smoother curve)
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid),1))
#plt.scatter(X, Y, color='red')
#plt.plot(X_grid, regressor.predict(X_grid), color='blue')
#plt.title('Polynomial Regression Results')
#plt.xlabel('Position Level')
#plt.ylabel('Salary')
#plt.grid()
#plt.show()