##################################################################################
# Creator     : Gaurav Roy
# Date        : 12 May 2019
# Description : The code contains the template for all non-SLR and non-MLR  
#               Regression on the Position_Salaries.csv.
##################################################################################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:3].values

## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#
#sc_Y = StandardScaler()
#Y_train = sc_Y.fit_transform(Y_train)

# Fitting Regression Model to the dataset
# Create regressor

# Predicting a new result with Regression
Y_pred =regressor.predict([[6.5]])

# Visualizing Regression results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.grid()
plt.show()

## Visualizing Regression results (for higher resolution and smoother curve)
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid),1))
#plt.scatter(X, Y, color='red')
#plt.plot(X_grid, regressor.predict(X_grid), color='blue')
#plt.title('Regression Results')
#plt.xlabel('Position Level')
#plt.ylabel('Salary')
#plt.grid()
#plt.show()
