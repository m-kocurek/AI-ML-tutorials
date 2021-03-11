import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

#read data
dataset = pd.read_csv('50_Startups.csv')


X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


ct = ColumnTransformer( transformers = [('encoder', OneHotEncoder(), [3] )], remainder ='passthrough')
X = np.array(ct.fit_transform(X))

#how to avoid dumy variable
X = X[:, 1:]
 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


regressor = LinearRegression()
regressor.fit(X_train, Y_train )

#predict results
Y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print( np.concatenate( ( Y_pred.reshape(len(Y_pred),1) , Y_test.reshape(len(Y_test),1) ), 1 ))


X = np.append(arr = np.ones((50,1)).astype(int), values= X, axis =1 )

X_opt = X[:, [0,1,2,3,4,5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog =y, exog = X_opt).fit()

regressor_OLS.summary()
X_opt = X[:,[0,1,3,4,5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog =y, exog = X_opt).fit()

regressor_OLS.summary()
X_opt = X[:,[0,3,4,5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog =y, exog = X_opt).fit()


regressor_OLS.summary()
X_opt = X[:,[0,3,5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog =y, exog = X_opt).fit()


regressor_OLS.summary()
X_opt = X[:,[0,3]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog =y, exog = X_opt).fit()
regressor_OLS.summary()



