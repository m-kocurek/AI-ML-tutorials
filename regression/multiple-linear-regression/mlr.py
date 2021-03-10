import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

#read data
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#print(X)

#encode categorical data  -> [3 rd] column is dummy data
ct = ColumnTransformer( transformers = [('encoder', OneHotEncoder(), [3] )], remainder ='passthrough')
X = np.array(ct.fit_transform(X))

#print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train )

#predict results
Y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print( np.concatenate( ( Y_pred.reshape(len(Y_pred),1) , Y_test.reshape(len(Y_test),1) ), 1 ))




