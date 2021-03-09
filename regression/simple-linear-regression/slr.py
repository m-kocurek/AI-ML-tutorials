import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#read data
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#model
regressor = LinearRegression()
regressor.fit( X_train, Y_train )

#predicting results
y_predicted = regressor.predict(X_test)

#visualise training set
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue' )
plt.title('Salary vs Experience - training set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#visualise test set
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue' )
plt.title('Salary vs Experience - test set')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#prediction
print(regressor.predict([[12]]))
print(regressor.coef_)
print(regressor.intercept_)
