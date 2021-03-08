import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#import dataset
dataset = pd.read_csv('Data.csv')
X =   dataset.iloc[:, :-1].values  #locate indexes
Y = dataset.iloc[:, -1].values  #dependent var vector
#-1 is the last column

#print(X)
#print(Y)

#what about missing data? lets handle this
#1st we can delete if the dataset is huge
#2nd replace by avrg

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, 1:3])
X[:, 1:3]= imputer.transform(X[:, 1:3]) #update data

#print(X)

#encoding categorical data
ct = ColumnTransformer( transformers=[('encoder', OneHotEncoder(), [0] )], remainder='passthrough' )
X= np.array( ct.fit_transform(X))
# France 100, Span 001, Germany 010
#print(X)

# encoding dependent variable
le = LabelEncoder()
Y= le.fit_transform(Y)
#print(Y) #0 no, 1 yes

# split data into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1)

#print(X_train)
#print(X_test)
#print(Y_train)
#print(Y_test)

#feature scalling
sc = StandardScaler()
X_train[:, 3:]= sc.fit_transform(X_train[:, 3:])
X_test[:, 3:]= sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)
