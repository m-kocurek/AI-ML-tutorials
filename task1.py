#libs
from pandas import read_csv
from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load data <3

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'class']
dataset= read_csv(url, names=names)

#shape
#print(dataset.shape)

#head
#print(dataset.head(20))

#descriptions
#print(dataset.describe())

#class distribution
#print(dataset.groupby('class').size())

#wizualizacjaaaaa -> wykresy wasy box
    #box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()

    #histogramy
#dataset.hist()
#pyplot.show()

#scatter plot matrix
#scatter_matrix(dataset)
#pyplot.show()

#split out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X,y, tets_size=0.20 , random_state=1)
