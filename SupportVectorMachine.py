from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

data = load_breast_cancer()

X = np.array(data.data)
Y = np.array(data.target)

x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.2)

#usually kernel value is 'polynomial' or 'rbf'
#C - is soft marging size
clf = SVC(kernel='linear', C=3)
clf.fit(x_train,y_train)

clf9 = KNeighborsClassifier(n_neighbors=9)
clf9.fit(x_train, y_train)
print(f'the accuracy of the SVC model is {clf.score(x_test,y_test)} \nthe accuracy of the K-NN model is {clf9.score(x_test,y_test)}')