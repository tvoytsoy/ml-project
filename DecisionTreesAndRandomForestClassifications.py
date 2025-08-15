from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

data = load_breast_cancer()

X = np.array(data.data)
Y = np.array(data.target)

x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.2)

#usually kernel value is 'polynomial' or 'rbf'
#C - is soft marging size
SV_clf = SVC(kernel='linear', C=3)
SV_clf.fit(x_train,y_train)

KN_clf = KNeighborsClassifier(n_neighbors=9)
KN_clf.fit(x_train, y_train)

DT_clf = DecisionTreeClassifier()
DT_clf.fit(x_train, y_train)

RF_clf = RandomForestClassifier()
RF_clf.fit(x_train,y_train)

print(f'the accuracy of the SVC model is {SV_clf.score(x_test,y_test)} '
      f'\nthe accuracy of the K-NN model is {KN_clf.score(x_test,y_test)}'
      f'\nthe accuracy of the DTC model is {DT_clf.score(x_test,y_test)}'
      f'\nthe accuracy of the RFC model is {RF_clf.score(x_test,y_test)}')