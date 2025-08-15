from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)


clf3 = KNeighborsClassifier(n_neighbors=3)
clf3.fit(x_train,y_train)
clf9 = KNeighborsClassifier(n_neighbors=9)
clf9.fit(x_train, y_train)

#comparison

print(f"K=3 score is {clf3.score(x_test, y_test)} \nK=9 score is {clf9.score(x_test, y_test)}")

