import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied = np.array([1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20]).reshape(-1,1)
scores = np.array([20,20,20,35,30,50,55,60,62,56,70,72,73,80,81,60,97,96,100]).reshape(-1,1)

"""
model_1 = LinearRegression()
model_1.fit(time_studied, scores)

#print(model.predict(np.array([12]).reshape(-1,1)))


plt.scatter(time_studied,scores)
plt.plot(np.linspace(0,20,100).reshape(-1,1),model_1.predict(np.linspace(0,20,100).reshape(-1,1)),'r')
plt.ylim(0,100)
#plt.show()
"""

# train and test split example

time_train, time_test, score_train, score_test = train_test_split(time_studied,scores, test_size= 0.2)

model_1 = LinearRegression()
model_1.fit(time_test, score_test)
model_2 = LinearRegression()
model_2.fit(time_train, score_train)

#here we check the accuracy of the prediction on a test data
print(model_2.score(time_test,score_test))

plt.scatter(time_train, score_train)
plt.plot(np.linspace(0,20,100).reshape(-1,1),model_2.predict(np.linspace(0,20,100).reshape(-1,1)),'b')
plt.plot(np.linspace(0,20,100).reshape(-1,1),model_1.predict(np.linspace(0,20,100).reshape(-1,1)),'g')
plt.show()