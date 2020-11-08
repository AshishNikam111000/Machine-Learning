import sklearn, math
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris

'''
PROBLEM 1:- Given A dataset of digits(which are blured images), and predict what is the number.
Loading my digits dataset which contains data having these columns: ['DESCR', 'data', 'images', 'target', 'target_names']'''
'''digits = load_digits()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(digits.data, digits.target, test_size=0.2)

model = linear_model.LogisticRegression()
model.fit(x_train,y_train)

#print(model.score(x_test, y_test))

print(model.predict(x_test))
'''

'''
PROBLEM 2:-Given A dataset of iris flower(which are iris images), and predict what is type os iris flower.
Loading my iris flower dataset which contains data having these columns: ['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']'''
iris = load_iris()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(iris.data, iris.target, test_size=0.2)

model = linear_model.LogisticRegression()
model.fit(x_train,y_train)

print(model.score(x_test, y_test))

print(model.predict(x_test))
