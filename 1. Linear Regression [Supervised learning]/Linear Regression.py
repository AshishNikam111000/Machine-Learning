import pandas as pd
import numpy as np
import sklearn, pickle
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style

best = 0
data = pd.read_csv("E:/  ASHISH/StudioCode/Python/Projects/ML/1. Linear Regression [Supervised learning]/student-mat.csv", sep=';')
data = data[["G1","G2","G3","studytime","failures","absences"]]
#print(data.head())

predict = "G3"

x = np.array(data.drop([predict], 1))      #ModelInput / attributes on which model predicts
y = np.array(data[predict])             #ModelOutput / labels / prediction

x_train , x_test, y_train  , y_test = sklearn.model_selection.train_test_split(x, y, test_size =0.1)
#print("x_train: ",x_train);print("y_train: ",y_train);
#print("x_test: ",x_test);print("y_test: ",y_test)

"""
for _ in range(30):
    '''this loop is to train the model and save it with best score(accuracy).'''
    x_train , x_test, y_train  , y_test = sklearn.model_selection.train_test_split(x, y, test_size =0.1)
    linear = linear_model.LinearRegression()        #creating LinearRegression Model object
    linear.fit(x_train, y_train)                      #finding best fit line/train model according to dataset
    accuracy = linear.score(x_test, y_test)*100       #test the model accourding to dataset and returns accuracy
    print(accuracy)

    if(accuracy>best):
        best = accuracy
        ''' Saving a model using pickle.
        once you have saved the trained model, you don't need the codes to train it again(in this case codes from line 20 to 23) '''
        with open("student.pickle", "wb") as f:
            pickle.dump(linear, f)
"""     

'''when you trained your model and saved it using pickle, then you can use that saved model to predict outputs by just loading the saved model in an object
like did below. '''
pickle_in = open("E:/  ASHISH/StudioCode/Python/Projects/ML/1. Linear Regression [Supervised learning]/student.pickle","rb")
linear = pickle.load(pickle_in)

'''Coefficient(m) and Intercepts(c) of line y=mx+c '''
#print("Coefficient: \n", linear.coef_)
#print("Intercepts: \n",linear.intercept_)

''' till now we have trained our model accourding to our dataset '''

predictions = linear.predict(x_test)        #here we are predicting outputs
#print(predictions)

'''Plotting i/p & o/p on a graph'''
p = "G1"
style.use('ggplot')
pyplot.scatter(data[p],data[predict])
pyplot.xlabel(p)
pyplot.ylabel("FinalGrade")
pyplot.show()