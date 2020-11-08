import pandas as pd
import numpy as np
import sklearn, pickle
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style

data = pd.read_csv("E:/  ASHISH/StudioCode/Python/Projects/ML/2. K-Nearest Neighbors (Classification) [Un-Supervised leaning]/car.data")
#print(data.head())
 
#object which will change string into numeric (because our algorithmns understand numeric value, not string )
le = preprocessing.LabelEncoder() 

'''Below we are taking mention column and convering into list,
then we are transforming that list into appropriate interger values 

fit_transform return numpy array
'''
buying=le.fit_transform(list(data["buying"]))
maint=le.fit_transform(list(data["maint"]))
door=le.fit_transform(list(data["door"]))
persons=le.fit_transform(list(data["persons"]))
lug_boot=le.fit_transform(list(data["lug_boot"]))
safety=le.fit_transform(list(data["safety"]))
cls=le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)*100
print(accuracy)

predicted = model.predict(x_test)
