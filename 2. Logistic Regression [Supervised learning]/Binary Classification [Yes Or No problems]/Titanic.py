import sklearn, math
import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("train.csv")
data = data.drop(["PassengerId","Name","Ticket","Fare","Cabin","Embarked"],1)

label = preprocessing.LabelEncoder()
Sex = label.fit_transform(list(data["Sex"]))
data["Sex"] = Sex

data.Age = data.Age.fillna(math.floor(data.Age.median()))
#print(data.head())

x = np.array(data.drop(["Survived"],1))
y = np.array(data["Survived"])

l = linear_model.LogisticRegression()
l.fit(x,y)
print("Accuracy: ",l.score(x,y)*100)

test = pd.read_csv("test.csv")
test = test.drop(["PassengerId","Name","Ticket","Fare","Cabin","Embarked"],1)

Sex = label.fit_transform(list(test["Sex"]))
test["Sex"] = Sex
#print(test.head())

test.Pclass = test.Pclass.fillna(math.floor(test.Pclass.median()))
test.Age = test.Age.fillna(math.floor(test.Age.median()))
test.SibSp = test.SibSp.fillna(math.floor(test.SibSp.median()))
test.Parch = test.Parch.fillna(math.floor(test.Parch.median()))
x_test = np.array(test)

yp = l.predict(x_test)
#print(yp)

final = pd.read_csv("test.csv")
final = final.drop(["Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"],1)
final["Survived"] = yp

final.to_csv("E:/  ASHISH/StudioCode/Python/Projects/ML/Practice/Submission.csv", index=False)