import sklearn, math
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

Slabel = LabelEncoder()

data = pd.read_csv('E:/  ASHISH/StudioCode/Python/Projects/ML/3. Decision Tree [Supervised Learning]/train.csv')
data = data.drop(['PassengerId','Name','Ticket','SibSp','Parch','Cabin','Embarked'], 1)

data["Sex"] = Slabel.fit_transform(list(data.Sex))
data.Age = data.Age.fillna(math.floor(data.Age.median()))
#print(data.info())

x = data.drop(["Survived"], 1)
y = data["Survived"]
#print(x.head(), y.head())

model = tree.DecisionTreeClassifier()
model.fit(x, y)
#print(model.score(x, y))

TSlabel = LabelEncoder()
test = pd.read_csv('E:/  ASHISH/StudioCode/Python/Projects/ML/3. Decision Tree [Supervised Learning]/test.csv')

test["Sex"] = TSlabel.fit_transform(list(test.Sex))

test = test.drop(['PassengerId','Name','Ticket','SibSp','Parch','Cabin','Embarked'], 1)
test.Age = test.Age.fillna(math.floor(test.Age.median()))
test.Fare = test.Fare.fillna(math.floor(test.Fare.median()))
#print(test.info())

yp = model.predict(test)

final = pd.read_csv("E:/  ASHISH/StudioCode/Python/Projects/ML/3. Decision Tree [Supervised Learning]/test.csv")
final = final.drop(["Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"], 1)
final["Survived"] = yp

final.to_csv("E:/  ASHISH/StudioCode/Python/Projects/ML/3. Decision Tree [Supervised Learning]/Submission.csv", index=False)