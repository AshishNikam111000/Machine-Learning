import sklearn, math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

Slabel = LabelEncoder()

data = pd.read_csv('E:/  ASHISH/StudioCode/Python/Projects/ML/4. Random Forest Tree [Supervised learning]/train.csv')
data = data.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], 1)

data["Sex"] = Slabel.fit_transform(list(data.Sex))
data.Age = data.Age.fillna(math.floor(data.Age.median()))
#print(data.info())

x = data.drop(["Survived"], 1)
y = data["Survived"]
#print(x.head(), y.head())

r = RandomForestClassifier()
r.fit(x,y)
yt = r.predict(x)
print(metrics.accuracy_score(y, yt))

TSlabel = LabelEncoder()
test = pd.read_csv('E:/  ASHISH/StudioCode/Python/Projects/ML/4. Random Forest Tree [Supervised learning]/test.csv')

test["Sex"] = TSlabel.fit_transform(list(test.Sex))

test = test.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], 1)
test.Age = test.Age.fillna(math.floor(test.Age.median()))
test.Fare = test.Fare.fillna(math.floor(test.Fare.median()))
print(test.info())

yp = r.predict(test)

final = pd.read_csv("E:/  ASHISH/StudioCode/Python/Projects/ML/4. Random Forest Tree [Supervised learning]/test.csv")
final = final.drop(["Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"], 1)
final["Survived"] = yp

final.to_csv("E:/  ASHISH/StudioCode/Python/Projects/ML/4. Random Forest Tree [Supervised learning]/Submission.csv", index=False)