import sklearn, math
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

Slabel = LabelEncoder()

data = pd.read_csv('E:/  ASHISH/StudioCode/Python/Projects/ML/5. Naive Bayes [Supervised Learning]/train.csv')
data = data.drop(['PassengerId','Name','Ticket','SibSp','Parch','Cabin','Embarked'], 1)

data["Sex"] = Slabel.fit_transform(list(data.Sex))
data.Age = data.Age.fillna(math.floor(data.Age.mean()))
#print(data.info())

x = data.drop(["Survived"], 1)
y = data["Survived"]
#print(x.head(), y.head())

model = GaussianNB()
model.fit(x,y)
yt = model.predict(x)
print(metrics.accuracy_score(y, yt))

TSlabel = LabelEncoder()
test = pd.read_csv('E:/  ASHISH/StudioCode/Python/Projects/ML/5. Naive Bayes [Supervised Learning]/test.csv')

test["Sex"] = TSlabel.fit_transform(list(test.Sex))

test = test.drop(['PassengerId','Name','Ticket','SibSp','Parch','Cabin','Embarked'], 1)
test.Age = test.Age.fillna(math.floor(test.Age.mean()))
test.Fare = test.Fare.fillna(math.floor(test.Fare.mean()))
#print(test.info())

yp = model.predict(test)

final = pd.read_csv("E:/  ASHISH/StudioCode/Python/Projects/ML/5. Naive Bayes [Supervised Learning]/test.csv")
final = final.drop(["Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"], 1)
final["Survived"] = yp

final.to_csv("E:/  ASHISH/StudioCode/Python/Projects/ML/5. Naive Bayes [Supervised Learning]/Submission.csv", index=False)