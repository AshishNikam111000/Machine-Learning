import sklearn, math
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

Slabel = LabelEncoder()

################################################# Cleaning Training Data Part ####################################
data = pd.read_csv('E:/  ASHISH/StudioCode/Python/Projects/ML/XGBoost/train.csv')
data = data.drop(['PassengerId','Name','Ticket','SibSp','Parch','Cabin','Embarked'], 1)

data["Sex"] = Slabel.fit_transform(list(data.Sex))
data.Age = data.Age.fillna(math.floor(data.Age.median()))
#print(data.info())
################################################# Creating Feaures & Targets Part ####################################

x = data.drop(["Survived"], 1)
y = data["Survived"]
#print(x.head(), y.head())

################################################# Training Part ####################################
xgb_class = xgb.XGBClassifier()
xgb_class.fit(x, y)
'''
train = xgb.DMatrix(x, label=y)
param = {
    'max_depth':4,
    'eta':0.3,
    'objective':'multi:softmax',
    'num_class':2
}
epochs = 10
model = xgb.train(param, train, epochs)
'''
################################################# Cleaning Testing Data Part ############################################
TSlabel = LabelEncoder()
test = pd.read_csv('E:/  ASHISH/StudioCode/Python/Projects/ML/XGBoost/test.csv')

test["Sex"] = TSlabel.fit_transform(list(test.Sex))

test = test.drop(['PassengerId','Name','Ticket','SibSp','Parch','Cabin','Embarked'], 1)
test.Age = test.Age.fillna(math.floor(test.Age.median()))
test.Fare = test.Fare.fillna(math.floor(test.Fare.median()))
#print(test.info())
#ptest = xgb.DMatrix(test)

################################################# Predict Part ############################################
yp = xgb_class.predict(test)
ap = pd.read_csv('E:/  ASHISH/StudioCode/Python/Projects/ML/XGBoost/Output.csv')
ap = ap.Survived
#print(ap, yp)
print(metrics.accuracy_score(ap, yp))

################################################# OutPut Part ############################################
final = pd.read_csv("E:/  ASHISH/StudioCode/Python/Projects/ML/XGBoost/test.csv")
final = final.drop(["Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"], 1)
final["Survived"] = yp

final.to_csv("E:/  ASHISH/StudioCode/Python/Projects/ML/XGBoost/Submission.csv", index=False)