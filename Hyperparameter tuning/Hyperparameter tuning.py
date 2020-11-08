import sklearn
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
x = iris.data
y = iris.target


'''If you have less parameters, you can use GridSearchCV to find score of every possible combination of the parameters '''
"""clf = GridSearchCV(svm.SVC(gamma="auto"), {
    'C':[10,20,30],
    'kernel':['rbf','linear']
}, cv=5, return_train_score=False)
clf.fit(x, y)
#print(clf.cv_results_)
df = pd.DataFrame(clf.cv_results_)
print(df[['param_C', 'param_kernel', 'mean_test_score']])
print('Best fit Parameters: ',clf.best_params_,' | Best fit Score: ', clf.best_score_)"""

'''If you have larger parameters, you can use RandomizedSearchCV to try random values of parameter '''
"""rs = RandomizedSearchCV(svm.SVC(gamma="auto"), {
    'C':[10,20,30],
    'kernel':['rbf','linear']
}, cv=5, return_train_score=False, n_iter=2)
rs.fit(x, y)
#print(rs.cv_results_)
df = pd.DataFrame(rs.cv_results_)
print(df[['param_C', 'param_kernel', 'mean_test_score']])
print('Best fit Parameters: ',rs.best_params_,' | Best fit Score: ', rs.best_score_)
"""

'''If you want to try more than one model to compare scores or to choose which model is best fit'''
model_param = {
    'svm':{
        'model':svm.SVC(gamma="auto"),
        'params':{
            'C':[10,20,30],
            'kernel':['rbf','linear']
        }
    },
    'random_forest':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[1,5,10]
        }
    },
    'logistic_regression':{
        'model':LogisticRegression(solver='liblinear', multi_class='auto'),
        'params':{
            'C':[1,5,10]
        }
    }
}

scores = []
for model_name, mp in model_param.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x, y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)