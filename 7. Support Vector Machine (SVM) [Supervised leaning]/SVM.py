import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()
#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

'''Creating objects of Support Vector Classifier(SVC) '''
clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_predict)*100
print(accuracy)