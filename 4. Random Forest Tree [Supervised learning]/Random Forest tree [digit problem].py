import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
x = digits.data
y = digits.target

xt, xtt , yt, ytt = train_test_split(x,y,test_size=0.2)

r = RandomForestClassifier()
r.fit(xt,yt)
print(r.score(xtt, ytt))