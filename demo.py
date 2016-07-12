from sklearn import cross_validation
from sklearn import svm
from sklearn import preprocessing
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from sklearn.grid_search import GridSearchCV
j = 0
x = np.zeros(10)
a = np.zeros(10)
b = []
f_in = open('scop_new.data')
f_out = open('new_file', 'w')
for line in f_in:
    new_str = ''.join(line.split(',')[1:])
    f_out.write(new_str)
print f_out
from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file("new_file")
X_norm = preprocessing.normalize(X, norm='l2', axis=1, copy=True)

deg =[]
c = []
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
for i in range(-5,5):
    x[j] = j
    classifier = svm.SVC(kernel='poly', degree =x[j], C=100,coef0=1).fit(X_train, y_train)
    a[j] = np.mean(cross_validation.cross_val_score(classifier, X, y, cv=5, scoring='roc_auc'))
    j = j + 1
print a
plt.plot(x,a)
#plt.xscale('log')
plt.xlabel('degree')
plt.ylabel('Accuracy')
plt.title('C=100')
plt.show()