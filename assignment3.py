from sklearn import cross_validation
from sklearn import svm
from sklearn import preprocessing
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from sklearn.grid_search import GridSearchCV
j = 0
x = []
a = []
b = []
f_in = open('scop_new.data')
f_out = open('new_file', 'w')
for line in f_in:
    new_str = ''.join(line.split(',')[1:])
    f_out.write(new_str)
print f_out
from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file("new_file")
deg =[]
c = []

####### generating data for 3d plot #######

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
for i in range(-5,5):
   
   for k in range(-5,5):
        x_temp = 0.001
        c_temp = 10**k
        c.append(c_temp)
        classifier = svm.SVC(kernel='rbf',gamma=x_temp, C=c_temp).fit(X_train, y_train)
    
        a.append(np.mean(cross_validation.cross_val_score(classifier, X, y, cv=5, scoring='roc_auc')))
        np.savetxt('acc1.txt',a)
        x.append(x_temp)
np.savetxt('C.txt',c)
np.savetxt('gamma.txt',x)
        



for i in range(1,11):
    deg_temp = i
    for k in range(-5,5):
            
        c_temp = 10**k
        
        classifier = svm.SVC(kernel='poly',degree=deg_temp, C=c_temp,coef0=1).fit(X_train, y_train)
        b.append(np.mean(cross_validation.cross_val_score(classifier, X, y, cv=5, scoring='roc_auc')))
        np.savetxt('acc2.txt',b)
        deg.append(i)
np.savetxt('C.txt',c)
np.savetxt('degree.txt',deg)
        


####### non-normalized ###########
param_grid = [
{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},]
classifier = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid)
print np.mean(cross_validation.cross_val_score(classifier, X, y, cv=5, scoring = 'roc_auc'))


#######kernelmatrix implementation######
Xt=np.transpose(X)
km = (X*Xt)
km = km.toarray()
plt.matshow(km,origin='lower')
plt.suptitle('Non-Normalized')
plt.show()
