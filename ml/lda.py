# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:29:40 2019

@author: isancmen

[LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from interfaces.reader_and_writer import load

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression,f_classif,chi2,mutual_info_classif,SelectFpr
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

import numpy as np
filename="../output/archimedean_ds.h5"
mode='r'

#load data
X = load(filename,'train_rd',mode)
#X = np.delete(X,4,axis=1)
#label preparation
y = load(filename,'labels',mode)

#prepare datasets for training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0, shuffle=True)

#As was the case with PCA, we need to perform feature scaling for LDA too. Execute the following script to do so
#Dimension reduction https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/

#anova_filter = SelectKBest(f_regression, k=5)
#anova_filter = SelectFpr(f_regression, alpha=0.01)

#predictive model        
lda = LDA()
lda.fit(X_train, y_train)

y_pred=lda.predict(X_test)
y_train_pred = lda.predict(X_train)

print('Test CM :')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy test ' + str(accuracy_score(y_test, y_pred)))
print('Train CM :')
cmt = confusion_matrix(y_train, y_train_pred)
print(cmt)
print('Accuracy train ' + str(lda.score(X_train, y_train)))

knn = KNeighborsClassifier(n_neighbors=3)
#anova_knn = Pipeline([('anova', anova_filter), ('knn', neigh)])
knn.fit(X_train, y_train) 

y_pred=knn.predict(X_test)
y_train_pred = knn.predict(X_train)

print('kNN Test CM :')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('kNN Accuracy test ' + str(accuracy_score(y_test, y_pred)))
print('kNN Train CM :')
cmt = confusion_matrix(y_train, y_train_pred)
print(cmt)
print('kNN Accuracy train ' + str(accuracy_score(y_train, y_train_pred)))
print(cross_val_score(lda, X, y, cv=3).mean()) 
print(cross_val_score(knn, X, y, cv=3).mean())   