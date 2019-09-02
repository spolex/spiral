# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:29:40 2019

@author: isancmen

[LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

"""
from time import time
from interfaces.reader_and_writer import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


filename = "../output/archimedean_ds.h5"
mode = 'r'

# load data
X = load(filename, 'train_rd', mode)
# X = np.delete(X,4,axis=1)
# label preparation
y = load(filename, 'labels', mode)

# prepare datasets for training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.38, random_state=0, shuffle=True)

# feature selection
# matlab https://www.mathworks.com/help/stats/relieff.html
# A sklearn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms.
# https://github.com/EpistasisLab/scikit-rebate
filter = ReliefF(n_features_to_select=5, n_neighbors=3)
#filter = SelectKBest(chi2, k=6)



# As was the case with PCA, we need to perform feature scaling for LDA too. Execute the following script to do so
# Dimension reduction https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/

# predictive model
svc = SVC(kernel='rbf',class_weight=None)

# make pipeline
pipe = make_pipeline(filter, svc)
#pipe.fit(X_train, y_train)

# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'svc__C': [10**(-5),10**(-5),10**(-5),10**(-5),10**(-5),10,10**2,10**3,10**4,10**5],
              'svc__gamma': [0.1,0.2,0.3,0.4,0.5,0.8,1.]}
clf = GridSearchCV(pipe,
                   param_grid, cv=5, iid=False)
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting ET on the test set")
t0 = time()
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print("Predicting ET on the train set")
t0 = time()
y_train_pred = clf.predict(X_train)
print("done in %0.3fs" % (time() - t0))



print(classification_report(y_train, y_train_pred))
print(confusion_matrix(y_train, y_train_pred))

# #############################################################################
# Qualitative evaluation of the predictions using matplotlib https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html