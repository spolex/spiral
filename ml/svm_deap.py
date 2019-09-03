# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:29:40 2019

@author: isancmen

[LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

"""
from interfaces.reader_and_writer import load
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np

filename = "../output/archimedean_ds.h5"
mode = 'r'

# load data
X = load(filename, 'train_rd', mode)
# X = np.delete(X,4,axis=1)
# label preparation
y = load(filename, 'labels', mode)

# prepare datasets for training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0, shuffle=True)

# feature selection
# matlab https://www.mathworks.com/help/stats/relieff.html
# A sklearn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms.
# https://github.com/EpistasisLab/scikit-rebate
filter = ReliefF(n_features_to_select=8, n_neighbors=3)

clf = SVC()

paramgrid = {"svc__kernel": ["rbf"],
             "svc__C": np.logspace(-5, 5, num=25, base=10),
             "svc__gamma": np.logspace(-9, 0, num=25, base=10)}

# make pipeline
pipe = make_pipeline(filter, clf)

cv = GridSearchCV(pipe, paramgrid, cv=5)
cv.fit(X_train, y_train)

print(cv.best_params_)
print(cv.best_score_)

#make predictions
y_pred = cv.predict(X_test)
y_train_pred = cv.predict(X_train)

print("Training report")
print(classification_report(y_train_pred, y_train))
print("Test report")
print(classification_report(y_test, y_pred))
