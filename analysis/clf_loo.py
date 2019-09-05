# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:29:40 2019

@author: isancmen

"""

from sklearn import preprocessing
from skrebate import ReliefF
from sklearn.feature_selection import RFE

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import LeaveOneOut, KFold


def analysis_loo(X, y, score='accuracy'):
    """

    :param score:
    :param X:
    :param y:
    :return:
    """
    # scale data
    norm = preprocessing.StandardScaler()

    # feature selection
    # A sklearn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms.
    # https://github.com/EpistasisLab/scikit-rebate
    fltr = RFE(ReliefF(), n_features_to_select=5, step=0.5)

    loo = LeaveOneOut()
    kf = KFold(n_splits=10, shuffle=True, random_state=4)
    # predictive model
    clf = SVC(kernel='rbf', gamma=0.1, C=10**4)

    # make pipeline
    pipe = make_pipeline(norm, fltr, clf)
    cv = cross_validate(pipe, X, y, cv=loo, scoring=score, return_train_score=True)
    print("train score svm")
    print(cv['train_score'].mean())
    print("test score svm")
    print(cv['test_score'].mean())

    # RandomForest
    clf = RandomForestClassifier(n_estimators=30)

    # make pipeline
    pipe = make_pipeline(norm, fltr, clf)

    cv = cross_validate(pipe, X, y, cv=loo, scoring=score, return_train_score=True)
    print("train score Random forest")
    print(cv['train_score'].mean())
    print("test score Random forest")
    print(cv['test_score'].mean())

    # Knn
    clf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', metric='euclidean')
    # make pipeline
    pipe = make_pipeline(norm, fltr, clf)
    cv = cross_validate(pipe, X, y, cv=loo, scoring=score, return_train_score=True)

    print("train score Knn")
    print(cv['train_score'].mean())
    print("test score Knn")
    print(cv['test_score'].mean())

    # predictive model
    clf = LDA()

    # make pipeline
    pipe = make_pipeline(norm, fltr, clf)
    cv = cross_validate(pipe, X, y, cv=loo, scoring=score, return_train_score=True)

    print("train score LDA")
    print(cv['train_score'].mean())
    print("test score LDA")
    print(cv['test_score'].mean())
