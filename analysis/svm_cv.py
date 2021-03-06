# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:29:40 2019

@author: isancmen

"""
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF, MultiSURFstar

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import RFE


def svm_cv(X, y, rfe=True, paramgrid=None):
    """

    :param X:
    :param y:
    :param rfe:
    :param paramgrid:
    :return:
    """
    norm = preprocessing.StandardScaler()

    # feature selection
    fltr = RFE(ReliefF(), n_features_to_select=5, step=1)

    # predictive model
    model = SVC()

    # make pipeline
    pipe = make_pipeline(norm, fltr, model)

    param_grid = {
        'svc__kernel': ['rbf'],
        'svc__C': [1, 10, 10e1, 10e2, 10e3, 10e4],
        'svc__gamma': [0.1, 0.2, 0.3, 0.4, 0.5]
    } if paramgrid is None else paramgrid

    scores = ['accuracy']

    kf = KFold(n_splits=10, shuffle=True, random_state=4)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(pipe, param_grid, cv=kf, n_jobs=2,
                           scoring=score, return_train_score=False, verbose=10)
        clf.fit(X, y)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print(clf.best_score_)

        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
