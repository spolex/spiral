# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:29:40 2019

@author: isancmen

"""
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.feature_selection import RFE
from sklearn import preprocessing


def svm_ga(X, y, rfe=True, paramgrid=None):

    # feature selection
    fltr = RFE(ReliefF(), n_features_to_select=5, step=0.5) if rfe else ReliefF(n_features_to_select=5, n_neighbors=3)

    clf = SVC()

    param_grid = {
        "svc__kernel": ["rbf"],
        'svc__C': [10e-2, 10e-1, 10, 10e1, 10e2, 10e3, 10e4],
        'svc__gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 1.1]
    } if paramgrid is None else paramgrid

    # make pipeline
    pipe = make_pipeline(preprocessing.StandardScaler(), fltr, clf)

    from evolutionary_search import EvolutionaryAlgorithmSearchCV
    cv = EvolutionaryAlgorithmSearchCV(estimator=pipe,
                                       params=param_grid,
                                       scoring="accuracy",
                                       cv=10,
                                       verbose=1,
                                       population_size=50,
                                       gene_mutation_prob=0.1,
                                       gene_crossover_prob=0.8,
                                       tournament_size=10,
                                       generations_number=25)
    cv.fit(X, y)

    print(cv.best_params_)
    print(cv.best_score_)
