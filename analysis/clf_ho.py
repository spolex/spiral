# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:29:40 2019

@author: isancmen

[LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

"""
from loaders.reader_and_writer import load

from sklearn import preprocessing
from skrebate import ReliefF
from sklearn.feature_selection import RFE

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def analysis_ho(X, y):
    """

    :return:
    """
    # scale data
    # X = preprocessing.scale(X)
    # normalizer
    norm = preprocessing.StandardScaler()

    # prepare splits for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.38, random_state=4, shuffle=True)

    # feature selection
    # A sklearn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms.
    # https://github.com/EpistasisLab/scikit-rebate
    fltr = RFE(ReliefF(), n_features_to_select=5, step=0.5)

    # predictive model
    clf = SVC(kernel='rbf', gamma=0.1, C=10**4)

    # make pipeline
    pipe = make_pipeline(norm, fltr, clf)
    pipe.fit(X_train, y_train)

    print("SVM scoring...")
    # make predictions
    y_pred = pipe.predict(X_test)
    y_train_pred = pipe.predict(X_train)

    # evaluation
    print("Training report")
    print(classification_report(y_train_pred, y_train))
    print(confusion_matrix(y_train_pred, y_train))

    print("Test report")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # RandomForest
    clf = RandomForestClassifier(n_estimators=30)

    # make pipeline
    pipe = make_pipeline(norm, fltr, clf)
    pipe.fit(X_train, y_train)
    print("Random forest scoring")

    # make predictions
    y_pred = pipe.predict(X_test)
    y_train_pred = pipe.predict(X_train)

    # evaluation
    print("Training report")
    print(classification_report(y_train_pred, y_train))
    print(confusion_matrix(y_train_pred, y_train))

    print("Test report")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Knn
    clf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', metric='euclidean')

    # make pipeline
    pipe = make_pipeline(norm, fltr, clf)
    pipe.fit(X_train, y_train)

    print("Knn scoring")

    # make predictions
    y_pred = pipe.predict(X_test)
    y_train_pred = pipe.predict(X_train)

    # evaluation
    print("Training report")
    print(classification_report(y_train_pred, y_train))
    print(confusion_matrix(y_train_pred, y_train))

    print("Test report")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # predictive model
    clf = LDA()

    # make pipeline
    pipe = make_pipeline(norm, fltr, clf)
    pipe.fit(X_train, y_train)

    # make predictions
    y_pred = pipe.predict(X_test)
    y_train_pred = pipe.predict(X_train)

    print("LDA scoring")
    # evaluation
    print("Training report")
    print(classification_report(y_train_pred, y_train))
    print(confusion_matrix(y_train_pred, y_train))

    print("Test report")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
