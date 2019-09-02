# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:29:40 2019

@author: isancmen

[LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

"""
from time import time
from interfaces.reader_and_writer import load
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, make_scorer

filename = "../output/archimedean_ds.h5"
mode = 'r'

# load data
X = load(filename, 'train_rd', mode)
# X = np.delete(X,4,axis=1)
# label preparation
y = load(filename, 'labels', mode)

# feature selection
# matlab https://www.mathworks.com/help/stats/relieff.html
# A sklearn-compatible Python implementation of ReBATE, a suite of Relief-based feature selection algorithms.
# https://github.com/EpistasisLab/scikit-rebate
filter = ReliefF(n_features_to_select=5, n_neighbors=3)
# filter = SelectKBest(chi2, k=6)


# As was the case with PCA, we need to perform feature scaling for LDA too. Execute the following script to do so
# Dimension reduction https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/

# predictive model
clf = SVC(kernel='rbf', gamma=0.2, C=10**(-4))

# make pipeline
pipe = make_pipeline(filter, clf)

# validation
cv = ShuffleSplit(n_splits=10, test_size=0.38, random_state=0)
cross_val_score(clf, X, y, cv=cv)


def classification_report_with_accuracy_score(y_true, y_pred):
    print(classification_report(y_true, y_pred))  # print classification report
    return accuracy_score(y_true, y_pred)  # return accuracy score


# Nested CV with parameter optimization
nested_score = cross_val_score(clf, X=X, y=y, cv=cv, scoring=make_scorer(classification_report_with_accuracy_score))
print(nested_score)
