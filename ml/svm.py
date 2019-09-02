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
filter = ReliefF(n_features_to_select=5, n_neighbors=3)
#filter = SelectKBest(chi2, k=6)



# As was the case with PCA, we need to perform feature scaling for LDA too. Execute the following script to do so
# Dimension reduction https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/

# predictive model
clf = SVC(kernel='rbf', gamma='auto', C=10-1)

# make pipeline
pipe = make_pipeline(filter, clf)
pipe.fit(X_train, y_train)


#make predictions
y_pred = pipe.predict(X_test)
y_train_pred = pipe.predict(X_train)

print("Training report")
print(classification_report(y_train_pred, y_train))
print("Test report")
print(classification_report(y_test, y_pred))