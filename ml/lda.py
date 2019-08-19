# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:29:40 2019

@author: isancmen
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from interfaces.reader_and_writer import load

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline

filename="../output/archimedean_ds.h5"
mode='r'

#load data
X = load(filename,'train_rd',mode)
#label preparation
y = load(filename,'labels',mode)

anova_filter = SelectKBest(f_regression, k=5)

#prediction model        
clf = LinearDiscriminantAnalysis()

anova_lda = Pipeline([('anova', anova_filter), ('lda', clf)])
anova_lda.fit(X, y)

prediction = anova_lda.predict(X)
print(anova_lda.score(X, y))  

print(anova_lda['anova'].get_support())