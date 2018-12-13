# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:51:14 2018

@author: nitesh.yadav
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import r2_score, make_scorer
from sklearn.cross_validation import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

def DataLoad():
    """ loads data from CSV file """
    try:
        data = pd.read_csv(r'C:\Users\nitesh.yadav\Desktop\boston_housing\housing.csv')
        features = data.drop('MED_VAL', axis = 1)
        labels = data['MED_VAL']
        print("Housing dataset has {} data points with {} variables each.".format(*data.shape))
    except FileNotFoundError:
        print("File 'housing.csv' does not exist, please check the provided path.")
    return data, features, labels

def ExploreData(data, features, labels):
    """ Explores dataset using Statistics and graphs """
    print("Statistics for housing dataset:-") 
    print("Minimum price: ${:,.2f}".format(np.min(labels)))
    print("Maximum price: ${:,.2f}".format(np.max(labels)))
    print("Mean price: ${:,.2f}".format(np.mean(labels)))
    print("Median price: ${:,.2f}".format(np.median(labels)))
    print("Standard Deviation of price: ${:,.2f}".format(np.std(labels)))
    for var in features.columns:
        sbn.regplot(data[var], labels)
        plt.show()

def PerformanceMetric(labels_true, labels_predict):
    score = r2_score(labels_true, labels_predict)
    return score

def fitModel(features_train, labels_train):
    """ Fit the training data to the model using grid search """
    cv_set = ShuffleSplit(features_train.shape[0], n_iter = 10, test_size = .2, random_state = 0)
    regressor = DecisionTreeRegressor()
    params = {'max_depth' : np.arange(1, 11)}
    scoring_fun = make_scorer(PerformanceMetric)
    grid = GridSearchCV(regressor, params, scoring_fun, cv = cv_set)
    grid = grid.fit(features_train, labels_train)
    return grid.best_estimator_
    