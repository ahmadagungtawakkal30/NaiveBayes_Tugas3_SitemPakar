# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:23:38 2021

@author: ahmad
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from naivebayes3 import NaiveBayes


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X, y = datasets.make_classification(n_samples=5000, n_features=50, n_classes=2, random_state=500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=500)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("Naive Bayes classification accuracy", accuracy(y_test, predictions))