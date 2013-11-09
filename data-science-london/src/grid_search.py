# -*- coding: utf-8 -*-

import os
import sys
from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
import numpy as np
import csv

def output_result(clf):
    test_feature_file = np.genfromtxt(open("../data/test.csv", "rb"), delimiter=",", dtype=float)

    test_features = []
    print "Id,Solution"
    i = 1
    for test_feature in test_feature_file:
        print str(i) + "," + str(int(clf.predict(test_feature)[0]))
        i += 1

def get_score(clf, train_features, train_labels):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_features, train_labels, test_size=0.4, random_state=0)

    clf.fit(X_train, y_train)
    print clf.score(X_test, y_test) 

def get_accuracy(clf, train_features, train_labels):
    scores = cross_validation.cross_val_score(clf, train_features, train_labels, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def grid_search(train_features, train_labels):
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    
    clf = GridSearchCV(svm.SVC(C=1), param_grid, n_jobs=-1)
    clf.fit(train_features, train_labels)
    print clf.best_estimator_
    

if __name__ == "__main__":
#    train_feature_file       = csv.reader(open("train.csv", "rb"))
#    train_label_file = csv.reader(open("trainLabels.csv", "rb"))
    train_feature_file = np.genfromtxt(open("../data/train.csv", "rb"), delimiter=",", dtype=float)
    train_label_file = np.genfromtxt(open("../data/trainLabels.csv", "rb"), delimiter=",", dtype=float)

    train_features = []
    train_labels = []
    for train_feature, train_label in zip(train_feature_file, train_label_file):
        train_features.append(train_feature)
        train_labels.append(train_label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)


    grid_search(train_features, train_labels)


#    clf.fit(train_features, train_labels)
#    output_result(clf)

