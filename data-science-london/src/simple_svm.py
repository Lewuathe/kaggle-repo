# -*- coding: utf-8 -*-

import os
import sys
from sklearn import svm
import numpy as np
import csv

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

    clf = svm.SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3,gamma=0.001, kernel="rbf", max_iter=-1, probability=False,random_state=None, shrinking=True, tol=0.001, verbose=False)

    clf.fit(train_features, train_labels)

    test_feature_file = np.genfromtxt(open("../data/test.csv", "rb"), delimiter=",", dtype=float)

    test_features = []
    print "Id,Solution"
    i = 1
    for test_feature in test_feature_file:
        print str(i) + "," + str(int(clf.predict(test_feature)[0]))
        i += 1
