# -*- coding: utf-8 -*-

import os
import sys
from sklearn import svm
import numpy as np
import csv

if __name__ == "__main__":
    #train_file = np.genfromtxt(open("../data/train.csv", "rb"), delimiter=",", dtype=float)
    train_file = csv.reader(open("../data/train.csv", "rb"))

    train_features = []
    train_labels   = []

    header = train_file.next()

    for train_row in train_file:
        delta  = train_row[1]
        starts = train_row[2:402]
        stops  = train_row[402:802]

        train_feature = delta + stops
        train_features.append(train_feature)
        train_labels.append(starts)

    clf = svm.SVR()
    clf.fit(
