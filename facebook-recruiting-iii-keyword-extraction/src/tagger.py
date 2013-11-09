# -*- coding: utf-8 -*-
import nltk
import sklearn
import csv
import re
import numpy as np

import os
import sys

if __name__ == "__main__":
#
#   Training data from Train.csv
#   Id, Title, Body, Tag
#
    print "Reading start"
    train_file = csv.reader(open("Train.csv", "rb"))
    train_header = train_file.next()

    test_file = csv.reader(open("Test.csv", "rb"))
    test_header = test_file.next()

    result_file = open("Result.csv", "w")
    result_file.write('"Id","Tags"\n')

    traindata = []
    testdata  = []
    docs      = []
    
    print "Train Start"
    i = 0
    for data in train_file:
        tokens = re.split(r"\W+", nltk.clean_html(data[2]))
        #tokens = nltk.word_tokenize(nltk.clean_html(data[2]))
        docs.append(tokens)
        i += 1
        if i > 100000:
            break

    print "Make collection start"
    # Make the collection for calculating TF-IDF
    collection = nltk.TextCollection(docs)
    
    print "Testing data start"

    for data in test_file:
        title_tokens = nltk.word_tokenize(data[1])
        tokens = re.split(r"\W+", nltk.clean_html(data[2]))
        #tokens = nltk.word_tokenize(nltk.clean_html(data[2]))
        for title_token in title_tokens:
            for i in range(0, 10):
                tokens.append(title_token)
        
        uniqTokens = set(tokens)
      
        tf_idf_scores = {}
        for token in uniqTokens:
            tf_idf_scores[token] = collection.tf_idf(token, tokens)

        sorted_tf_idf_scores = sorted(tf_idf_scores.items(), key=lambda x:x[1])

        keywords = [ k for k, v in sorted_tf_idf_scores if v > 0.1]
        if len(keywords) <= 0:
            keywords = [ sorted_tf_idf_scores[-1][0] ]

        result_file.write("%s,\"%s\"\n" % (data[0], " ".join(keywords)))

        

        
