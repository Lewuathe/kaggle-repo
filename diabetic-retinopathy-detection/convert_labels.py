# -*- coding: utf-8 -*-

import os
import sys
import csv

if __name__ == "__main__":
    with open('trainLabels.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        with open('converted_labels.txt', 'w') as wf:
            writer = csv.writer(wf, lineterminator='\n', delimiter=' ')
            for row in reader:
                writer.writerow([row[0] + '.jpeg', row[1]])
            
            
