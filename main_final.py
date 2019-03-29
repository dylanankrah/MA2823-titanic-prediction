# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import cross_validation
import csv as csv
from classify import classify

from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
from matplotlib import colors

# (I) ---------- data loading ----------

def dataOpener(datapath):
    
    data = []                                       # this variable will hold the data
    with open(datapath, 'r') as csvfile:            # loading the csv file
        csvreader = csv.reader(csvfile)
        next(csvreader)                             # first line skipping
        for row in csvreader:                       # Skip through each rown in csv file
            data.append([elt for elt in ''.join(row[0:]).split(',')])    # adding each row to the data variable
    return(np.array(data))

X = dataOpener('train.csv')
y = X[:,1].astype(int)                          # save labels to y
X = np.delete(X,1,1)                            # remove survival column from matrix X (this is what we're looking to determine)

# (II) ---------- data pre-processing ----------

fares = []                                      # will hold fares of all passengers
for k in range(len(X)):
    fares.append(X[k,6])                        # listing all the fares
max_fare = max(fares)                           # calculating max fare in train data for future normalization

titles = []                                     # will contain the list of all titles of every single passenger
sexes = []                                      # same for sexes (supposedly only two types)
for row in X:
    titles.append(row[1])
    sexes.append(row[2])
    
titles, sexes = np.unique(titles), np.unique(sexes)       # returns the unique table of those (therefore the different titles and sexes, once)

def featureEngineering(X):
    
    X = np.delete(X,0,1)                            # deleting ID's
    X = np.delete(X,7,1)                            # and ticket numbers (note that the position of the feature changes...)
    X = np.delete(X,8,1)                            # and cabin numbers 
    X = np.delete(X,1,1)                            # deleting first names
    
    for row in X:
        row[1] = row[1].split(" ")[1]               # since every title+lastname starts with a " " (because of formatting reasons), this code will only keep the word after it, therefore the title-status
        
    for k in range(len(X)):
        X[k,6] = float(X[k,6]) / float(max_fare)    # normalizing fares
    
    for row in X:
        for k in range(len(titles)):
            if row[1] == titles[k]:                
                row[1] = k                     # we're now mapping each position number in the titles array. Therefore, Master. is equivalent to 7, Miss. to 8, Mr. to 11 and Mrs. to 12...
            
        for k in range(len(sexes)):   
            if row[2] == sexes[k]:                 
                row[2] = k                         # we're now mapping each position number in the sexes array. From now on,    male is equivalent to 1, female to 0
        if row[-1] == 'C':
            row[-1] = 0
        if row[-1] == 'Q':
            row[-1] = 1
        if row[-1] == 'S':
            row[-1] = 2                            # we're making it so that the embarkation port data is also a digit
            
    return(X)