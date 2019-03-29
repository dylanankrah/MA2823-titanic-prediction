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

data = []                                       # this variable will hold the data
with open('train.csv', 'r') as csvfile:         # loading the csv file
    csvreader = csv.reader(csvfile)
    next(csvreader)                             # first line skipping
    for row in csvreader:                       # Skip through each rown in csv file
        data.append([elt for elt in ''.join(row[0:]).split(',')])    # adding each row to the data variable
X = np.array(data)                              # then convert to an array
y = X[:,1].astype(int)                          # save labels to y
X = np.delete(X,1,1)                            # remove survival column from matrix X (this is what we're looking to determine)

# (II) ---------- data pre-processing ----------

# (1) feature deletion (deleting ID's, first names, cabin and ticket numbers)

""" structure of our data: [ID's, class, firstname, title+lastname, sex, age, sibsp, parch, ticket number, fare, cabin number, port_emb]
    NB: the reason names are split this way is the way we used the comma as a separator a few lines above. We'll fix this next. """

X = np.delete(X,0,1)                            # deleting ID's
X = np.delete(X,7,1)                            # and ticket numbers (note that the position of the feature changes...)
X = np.delete(X,8,1)                            # and cabin numbers 
X = np.delete(X,1,1)                            # deleting first names

""" strucutre of our data: [class, title+lastname, sex, age, sibsp, parch, fare, port_emb] """

# (2) feature engineering : keeping title statuses

for row in X:
    row[1] = row[1].split(" ")[1]               # since every title+lastname starts with a " " (because of formatting reasons), this code will only keep the word after it, therefore the title-status
    
""" structure of our data: [class, title-status, sex, age, sibsp, parch, fare, port_emb] """

# (3) feature normalization/shaping (title-status, sex, fare, port, family)
    
# - fares
fares = []                                      # will hold fares of all passengers
for k in range(len(X)):
    fares.append(X[k,6])                        # listing all the fares
max_fare = max(fares)                           # calculating max fare in test.csv (will be used for both normalizations)
for k in range(len(X)):
    X[k,6] = float(X[k,6]) / float(max_fare)    # normalizing fares

titles = []                                     # will contain the list of all titles of every single passenger
sexes = []                                      # same for sexes (supposedly only two types)

# - titles & sexes
for row in X:
    titles.append(row[1])
    sexes.append(row[2])
    
titles, sexes = np.unique(titles), np.unique(sexes)       # returns the unique elements of those (therefore the different titles and sexes, once)

""" (titles) [Out]: array(['Capt.', 'Col.', 'Don.', 'Dr.', 'Jonkheer.', 'Lady.', 'Major.',
        'Master.', 'Miss.', 'Mlle.', 'Mme.', 'Mr.', 'Mrs.', 'Ms.', 'Rev.',
        'Sir.', 'the'], dtype='<U9'),
                    array([  1,   2,   1,   7,   1,   1,   2,  40, 182,   2,   1, 517, 125,
         1,   6,   1,   1], dtype=int64)) 
    
    (sexes)  [Out]: array(['female', 'male'], dtype='<U6') """
    
for row in X:
    for k in range(len(titles)):
        if row[1] == titles[k]:                
            row[1] = k                         # we're now mapping each position number in the titles array. Therefore, Master. is equivalent to 7, Miss. to 8, Mr. to 11 and Mrs. to 12...
            
    for k in range(len(sexes)):   
        if row[2] == sexes[k]:                 
            row[2] = k                         # we're now mapping each position number in the sexes array. From now on, male is equivalent to 1, female to 0
    if row[-1] == 'C':
        row[-1] = 0
    if row[-1] == 'Q':
        row[-1] = 1
    if row[-1] == 'S':
        row[-1] = 2                            # we're making it so that the embarkation port data is also a digit

""" (X[0]) [Output]: array(['3', '11', '1', '22', '1', '0', '0.07754010695187166', '2'], dtype='<U62')
    <=>
    The first individual was a male (1) third-class passenger ('11' = Mr.), aged 22, etc """
   
# (4) feature fixing (missing values, NaN...)

""" Let's consider an example: Mr. Philip Kiernan. A key information is missing: the age. Among all the selected features so far, we need to take care of missing values. Using np.mean requires
the matrix to contain the same type of value everywhere, which isn't the case (float transformation will come right after this, but '' values prevents us from proceeding). We will calculate values another way. """

m,n = X.shape                                  # getting out datasets shape at this point 
X_f = np.zeros((m,n))                          # X_f will hold our pre-processed dataset at the end of this step
for k in range(m):
    for j in range(n):
        if (X[k,j] == ''):
            X_f[k,j] = np.nan                  # if there's a missing value, replace it with "NaN"
        else:
            X_f[k,j] = float(X[k,j])           # else, convert the string value to a float64
            
imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
X_f = imp.fit_transform(X_f)                   # using Imputer to "fill" the missing values
    
# (5) creating "family size" feature (with features 4 & 5)

for k in range(m):
    X_f[k,4] += X_f[k,5]
X_f = np.delete(X_f,5,1)                       # deleting attribute 5 (feature 4 = family size)

# (6) dataset status at this point | visualisation

""" X_f is a 891x7 matrix containing only the [class, title-status, sex, age, familysize, fare, port_emb] features, with only float values, and a normalized-to-1 fare value with 0 being null and 1 being the most expensive. More info on data meaning in the report. """

# (III) ---------- learning algorithm ----------

# REMINDER: dataset = X_f, labels = y

classifiers = []

classifiers.append(RandomForestClassifier()) # random forest
classifiers.append(KNeighborsClassifier()) # k-NN with default 5 neighbors
classifiers.append(LogisticRegression()) # logistic regression
classifiers.append(LinearDiscriminantAnalysis()) # LDA

classifiers.append(VotingClassifier(estimators=[('rfc',classifiers[0]),('knn',classifiers[1]), ('lr',classifiers[2]), ('lda',classifiers[3])], voting='hard')) # ensemble learning
classifiers.append(VotingClassifier(estimators=[('rfc',classifiers[0]),('knn',classifiers[1]), ('lr',classifiers[2]), ('lda',classifiers[3])], voting='soft')) # ensemble learning

classifiers.append(MLPClassifier(hidden_layer_sizes=(100,50,),activation='logistic',solver='adam',shuffle=True))
classifiers.append(MLPClassifier(hidden_layer_sizes=(200,100,50,),activation='relu',solver='adam',shuffle=True)) # examples of neural networks

classifiers.append(VotingClassifier(estimators=[('rfc',classifiers[0]),('knn',classifiers[1]), ('lr',classifiers[2]), ('lda',classifiers[3]), ('mlp1',classifiers[6]),('mlp2',classifiers[7])], weights=[2,1,2,2,2,1], voting='soft')) # global voting classifier

# since kNN's and MLP classifiers are not that efficient, we weight them less in the decision step

classifiers.append(LinearDiscriminantAnalysis(solver='lsqr')) # extra LDA classifier

k_fold = cross_validation.KFold(len(X_f), 20, shuffle=True) # kfold for cross validation

cResults = []                                  # will contain one arrow for each classifier with several tests within each of them
for classifier in classifiers:
    cResults.append(cross_val_score(classifier, X_f, y=y, scoring = "accuracy", cv=k_fold))

cMeans = []
cStd = []
for cResult in cResults:
    cMeans.append(cResult.mean())
    cStd.append(cResult.std())

cRes = pd.DataFrame({"CrossValMeans":cMeans,"CrossValerrors": cStd,"Algorithm":["RandomForest","KNearestNeighboors","LogisticRegression","LinearDiscriminantAnalysis","VotingClassifierHard","VotingClassifierSoft","100-50-1_SigmoidADAM_MLP","200-100-50-1_ReluADAM_MLP","VotingClassifierTotal", "LDA_lsqr"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cRes, palette="Set3",orient = "h",**{'xerr':cStd}) # plots the efficiency of our classifiers
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
plt.show()

print(cMeans)

# (IV) ---------- testing phase ----------

datatest = []                                   # this variable will hold the data
with open('test.csv', 'r') as csvfile:          # loading the csv file
    csvreader = csv.reader(csvfile)
    next(csvreader)                             # first line skipping
    for row in csvreader:                       # Skip through each rown in csv file
        datatest.append([elt for elt in ''.join(row[0:]).split(',')])    # adding each row to the data variable
Xtest = np.array(datatest)                      # then convert to an array

Xtest = np.delete(Xtest,0,1)                    # deleting ID's
Xtest = np.delete(Xtest,7,1)                    # and ticket numbers
Xtest = np.delete(Xtest,8,1)                    # and cabin numbers 
Xtest = np.delete(Xtest,1,1)                    # deleting first names

for row in Xtest:
    row[1] = row[1].split(" ")[1]               # since every title+lastname starts with a " " (because of formatting reasons), this code will only keep the word after it, therefore the title-status

for k in range(len(Xtest)):
    if(Xtest[k,6] == ''):
        Xtest[k,6] = np.nan
    else:
        Xtest[k,6] = float(Xtest[k,6]) / float(max_fare)    # normalizing fares with previous maximum value

for row in Xtest:
    for k in range(len(titles)):
        if row[1] == titles[k]:                
            row[1] = k                         # we're now mapping each position number in the titles array. Therefore, Master. is equivalent to 7, Miss. to 8, Mr. to 11 and Mrs. to 12...
            
    for k in range(len(sexes)):   
        if row[2] == sexes[k]:                 
            row[2] = k                         # we're now mapping each position number in the sexes array. From now on, male is equivalent to 1, female to 0
    if row[-1] == 'C':
        row[-1] = 0
    if row[-1] == 'Q':
        row[-1] = 1
    if row[-1] == 'S':
        row[-1] = 2                            # we're making it so that the embarkation port data is also a digit
        
m,n = Xtest.shape                              # getting out datasets shape at this point 
Xtest_f = np.zeros((m,n))                      # Xtest_f will hold our pre-processed test dataset at the end of this step

for k in range(m):
    for j in range(n):
        print(k,j)
        if (Xtest[k,j] == ''):
            Xtest_f[k,j] = np.nan  # if there's a missing value, replace it with "NaN"
        if (Xtest[k,j] == 'Dona.'):
            Xtest_f[k,j] = np.nan  # one specific exception for the dataset here
        if (Xtest[k,j] != '') and (Xtest[k,j] != 'Dona.'):
            Xtest_f[k,j] = float(Xtest[k,j])  # else, convert the string value to a float64

mp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
Xtest_f = imp.fit_transform(Xtest_f)          # using Imputer to "fill" the missing values with mean values

for k in range(m):
    Xtest_f[k,4] += Xtest_f[k,5]

Xtest_f = np.delete(Xtest_f,5,1)              # deleting attribute 5 (feature 4 = family size)

""" best classifier: classifiers[5] (soft votingclassifier) (see report) """

passenger_id_test = [892 + i for i in range(418)]

classifiers[5].fit(X_f,y)                     # fitting the SoftVotingClassifier
predictions = classifiers[5].predict(Xtest_f) # making a prediction using the classifier
predictions = predictions.astype(int)
submission = pd.DataFrame({"PassengerId" : passenger_id_test, "Survival" : predictions})
submission.to_csv("sub_voting_soft.csv", index=False) # exporting to a submittable file

classifiers[2].fit(X_f,y)                    # fitting the LogisticRegression classifier
predictions = classifiers[2].predict(Xtest_f)
predictions = predictions.astype(int)
submission = pd.DataFrame({"PassengerId" : passenger_id_test, "Survival" : predictions})
submission.to_csv("sub_logregression.csv", index=False)

classifiers[9].fit(X_f,y)                    # fitting the second LDA classifier
predictions = classifiers[9].predict(Xtest_f)
predictions = predictions.astype(int)
submission = pd.DataFrame({"PassengerId" : passenger_id_test, "Survival" : predictions})
submission.to_csv("sub_lda_lsqr.csv", index=False)


# (V) ---------- data visualization ----------



    
        