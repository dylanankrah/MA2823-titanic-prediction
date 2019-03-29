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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

# (I) ---------- data loading ----------


def dataOpener(datapath):
    data = []
    with open(datapath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:                       # Skip through each rown in csv file
            data.append([elt for elt in ''.join(row[0:]).split(',')])
    X = np.array(data)
    return(X)

X = dataOpener('train.csv')
y = X[:,1].astype(int)                          # save labels to y
X = np.delete(X,1,1)                            # remove survival column from matrix X (this is what we're looking to determine)

# (II) ---------- data pre-processing ----------

def processing(X):
    
    # (1) feature deletion (deleting ID's, first names, cabin and ticket numbers)

    """ structure of our data: [ID's, class, firstname, title+lastname, sex, age, sibsp, parch, ticket number, fare, cabin number, port_emb]
    NB: the reason names are split this way is the way we used the comma as a separator a few lines above. We'll fix this next. """
    
    X = np.delete(X,0,1)                            # deleting ID's
    X = np.delete(X,7,1)                            # and ticket numbers (note that the position of the feature changes...)
    X = np.delete(X,8,1)                            # and cabin numbers 
    X = np.delete(X,1,1)                            # deleting first names
    
    """ strucutre of our data: [class, title+lastname, sex, age, sibsp, parch, fare, port_emb] """

    # (2) feature engineering : creating new features. The idea is to create a new feature to replace "Name", that would contain the information about the title-status of the person

    for row in X:
        row[1] = row[1].split(" ")[1]               # since every title+lastname starts with a " " (because of formatting reasons), this code will only keep the word after it, therefore the title-status
        
    """ strucutre of our data: [class, title+lastname, sex, age, sibsp, parch, fare, port_emb] """
    
    # (3) feature normalization/shaping (title-status, sex, fare, port, family)
    
    """ NB: there might be a correlation between port of embarkation and passenger fare. We will try to verify it using dimensionality reduction. """
    
    fares = []                                      # will hold fares of all passengers
    for k in range(len(X)):
        fares.append(X[k,6])                        # listing all the fares
    max_fare = max(fares)                           # calculating mean global mean fare (no fare missing)
    for k in range(len(X)):
        X[k,6] = float(X[k,6]) / float(max_fare)    # normalizing fares to 1


    titles = []                                     # will contain the list of all titles of every single passenger
    sexes = []                                      # same for sexes (supposedly only two types)

    for row in X:
        titles.append(row[1])
        sexes.append(row[2])
    
    titles, sexes = np.unique(titles), np.unique(sexes)       # returns the unique elements of those (therefore the different titles and sexes, once)

    """ (titles) [Out]: array(['Capt.', 'Col.', 'Don.', 'Dr.', 'Jonkheer.', 'Lady.', 'Major.',
            'Master.', 'Miss.', 'Mlle.', 'Mme.', 'Mr.', 'Mrs.', 'Ms.', 'Rev.',
            'Sir.', 'the'], dtype='<U9'),
    array([  1,   2,   1,   7,   1,   1,   2,  40, 182,   2,   1, 517, 125,
           1,   6,   1,   1], dtype=int64)) 
    
        (sexes) [Out]: array(['female', 'male'], dtype='<U6')
    
        Now, there's almost only Masters (40), Miss (182), Mr and Mrs (517 and 125)
        """
    for row in X:
        for k in range(len(titles)):
            if row[1] == titles[k]:                
                row[1] = k                     # we're now mapping each position number in the titles array. Therefore, Master. is equivalent to 7, Miss. to 8, Mr. to 11 and Mrs. to 12...
            
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
    The first individual was a male (1) third-class passenger ('11' = Mr.), aged 22, etc
    
    NB: the report will contain a table with more info about the feature selection process
    """
    return(X)


X = processing(X)



# (4) feature fixing (missing values, NaN...)

""" Let's consider an example: Mr. Philip Kiernan. A key information is missing: the age. Among all the selected features so far, we need to take care of missing values. Using np.mean requires
the matrix to contain the same type of value everywhere, which isn't the case (float transformation will come right after this, but '' values prevents us from proceeding). We will calculate values another way. """

m,n = X.shape                                  # getting out datasets shape at this point 
X_f = np.zeros((m,n))          # X_f will hold our pre-processed dataset at the end of this step

for k in range(m):
    for j in range(n):
        if (X[k,j] == ''):
            X_f[k,j] = np.nan  # if there's a missing value, replace it with "NaN"
        else:
            X_f[k,j] = float(X[k,j])    # else, convert the string value to a float64
            
imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
X_f = imp.fit_transform(X_f)            # using Imputer to "fill" the missing values
    
# (bonus) creating "family size" feature (with features 4 & 5)

for k in range(m):
    X_f[k,4] += X_f[k,5]

X_f = np.delete(X_f,5,1)                       # deleting attribute 5 (feature 4 = family size)
    

# (5) dataset status at this point | visualisation

""" X_f is a 891x7 matrix containing only the [class, title-status, sex, age, familysize, fare, port_emb] features, with only float values, and a normalized-to-1 fare value with 0 being null and 1 being the most expensive. More info on data meaning in the report. """


# (III) ---------- dimensionality reduction (optional) ----------

# PCA? mandatory?

# (IV) ---------- learning algorithm ----------

# REMINDER: dataset = X_f, labels = y
"""
classifiers = []

classifiers.append(RandomForestClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())

classifiers.append(VotingClassifier(estimators=[('rfc',classifiers[0]),('knn',classifiers[1]), ('lr',classifiers[2]), ('lda',classifiers[3])], voting='hard'))
classifiers.append(VotingClassifier(estimators=[('rfc',classifiers[0]),('knn',classifiers[1]), ('lr',classifiers[2]), ('lda',classifiers[3])], voting='soft'))

classifiers.append(MLPClassifier(hidden_layer_sizes=(100,50,),activation='logistic',solver='adam',shuffle=True))
classifiers.append(MLPClassifier(hidden_layer_sizes=(200,100,50,),activation='relu',solver='adam',shuffle=True))

classifiers.append(VotingClassifier(estimators=[('rfc',classifiers[0]),('knn',classifiers[1]), ('lr',classifiers[2]), ('lda',classifiers[3]), ('mlp1',classifiers[6]),('mlp2',classifiers[7])], weights=[2,1,2,2,2,1], voting='soft'))

# weighted version: according to results?

# kfold for cross validation
k_fold = cross_validation.KFold(len(X_f), 20, shuffle=True)

cResults = []                                  # will contain one arrow for each classifier with several tests within each of them
for classifier in classifiers:
    cResults.append(cross_val_score(classifier, X_f, y=y, scoring = "accuracy", cv=k_fold))

cMeans = []
cStd = []
for cResult in cResults:
    cMeans.append(cResult.mean())
    cStd.append(cResult.std())

cRes = pd.DataFrame({"CrossValMeans":cMeans,"CrossValerrors": cStd,"Algorithm":["RandomForest","KNearestNeighboors","LogisticRegression","LinearDiscriminantAnalysis","VotingClassifierHard","VotingClassifierSoft","100-50-1_SigmoidADAM_MLP","200-100-50-1_ReluADAM_MLP","VotingClassifierTotal"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cRes, palette="Set3",orient = "h",**{'xerr':cStd})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
plt.show()

print(cMeans)
"""