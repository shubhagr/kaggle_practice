# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 17:01:04 2018

@author: Shubham Agarwal (shubhagr)
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:, 1:31].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVC Kernel to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

# gamma = default
# Recall = 100%
# Precision = 85% approx.
# F1_Sscore = 91%
'''

# Fit Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

classifier = RandomForestClassifier(n_estimators= 100,
                                    criterion='entropy')
classifier.fit(X_train, y_train)

# n_trees = 100
# precision = 100%
# recall = 100%
# f1_score = 100%

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate Classification Metrics
recall = cm[1,1]/(cm[1,1]+cm[0,1])
precision = cm[1,1]/(cm[1,1]+cm[1,0])
f1_score = 2*recall*precision/(recall+precision)


'''
f1Score = []

for i in range (1, 51) :
    classifier = RandomForestClassifier(n_estimators = i,
                                    criterion='entropy')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    recall = cm[1,1]/(cm[1,1]+cm[0,1])
    precision = cm[1,1]/(cm[1,1]+cm[1,0])
    f1 = 2*recall*precision/(recall+precision)
    
    f1Score.append(f1)
    
plt.plot(range(1,51),f1Score)
plt.title("F1-Score trend for number of trees")
plt.xlabel("No. of trees")
plt.ylabel("F1-Score")
plt.show()
'''