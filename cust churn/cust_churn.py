# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
X = dataset.iloc[:, 1:20].values
y = dataset.iloc[:, 20:21].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoderY = LabelEncoder()
y[:, 0] = labelEncoderY.fit_transform(y[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()
y = y[:, 1:2]

# Label Encoding for Indices - 0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,18
l = [0,2,3,5,6,7,8,9,10,11,12,13,14,15,16]
count = 0
for i in l :
    labelEncoderX = LabelEncoder()
    X[:, i] = labelEncoderX.fit_transform(X[:, i])
    
height = np.shape(X)[0]
X = np.append(X, np.reshape(X[:, 4]**2, (height, 1)), 1)
X = np.append(X, np.reshape(X[:, 4]**3, (height, 1)), 1)
X = np.append(X, np.reshape(X[:, 17]**2, (height, 1)), 1)
X = np.append(X, np.reshape(X[:, 17]**3, (height, 1)), 1)
X = np.append(X, np.reshape(X[:, 18]**2, (height, 1)), 1)
X = np.append(X, np.reshape(X[:, 18]**3, (height, 1)), 1)

#One_Hot Encoding and  Removing Dummy Variable Trap
for i,j in enumerate(l) :
    onc = OneHotEncoder(categorical_features = [count + j - (2*i)])
    onc.fit(X)
    count = count+ int(onc.n_values_)
    X = onc.fit_transform(X).toarray()
    X= X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#underfitting
"""

# Fitting SVC Kernel to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

"""
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 10,
                                    criterion='entropy')
classifier.fit(X_train, y_train)

F1_train = 0.998
F1_test = 0.513

#overfitting
"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_train = classifier.predict(X_train)
#underfitting

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
precision = cm[1,1]/(cm[0,1] + cm[1,1])
recall = cm[1,1]/(cm[1,0] + cm[1,1])

cm_train = confusion_matrix(y_train, y_pred_train)
precision_train = cm_train[1,1]/(cm_train[0,1] + cm_train[1,1])
recall_train = cm_train[1,1]/(cm_train[1,0] + cm_train[1,1])

f1_score = 2*precision*recall/(precision + recall)
f1_score_train = 2*precision_train*recall_train/(precision_train + recall_train)
