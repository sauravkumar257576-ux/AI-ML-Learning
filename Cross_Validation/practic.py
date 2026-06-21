import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
data = {
    "Study_Hours": [
        1,2,3,4,5,6,7,8,9,10,
        2,4,6,8,10,12,14,16,18,20,
        3,5,7,9,11,13,15,17,19,21,
        1,3,5,7,9,11,13,15,17,19,
        2,5,8,11,14,17,20,23,26,29
    ],

    "Attendance": [
        40,42,45,48,50,55,58,60,63,65,
        43,47,52,57,61,66,70,74,78,82,
        44,49,54,59,64,69,73,77,81,85,
        41,46,51,56,62,67,71,75,79,83,
        45,53,60,68,72,80,84,88,92,96
    ],

    "Assignments": [
        1,1,2,2,3,3,4,4,5,5,
        2,3,4,5,6,7,8,9,10,10,
        2,3,4,5,6,7,8,9,10,10,
        1,2,3,4,5,6,7,8,9,10,
        2,3,5,6,7,8,9,10,10,10
    ],

    "Pass": [
        0,0,0,0,0,0,0,1,1,1,
        0,0,0,1,1,1,1,1,1,1,
        0,0,1,1,1,1,1,1,1,1,
        0,0,0,1,1,1,1,1,1,1,
        0,0,1,1,1,1,1,1,1,1
    ]
}

df = pd.DataFrame(data)

X = df[["Study_Hours","Attendance","Assignments"]].values
y = df["Pass"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modal = LogisticRegression()

scores = cross_val_score(modal, X, y, cv=10)

print(scores)
print(scores.mean())