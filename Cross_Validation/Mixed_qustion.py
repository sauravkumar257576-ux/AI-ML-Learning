import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10,11,12],
    "Attendance": [45,50,55,60,65,70,75,80,85,90,95,98],
    "Result": [0,0,0,0,0,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Study_Hours", "Attendance"]].values
y = df["Result"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

logical_regression_accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy is : {logical_regression_accuracy}")
print(f"Confusion Matrix is : {confusion}")
print(f"Precision is : {precision}")
print(f"Recall is : {recall}")
print(f"F1 score is : {f1}")


# KNN
best_Accuracy = 0
best_K = None
accuracies = []
k_values = []

for k in [1, 3, 5, 7]:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracies.append(accuracy)
    
    k_values.append(k)

    print(f"Accuracy at K={k} is : {accuracy}")

    if accuracy > best_Accuracy:
        best_Accuracy = accuracy
        best_k = k

print(f"Best Accuracy is : {best_Accuracy}")
print(f"Best K is : {best_K}")


# Decision Tree

model = DecisionTreeClassifier()
# Train
model.fit(X_train, y_train)
# predict
y_pred = model.predict(X_test)

decision_Tree_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy is : {decision_Tree_accuracy}")
print(f"Prediction on [5, 65] is : {model.predict([[5,65]])}")
print(f"Prediction on [10, 90] is : {model.predict([[10,90]])}")


# Random Forest
model = RandomForestClassifier(n_estimators=20, random_state=42)
# modal Train
model.fit(X_train, y_train)
# modal prdict 
y_pred = model.predict(X_test)
# Accuracy 
random_forest_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy is : {random_forest_accuracy}")


# SVM
model = SVC(kernel="linear")
model.fit(X_train, y_train)
# modal prdict 
y_pred = model.predict(X_test)
# Accuracy 
svm_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy is : {svm_accuracy}")


# Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)
# modal prdict 
y_pred = model.predict(X_test)
# Accuracy 
naive_bayes_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy is : {naive_bayes_accuracy}")
print(f"Prediction on [5, 65] is : {model.predict([[5,65]])}")
print(f"Prediction on [10, 90] is : {model.predict([[10,90]])}")


# Cross Validation
model = RandomForestClassifier(n_estimators=20, random_state=42)
scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print(f"Accuracies are : {scores}")
print(f"Avarage Accuracy is : {scores.mean()}")

