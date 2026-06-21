import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

data = {
    "Study_Hours": [2,4,1,6,8,5,7,3,9,10,11,12,4,6,8,2,5,7,9,11,13,14,15,16,17,18,19,20,10,12],
    "Attendance": [50,60,40,70,80,65,75,55,85,90,92,95,58,68,78,48,64,74,84,91,96,97,98,99,95,96,97,98,88,90],
    "Assignments": [1,2,1,3,4,3,4,2,5,6,7,8,2,3,5,1,3,4,6,7,8,8,9,9,10,10,10,10,6,7],
    "Pass": [0,0,0,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Study_Hours", "Attendance", "Assignments"]].values
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standerd Scaler
scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train)

X_test_scaled = scalar.transform(X_test)

# Logistic Regression 
lr_model_scaled = LogisticRegression()

lr_model_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = lr_model_scaled.predict(X_test_scaled)

scores = cross_val_score(lr_model_scaled, X, y, cv=5)

print(f"LR Accuracy is : {accuracy_score(y_test, y_pred_scaled)}")
print(f"LR Precision is : {precision_score(y_test, y_pred_scaled)}")
print(f"LR Recall is : {recall_score(y_test, y_pred_scaled)}")
print(f"LR f1 score is : {f1_score(y_test, y_pred_scaled)}")
print(f"accuracies are : {scores}")
print(f"Avarage accuracie is : {scores.mean()}")


# K-Nearst-Neighbour
knn_model_scaled = KNeighborsClassifier(n_neighbors=3)
knn_model_scaled.fit(X_train_scaled, y_train)
knn_y_pred = knn_model_scaled.predict(X_test_scaled)

scores = cross_val_score(knn_model_scaled, X, y, cv=5)
param_grid = {
    "n_neighbors" : [1,3,5,7]
}

grid = GridSearchCV(
    estimator=knn_model_scaled,
    param_grid=param_grid,
    cv= 5
)

grid.fit(X, y)

print(f"knn Accuracy is : {accuracy_score(y_test, knn_y_pred)}")
print(f"knn Precision is : {precision_score(y_test, knn_y_pred)}")
print(f"knn Recall is : {recall_score(y_test, knn_y_pred)}")
print(f"knn f1 score is : {f1_score(y_test, knn_y_pred)}")
print(f"accuracies are : {scores}")
print(f"Avarage accuracie is : {scores.mean()}")
print(f"Best parameter is : {grid.best_params_}")
print(f"Best score is : {grid.best_score_}")

# Decision Tree
dt_model_scaled = DecisionTreeClassifier(max_depth=3)
dt_model_scaled.fit(X_train_scaled, y_train)
dt_y_predict = dt_model_scaled.predict(X_test_scaled)
scores = cross_val_score(dt_model_scaled, X, y, cv=5)

print(f"dt Accuracy is : {accuracy_score(y_test, dt_y_predict)}")
print(f"dt Precision is : {precision_score(y_test, dt_y_predict)}")
print(f"dt Recall is : {recall_score(y_test, dt_y_predict)}")
print(f"dt f1 score is : {f1_score(y_test, dt_y_predict)}")
print(f"accuracies are : {scores}")
print(f"Avarage accuracie is : {scores.mean()}")

# Random Forest 
rf_model_scaled = RandomForestClassifier(n_estimators=10)
rf_model_scaled.fit(X_train_scaled, y_train)
rf_y_pred = rf_model_scaled.predict(X_test_scaled)
scores = cross_val_score(rf_model_scaled, X, y, cv=5)

print(f"rf Accuracy is : {accuracy_score(y_test, rf_y_pred)}")
print(f"rf Precision is : {precision_score(y_test, rf_y_pred)}")
print(f"rf Recall is : {recall_score(y_test, rf_y_pred)}")
print(f"rf f1 score is : {f1_score(y_test, rf_y_pred)}")
print(f"accuracies are : {scores}")
print(f"Avarage accuracie is : {scores.mean()}")

# SVC

svc_model_scaled = SVC()
svc_model_scaled.fit(X_train_scaled, y_train)
svc_y_pred = svc_model_scaled.predict(X_test_scaled)
scores = cross_val_score(svc_model_scaled, X, y, cv=5)


print(f"svc Accuracy is : {accuracy_score(y_test, svc_y_pred)}")
print(f"svc Precision is : {precision_score(y_test, svc_y_pred)}")
print(f"svc Recall is : {recall_score(y_test, svc_y_pred)}")
print(f"svc f1 score is : {f1_score(y_test, svc_y_pred)}")
print(f"accuracies are : {scores}")
print(f"Avarage accuracie is : {scores.mean()}")

# Gaussian Naive Bays
nb_model_scaled = GaussianNB()
nb_model_scaled.fit(X_train_scaled, y_train)
nb_y_pred = nb_model_scaled.predict(X_test_scaled)
scores = cross_val_score(nb_model_scaled, X, y, cv=5)

print(f"nb Accuracy is : {accuracy_score(y_test, nb_y_pred)}")
print(f"nb Precision is : {precision_score(y_test, nb_y_pred)}")
print(f"nb Recall is : {recall_score(y_test, nb_y_pred)}")
print(f"nb f1 score is : {f1_score(y_test, nb_y_pred)}")
print(f"accuracies are : {scores}")
print(f"Avarage accuracie is : {scores.mean()}")











































