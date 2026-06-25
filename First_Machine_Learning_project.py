import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import joblib

data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10,11,12],
    "Attendance": [45,50,55,60,65,70,75,80,85,90,95,98],
    "Assignments": [1,1,2,2,3,3,4,4,5,5,6,6],
    "Result": [0,0,0,0,0,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Study_Hours", "Attendance", "Assignments"]].values
y= df["Result"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "knn_n_neighbors " : [1,3,5]
}
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=3
)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

y_pred = grid.predict(X_test)

print(
    accuracy_score(
        y_test, y_pred
    )
)

print(
    confusion_matrix(
        y_test, y_pred
    )
)

print(
    precision_score(
        y_test, y_pred
    )
)

print(
    recall_score(
        y_test, y_pred
    )
)

print(
    f1_score(
        y_test, y_pred
    )
)

joblib.dump(
    grid,
    "student_model.pkl"
)

saved_model = joblib.load(
    "student_model.pkl"
)

print(
    saved_model.predict(
        [[8,85,5]]
    )
)















































