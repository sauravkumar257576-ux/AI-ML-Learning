import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Dataset
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10,11,12],
    "Attendance": [45,50,55,60,65,70,75,80,85,90,95,98],
    "Assignments": [1,1,2,2,3,3,4,4,5,5,6,6],
    "Result": [0,0,0,0,0,1,1,1,1,1,1,1]
}

# DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[["Study_Hours", "Attendance", "Assignments"]].values
y = df["Result"].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Hyperparameter Tuning
param_grid = {
    "knn__n_neighbors": [1, 3, 5]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=4
)

# Training
grid.fit(X_train, y_train)

# Best Model
print("Best Parameters :", grid.best_params_)
print("Best CV Score :", grid.best_score_)

# Prediction
y_pred = grid.predict(X_test)

# Evaluation
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Confusion Matrix :")
print(confusion_matrix(y_test, y_pred))

print("Precision :", precision_score(y_test, y_pred))
print("Recall :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

# Save Model
joblib.dump(grid, "student_model.pkl")

# Load Model
saved_model = joblib.load("student_model.pkl")

# New Predictions
print("Prediction for [8,85,5] :",
      saved_model.predict([[8,85,5]]))

print("Prediction for [3,55,2] :",
      saved_model.predict([[3,55,2]]))