# Real life Industary use
# Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", LogisticRegression())
# ])


# Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", SVC())
# ])


# Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", KNeighborsClassifier())
# ])


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "Study_Hours":[1,2,3,4,5,6,7,8,9,10],
    "Attendance":[50,55,60,65,70,75,80,85,90,95],
    "Result":[0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[["Study_Hours", "Attendance"]].values
y = df["Result"].values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=3))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Actual Values :", y_test)
print("Predicted Values :", y_pred)
print("Accuracy :", accuracy)

# New Prediction
print("Prediction for [5,70] :",
      pipeline.predict([[5,70]]))
