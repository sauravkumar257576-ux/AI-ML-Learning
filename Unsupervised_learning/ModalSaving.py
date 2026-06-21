import pandas as pd
import joblib

from sklearn.neighbors import KNeighborsClassifier

# Dataset
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10],
    "Result": [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[["Study_Hours"]].values
y = df["Result"].values

# Model
model = KNeighborsClassifier(n_neighbors=3)

# Train
model.fit(X, y)

# Save Model
joblib.dump(model, "student_model.pkl")

print("Model Saved Successfully!")

# Load Model
saved_model = joblib.load("student_model.pkl")

print("Model Loaded Successfully!")

# Predictions
print("Prediction for [5] :", saved_model.predict([[5]]))
print("Prediction for [8] :", saved_model.predict([[8]]))