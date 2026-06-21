import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10,11,12],
    "Attendance": [45,50,55,60,65,70,75,80,85,90,95,98],
    "Result": [0,0,0,0,0,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Study_Hours", "Attendance"]].values
y = df["Result"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=20, random_state=42)


scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print(f"Accuracies are : {scores}")
print(f"Avarage Accuracy is : {scores.mean()}")

