import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10],
    "Attendance": [50,55,60,65,70,75,80,85,90,95],
    "Result": [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[["Study_Hours", "Attendance"]].values
y = df["Result"].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# KNN WITHOUT SCALING
# -------------------------
modal = KNeighborsClassifier(n_neighbors=3)

modal.fit(X_train, y_train)

y_pred = modal.predict(X_test)

accuracy_without_scaling = accuracy_score(y_test, y_pred)

print("Accuracy without scaling :", accuracy_without_scaling)

# -------------------------
# KNN WITH SCALING
# -------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

modal_scaled = KNeighborsClassifier(n_neighbors=3)

modal_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = modal_scaled.predict(X_test_scaled)

accuracy_with_scaling = accuracy_score(y_test, y_pred_scaled)

print("Accuracy with scaling :", accuracy_with_scaling)