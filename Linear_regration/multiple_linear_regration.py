import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Dataset
data = {
    "Study_Hours": [5,6,7,8,9,4,3,10,6,7],
    "Sleep_Hours": [7,8,8,7,8,6,7,8,7,6],
    "Practice":    [3,4,5,6,7,2,1,8,4,5],
    "Marks":       [60,75,85,95,105,50,40,115,72,82]
}

df = pd.DataFrame(data)

# Features
X = df[["Study_Hours", "Sleep_Hours", "Practice"]].values

# Target
y = df["Marks"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE =", mean_absolute_error(y_test, y_pred))
print("MSE =", mean_squared_error(y_test, y_pred))
print("R2 =", r2_score(y_test, y_pred))

# New Student
result = model.predict([[6, 8, 5]])

print("Predicted Marks =", result)