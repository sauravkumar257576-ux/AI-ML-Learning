import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [15, 25, 35, 45, 55, 65, 75, 85, 90, 98]
}

df = pd.DataFrame(data)
X = df["Study_Hours"].values.reshape(-1, 1)
y = df["Marks"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test) 

print("Pridicted Result : ", y_pred)
print("Actual Result : ", y_test)

mae = mean_absolute_error(y_pred, y_test)
mse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_pred, y_test)

print("\nMean absolute error is : ", mae)
print("Mean square error is : ", mse)
print("r2 Score is : ", r2)