import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [15, 25, 35, 45, 55, 65, 75, 85, 90, 98]
}

df = pd.DataFrame(data)
print(df.head(5))

X = df["Study_Hours"].values   # convert into array
X = X.reshape(-1, 1)        # convert into 2D array

y = df["Marks"].values     # convert pandas into array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Result : ",y_pred)  # Marks predicted by model in test data
print("Actual Result : ",y_test)   # Actual value
print(model.predict([[9.5]]))  # Marks predicted by model on new data


print("Slop of the best fit line : ",model.coef_)
print("Intecept of the best fit line : ",model.intercept_)

plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.title("Study_Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.grid()
plt.show()