import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10],
    "Result": [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df["Study_Hours"].values.reshape(-1, 1)
y = df["Result"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Predicted value : {y_pred}")
print(f"Actual value : {y_test}")

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy is : {accuracy}")

confusion = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix is : {confusion}")

print(f"Predict at point 4.5 is: {model.predict([[4.5]])}")
print(f"Predict at point 2.5 is: {model.predict([[2.5]])}")
print(f"Predict at point 7.5 is: {model.predict([[7.5]])}")