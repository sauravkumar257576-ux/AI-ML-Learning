import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
data = {
    "Study_Hours":[1,2,3,4,5,6,7,8,9,10],
    "Result":[0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)
X = df["Study_Hours"].values.reshape(-1, 1)
y = df["Result"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel="linear")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Actual values is : {y_test}")
print(f"Predicted values is : {y_pred}")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy is :{accuracy}")

print(f"Pridicted value at 4.5 is : {model.predict([[4.5]])}")
print(f"Pridicted value at 7.5 is :{model.predict([[7.5]])}")