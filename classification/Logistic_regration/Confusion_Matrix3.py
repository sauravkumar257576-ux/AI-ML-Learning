import numpy as np
import pandas as pd 
import matplotlib as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = {
    "Study_Hours" : [1,2,3,4,6,7,8,10],
    "Result" : [0,0,0,0,1,1,1,1]



}

df = pd.DataFrame(data)

X = df["Study_Hours"].values.reshape(-1, 1)
y = df["Result"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Pridicted Result: ", y_pred)
print("Actual Result : ", y_test)

accuracy = accuracy_score(y_pred, y_test)
print("Accuracy is : ", accuracy)
confusion = confusion_matrix(y_pred, y_test)
print("Confusion matrix is : ",confusion)