import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

data = {
    "Study_Hours" : [1,2,3,4,5,6,7,8,9,10],
    "Result" : [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df["Study_Hours"].values.reshape(-1, 1)

y = df["Result"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Values : ",y_pred)
print("Actual Values :", y_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy : ",accuracy)

confusion = confusion_matrix(y_test, y_pred)

print("Confusion matrix : ", confusion)

print(model.predict([[4.5]]))

plt.scatter(X, y, color="green")
plt.plot(X, model.predict(X))
plt.title("KNN Mini Project")
plt.xlabel("Study Hours")
plt.ylabel("Result")
plt.grid()
plt.show()


for k in [1, 3, 5, 7]:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy at k={k} :", accuracy)

