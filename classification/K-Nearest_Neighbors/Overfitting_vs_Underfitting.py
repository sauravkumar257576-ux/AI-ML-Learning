# KNN me:

# K bahut chhota
# ↓
# Overfitting
# K bahut bada
# ↓
# Underfitting


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10,11,12],
    "Result":      [0,0,0,0,1,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df["Study_Hours"].values.reshape(-1, 1)
y = df["Result"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


best_accuracy = 0
best_k = None
k_values =[]
accuracies = []
for k in [1,3,5, 7, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracies.append(accuracy)
    k_values.append(k)
    print(f"Accuracy is :{accuracy}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k


print(f"Best k is : {best_k}")
print(f"Best Accuracy is :{best_accuracy}")

plt.plot(k_values, accuracies, marker="o")
plt.title("k vs Accuracy")
plt.xlabel("K values")
plt.ylabel("Accuracy")
plt.grid()
plt.show()