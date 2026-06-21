import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
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

plt.figure(figsize=(8,5))



plot_tree(model,feature_names=["Study_Hours"],filled=True
)

plt.show()