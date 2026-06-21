import numpy as np
from sklearn.tree import DecisionTreeClassifier

X = np.array([[1],[2],[3],[5],[6],[7]])
y = np.array([0,0,0,1,1,1])

model = DecisionTreeClassifier()

model.fit(X, y)

print(model.predict([[4]]))