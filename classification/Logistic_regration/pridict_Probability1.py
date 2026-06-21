import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[3], [2],[1], [4], [5],[6], [7]])
y = np.array([0, 0, 0, 0, 1, 1, 1])

model = LogisticRegression()

model.fit(X, y)
print(model.predict_proba([[10]]))
print(model.predict([[10]]))