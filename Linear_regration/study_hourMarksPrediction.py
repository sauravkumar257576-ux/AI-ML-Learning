import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1,2,3,4,5,6]).reshape(-1,1)
y = np.array([20, 35, 50, 65, 80, 95])

model = LinearRegression()

model.fit(X, y)

print(model.predict([[7]]))

plt.scatter(X, y, color="green")
plt.plot(X, model.predict(X))
plt.title("Study Hour prediction")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.grid()
plt.show()

print("Slop(m):",model.coef_)
print("Intersect(c):", model.intercept_)