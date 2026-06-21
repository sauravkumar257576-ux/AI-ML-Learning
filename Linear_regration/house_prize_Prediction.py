import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([600, 800, 1000, 1200, 1500]).reshape(-1,1)
y = np.array([30, 40, 55, 65, 80])

model = LinearRegression()
model.fit(X, y)
print(model.predict([[1300]]))
print("Slop(m):",model.coef_)
print("Intercept(c):",model.intercept_)

plt.scatter(X, y, color="green")
plt.plot(X, model.predict(X), color="blue")
plt.title("House price Prediction")
plt.xlabel("Area(sq ft)")
plt.ylabel("Price(lakh)")
plt.grid()
plt.show()