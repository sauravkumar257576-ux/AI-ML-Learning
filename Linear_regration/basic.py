# import numpy as np
# from sklearn.linear_model import LinearRegression

#data
# X = np.array([1,2,3,4,5]).reshape(-1,1)
# y = np.array([10,20,30, 40, 50])

# model
# model = LinearRegression()

# training
# model.fit(X,y)


#prediction
# pred = model.predict([[98]])
# print(pred)

#learned value
# print(model.coef_)
# print(model.intercept_)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([10, 20, 30, 40, 50])

model = LinearRegression()
model.fit(X,y)

print(model.predict([[6]]))

plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')

plt.title("Linear Regrassion")
plt.xlabel("Hours")
plt.ylabel("Marks")

plt.show()

print("Slope (m):", model.coef_)
print("Intercept (c):", model.intercept_)
