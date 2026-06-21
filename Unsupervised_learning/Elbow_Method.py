import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    "Study_Hours":[1,2,3,4,8,9,10,11]
}

df = pd.DataFrame(data)

X = df["Study_Hours"].values.reshape(-1, 1)

inertia_values = []

for k in range(1, 6):
    model = KMeans(n_clusters=k, random_state=42)

    model.fit(X)
    inertia_values.append(model.inertia_)

plt.plot(range(1,6), inertia_values, marker="o")
plt.title("Inertia values vs K")
plt.xlabel("K values")
plt.ylabel("Inertia values")
plt.show()