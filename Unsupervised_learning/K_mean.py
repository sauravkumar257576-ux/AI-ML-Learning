import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans

data = {
    "Study_Hours":[1,2,3,4,8,9,10,11]
}

df = pd.DataFrame(data)

X = df["Study_Hours"].values.reshape(-1, 1)

modal = KMeans(n_clusters=3, random_state=42)

modal.fit(X)

print(modal.labels_)
print(modal.cluster_centers_)