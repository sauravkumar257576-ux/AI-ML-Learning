import pandas as pd

from sklearn.decomposition import PCA

# Dataset
data = {
    "Study_Hours":[1,2,3,4,5],
    "Attendance":[50,60,70,80,90]
}

df = pd.DataFrame(data)

# Features
X = df[["Study_Hours","Attendance"]].values

# PCA
pca = PCA(n_components=1)

X_pca = pca.fit_transform(X)

# Results
print("Original Shape :", X.shape)

print("PCA Shape :", X_pca.shape)

print("Explained Variance Ratio :")
print(pca.explained_variance_ratio_)

print("PCA Data :")
print(X_pca)