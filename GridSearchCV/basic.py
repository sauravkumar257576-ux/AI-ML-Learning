import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Dataset
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10],
    "Result": [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[["Study_Hours"]].values
y = df["Result"].values

# Model
model = KNeighborsClassifier()

# Hyperparameters to test
param_grid = {
    "n_neighbors": [1, 3, 5, 7]
}

# Grid Search with 5-Fold Cross Validation
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5
)

# Training
grid.fit(X, y)

# Results
print("Best Parameters :", grid.best_params_)
print("Best Score :", grid.best_score_)