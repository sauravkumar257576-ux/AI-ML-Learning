import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

data = {
    "Experience": [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20],
    "Age": [22,23,24,25,26,27,28,29,30,31,35,38,42,46,50],
    "Current_Salary": [
        20000,25000,30000,35000,40000,
        50000,60000,70000,80000,90000,
        110000,130000,150000,170000,200000
    ],
    "High_Salary": [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Experience", "Age", "Current_Salary"]].values
y = df["High_Salary"].values

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracies = accuracy_score(y_test, y_pred)
print(f"Accuracy is :{accuracies}")


scaler = StandardScaler()

X_test_scaled = scaler.fit_transform(X_test)
X_train_scaled = scaler.transform(X_train)

model_scaled = KNeighborsClassifier()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model.predict(X_test_scaled)
accuracies = accuracy_score(y_test, y_pred)
print(accuracies)