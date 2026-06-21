
# F1 Score = 2 × (Precision × Recall)
#            ------------------------
#             Precision + Recall



# from sklearn.metrics import f1_score
# f1 = f1_score(y_test, y_pred)
# print("F1 Score = ", f1)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))