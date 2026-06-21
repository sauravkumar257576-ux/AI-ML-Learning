# Precision
# Model ne jitno ko Pass bola,
# unme se kitne sach me Pass the?
# Formula:  TP / (TP + FP)



# Recall
# Jo sach me Pass the,
# unme se kitno ko model ne pakda?
# Formula: TP / (TP + FN)

# TP = Sahi Pass
# TN = Sahi Fail
# FP = Galat Pass
# FN = Galat Fail



from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

y_test = [1, 0, 1, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Precision =", precision)
print("Recall =", recall)