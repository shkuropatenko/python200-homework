import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---
# Q1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train_scaled means:", np.mean(X_train_scaled, axis=0))

# We fit the scaler on X_train only to avoid leaking information from the test set.

# --- KNN ---
# Q1
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_knn_unscaled = knn_unscaled.predict(X_test)

print("\nKNN unscaled accuracy:")
print(accuracy_score(y_test, y_pred_knn_unscaled))
print("\nKNN unscaled classification report:")
print(classification_report(y_test, y_pred_knn_unscaled))

# Q2
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

print("\nKNN scaled accuracy:")
print(accuracy_score(y_test, y_pred_knn_scaled))

# Scaling may make little difference here because the Iris features are already on somewhat similar scales.

# Q3
cv_scores_knn = cross_val_score(
    KNeighborsClassifier(n_neighbors=5),
    X_train,
    y_train,
    cv=5
)

print("\nKNN CV scores:")
print(cv_scores_knn)
print("KNN CV mean:", cv_scores_knn.mean())
print("KNN CV std:", cv_scores_knn.std())

# This is more trustworthy than a single train/test split because it averages performance across multiple folds.

# Q4
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
best_k = None
best_score = -1

print("\nKNN mean CV scores by k:")
for k in k_values:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5)
    mean_score = scores.mean()
    print(f"k={k}: {mean_score:.4f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"Best k based on mean CV score: {best_k} ({best_score:.4f})")

# I would choose the k with the highest mean CV score because it performed best across the folds.

# --- Classifier Evaluation ---
# Q1
cm = confusion_matrix(y_test, y_pred_knn_unscaled)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()

# The model mostly confuses versicolor and virginica, if there is any confusion at all.

# --- Decision Trees ---
# Q1
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\nDecision Tree accuracy:")
print(accuracy_score(y_test, y_pred_dt))
print("\nDecision Tree classification report:")
print(classification_report(y_test, y_pred_dt))

# The Decision Tree accuracy is similar to KNN, though the exact result may differ slightly.
# Scaled vs. unscaled data should not meaningfully affect a Decision Tree because it does not rely on distance calculations.

# --- Logistic Regression and Regularization ---
# Q1
for c_value in [0.01, 1.0, 100]:
    log_model = LogisticRegression(C=c_value, max_iter=1000, solver="liblinear")
    log_model.fit(X_train_scaled, y_train)
    coef_size = np.abs(log_model.coef_).sum()
    print(f"C={c_value}, total coefficient magnitude={coef_size:.4f}")

# As C increases, the total coefficient magnitude usually increases.
# This shows that weaker regularization allows the model to use larger coefficients.