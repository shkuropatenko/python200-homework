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