import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Task 1: Load and Explore
# ----------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
df = pd.read_csv(url, header=None)

print("Shape:", df.shape)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("X shape:", X.shape)
print("y distribution:")
print(y.value_counts())

print("Class balance (%):")
print(y.value_counts(normalize=True) * 100)

# Accuracy can be misleading when classes are not perfectly balanced.

# Boxplots for the requested features
features = {
  "word_freq_free": 15,
  "char_freq_!": 51,
  "capital_run_length_total": 56
}

df["spam_label"] = y

for feature_name, col_idx in features.items():
  plt.figure()
  df.boxplot(column=col_idx, by="spam_label")
  plt.title(f"{feature_name} by spam (1) vs ham (0)")
  plt.suptitle("")
  plt.savefig(os.path.join(OUTPUT_DIR, f"{feature_name}_boxplot.png"))
  plt.close()

# Many of these features are heavily skewed toward zero because many emails
# do not contain these words or symbols at all.

# ----------------------------
# Task 2: Prepare Your Data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scaling is important because the features have very different ranges.
# We fit the scaler on the training data only to avoid data leakage.

pca = PCA()
pca.fit(X_train_scaled)

explained_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(explained_variance)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA cumulative explained variance")
plt.savefig(os.path.join(OUTPUT_DIR, "pca_variance_project.png"))
plt.close()

n = np.argmax(explained_variance >= 0.9) + 1
print("Components for 90% variance:", n)

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

# ----------------------------
# Task 3: Classifier Comparison
# ----------------------------

# KNN on unscaled data
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_knn_unscaled = knn_unscaled.predict(X_test)

print("\nKNN (unscaled) accuracy:")
print(accuracy_score(y_test, y_pred_knn_unscaled))
print(classification_report(y_test, y_pred_knn_unscaled))

# KNN on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

print("\nKNN (scaled) accuracy:")
print(accuracy_score(y_test, y_pred_knn_scaled))
print(classification_report(y_test, y_pred_knn_scaled))

# KNN on PCA data
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_knn_pca = knn_pca.predict(X_test_pca)

print("\nKNN (scaled + PCA) accuracy:")
print(accuracy_score(y_test, y_pred_knn_pca))
print(classification_report(y_test, y_pred_knn_pca))

# Decision Tree depth comparison
print("\nDecision Tree depth comparison")
for depth in [3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))

    print(f"max_depth={depth}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

# Choose one depth for final tree model
chosen_depth = 5

tree_final = DecisionTreeClassifier(max_depth=chosen_depth, random_state=42)
tree_final.fit(X_train, y_train)
y_pred_tree = tree_final.predict(X_test)

print(f"\nDecision Tree (max_depth={chosen_depth}) accuracy:")
print(accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# As depth increases, training accuracy usually increases,
# but test accuracy may stop improving, which suggests overfitting.

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest accuracy:")
print(accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Logistic Regression on scaled data
log_scaled = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
log_scaled.fit(X_train_scaled, y_train)
y_pred_log_scaled = log_scaled.predict(X_test_scaled)

print("\nLogistic Regression (scaled) accuracy:")
print(accuracy_score(y_test, y_pred_log_scaled))
print(classification_report(y_test, y_pred_log_scaled))

# Logistic Regression on PCA data
log_pca = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
log_pca.fit(X_train_pca, y_train)
y_pred_log_pca = log_pca.predict(X_test_pca)

print("\nLogistic Regression (scaled + PCA) accuracy:")
print(accuracy_score(y_test, y_pred_log_pca))
print(classification_report(y_test, y_pred_log_pca))

# For spam filtering, accuracy is useful, but false positives and false negatives matter too.

# ----------------------------
# Best Model Confusion Matrix
# ----------------------------
model_accuracies = {
    "KNN (unscaled)": accuracy_score(y_test, y_pred_knn_unscaled),
    "KNN (scaled)": accuracy_score(y_test, y_pred_knn_scaled),
    "KNN (scaled + PCA)": accuracy_score(y_test, y_pred_knn_pca),
    f"Decision Tree (max_depth={chosen_depth})": accuracy_score(y_test, y_pred_tree),
    "Random Forest": accuracy_score(y_test, y_pred_rf),
    "Logistic Regression (scaled)": accuracy_score(y_test, y_pred_log_scaled),
    "Logistic Regression (scaled + PCA)": accuracy_score(y_test, y_pred_log_pca),
}

print("\nModel accuracies:")
for model_name, acc in model_accuracies.items():
    print(f"{model_name}: {acc:.4f}")

best_model_name = max(model_accuracies, key=model_accuracies.get)
print("\nBest model:", best_model_name)

if best_model_name == "KNN (unscaled)":
    best_preds = y_pred_knn_unscaled
elif best_model_name == "KNN (scaled)":
    best_preds = y_pred_knn_scaled
elif best_model_name == "KNN (scaled + PCA)":
    best_preds = y_pred_knn_pca
elif best_model_name == f"Decision Tree (max_depth={chosen_depth})":
    best_preds = y_pred_tree
elif best_model_name == "Random Forest":
    best_preds = y_pred_rf
elif best_model_name == "Logistic Regression (scaled)":
    best_preds = y_pred_log_scaled
else:
    best_preds = y_pred_log_pca

cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
disp.plot()
plt.title(f"Best Model Confusion Matrix: {best_model_name}")
plt.savefig(os.path.join(OUTPUT_DIR, "best_model_confusion_matrix.png"))
plt.close()

tn, fp, fn, tp = cm.ravel()
print("\nConfusion matrix values:")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

if fp > fn:
    print("The best model makes more false positives than false negatives.")
elif fn > fp:
    print("The best model makes more false negatives than false positives.")
else:
    print("The best model makes false positives and false negatives equally often.")

# ----------------------------
# Feature Importances
# ----------------------------
tree_importances = pd.Series(tree_final.feature_importances_, index=X.columns).sort_values(ascending=False)
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nTop 10 Decision Tree feature importances:")
print(tree_importances.head(10))

print("\nTop 10 Random Forest feature importances:")
print(rf_importances.head(10))

plt.figure(figsize=(10, 6))
rf_importances.head(10).sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Random Forest Feature Importances")
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importances.png"))
plt.close()

# The Decision Tree and Random Forest may overlap on some important features,
# but Random Forest importances are usually more stable.