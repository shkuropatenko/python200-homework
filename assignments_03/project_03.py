import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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