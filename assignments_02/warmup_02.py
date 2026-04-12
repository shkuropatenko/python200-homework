# Imports
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# outputs folder exists
os.makedirs("outputs", exist_ok=True)
base_dir = os.path.dirname(__file__)
output_dir = os.path.join(base_dir, "outputs")

# =========================
# --- scikit-learn API ---
# =========================

# Q1
years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)

pred_4 = model.predict([[4]])[0]
pred_8 = model.predict([[8]])[0]

print("Q1")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salary for 4 years:", pred_4)
print("Predicted salary for 8 years:", pred_8)
print()

# Q2
x = np.array([10, 20, 30, 40, 50])

print("Q2")
print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)
print("Reshaped shape:", x_2d.shape)

# scikit-learn expects X to be 2D because each row is one sample
# and each column is one feature.
print()

# Q3
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)

labels = kmeans.predict(X_clusters)

print("Q3")
print("Cluster centers:")
print(kmeans.cluster_centers_)
print("Points in each cluster:")
print(np.bincount(labels))
print()

plt.figure()
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="x",
    s=200
)
plt.title("KMeans Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig(os.path.join(output_dir, "kmeans_clusters.png"))
plt.close()


# =========================
# --- Linear Regression ---
# =========================

# Generate dataset once and reuse it for all questions
np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Q1
plt.figure()
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")
plt.savefig(os.path.join(output_dir, "cost_vs_age.png"))
plt.close()

# Comment:
# Yes, there appear to be two visible groups.
# This suggests that smoker status has a strong effect on medical cost.

# Q2
X = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Linear Regression Q2")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print()

# Q3
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Linear Regression Q3")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print("RMSE:", rmse)

r2 = model.score(X_test, y_test)
print("R2:", r2)
print()

# Comment:
# The slope represents how much medical cost increases with age.
# For each additional year of age, the cost increases by the slope amount.

# Q4

X_full = np.column_stack([age, smoker])
y = cost

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

r2_full = model_full.score(X_test_full, y_test_full)

print("Linear Regression Q4")
print("R2 with age only:", r2)
print("R2 with age + smoker:", r2_full)

print("age coefficient:", model_full.coef_[0])
print("smoker coefficient:", model_full.coef_[1])
print()

# Comment:
# Adding smoker improves the model because smoking strongly increases medical cost.
# The smoker coefficient represents the additional cost for smokers.

# Q5
y_pred_full = model_full.predict(X_test_full)

plt.figure()
plt.scatter(y_pred_full, y_test_full)

plt.plot(
    [y_test_full.min(), y_test_full.max()],
    [y_test_full.min(), y_test_full.max()]
)

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig(os.path.join(output_dir, "predicted_vs_actual.png"))
plt.close()

# Comment:
# A point above the diagonal means the actual value is higher than the prediction.
# A point below the diagonal means the actual value is lower than the prediction.