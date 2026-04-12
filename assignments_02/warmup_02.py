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


# =========================
# --- Linear Regression ---
# =========================

# Generate dataset once and reuse it for all questions
np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)