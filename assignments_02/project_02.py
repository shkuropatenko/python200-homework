import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# The dataset uses semicolon (;) as separator
# so we need to specify sep=";" when loading the file
base_dir = os.path.dirname(__file__)
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(base_dir, "student_performance_math.csv")

df = pd.read_csv(file_path, sep=";")

# Pre-preprocessing

print(df.shape)
print(df.head())
print(df.dtypes)

# Task 1: Load and Explore
plt.figure()
plt.hist(df["G3"], bins=21)

plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Count")

plt.savefig(os.path.join(output_dir, "g3_distribution.png"))
plt.close()

# Task 2: Preprocess the Data

print("Shape before:", df.shape)

df_clean = df[df["G3"] != 0].copy()

print("Shape after:", df_clean.shape)

# convert yes/no to 1/0

yes_no_cols = ["schoolsup", "internet", "higher", "activities"]

for col in yes_no_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})
df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

print(df_clean.head())
print(df_clean.dtypes)

corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])

print("Correlation absences vs G3 (original):", corr_original)
print("Correlation absences vs G3 (filtered):", corr_filtered)
print()
# Filtering out G3 = 0 changes the correlation because those students
# likely missed the final exam, and many of them also had high absences.
# Keeping them makes absences look less like a normal academic pattern
# and more like exam non-participation.

# Task 3

corr = df_clean.corr(numeric_only=True)["G3"].sort_values()
print("Correlation with G3:")
print(corr)
print()
# Failures appear to have one of the strongest negative relationships with G3.
# Study-related variables are generally more useful than lifestyle variables.
# Some weaker relationships may be surprising because real student performance
# depends on many factors that are not captured in this dataset.

# failures vs G3
# Failures show a negative relationship with G3.
# More failures generally lead to lower final grades.
plt.figure()
plt.scatter(df_clean["failures"], df_clean["G3"])

plt.title("Failures vs Final Grade")
plt.xlabel("Failures")
plt.ylabel("G3")

plt.savefig(os.path.join(base_dir, "outputs", "failures_vs_g3.png"))
plt.close()

# studytime vs G3
# Study time shows a positive relationship with G3.
# More study time tends to result in higher grades.
plt.figure()
plt.scatter(df_clean["studytime"], df_clean["G3"])

plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("G3")

plt.savefig(os.path.join(base_dir, "outputs", "studytime_vs_g3.png"))
plt.close()

# Task 4 — Baseline Model

X = df_clean[["failures"]].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

slope = model.coef_[0]

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print("Task 4 Baseline Model")
print("Slope:", slope)
print("RMSE:", rmse)
print("R2:", r2)
print()
# Failures negatively impact final grades.
# A higher number of failures generally leads to lower G3.
# RMSE shows the average prediction error.
# R2 shows how well failures alone explain student performance.

# Task 5

feature_cols = [
    "failures", "Medu", "Fedu", "studytime", "higher",
    "schoolsup", "internet", "sex", "freetime",
    "activities", "traveltime"
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test, y_test)

print("Task 5 Full Model")
print("Train R2:", r2_train)
print("Test R2:", r2_test)
print("RMSE:", rmse)
print()

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# The full model improves on the baseline by using several features instead of failures alone.
# If some coefficients have unexpected signs, that may be due to overlap between predictors
# and the small size of the dataset.
# In production, I would keep features with clearer signal such as failures, studytime,
# and education-related variables, and consider dropping weaker lifestyle features
# if they do not improve performance much.

# Task 6

plt.figure()
plt.scatter(y_pred, y_test)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")

plt.savefig(os.path.join(base_dir, "outputs", "predicted_vs_actual_full_model.png"))
plt.close()

# A point above the diagonal means the actual grade is higher than predicted.
# A point below the diagonal means the predicted grade is higher than actual.
# If points are spread widely, the model has larger prediction error.

# Summary:
# The filtered dataset has 357 rows, and the test set has about 71 rows.
# The RMSE is 2.86, meaning predictions are typically off by about 3 grade points.
# The R2 is 0.15, so the model explains about 15% of the variation in G3.
# The strongest negative coefficient is failures, showing that more failures lead to lower grades.
# Positive coefficients like studytime and higher suggest that more study and motivation
# are associated with better performance.

# Neglected Feature: G1

feature_cols_g1 = feature_cols + ["G1"]

X = df_clean[feature_cols_g1].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_train, y_train)

r2_test_g1 = model_g1.score(X_test, y_test)

print()
print("Task 6 with G1")
print("Test R2 with G1:", r2_test_g1)

# Comment:
# A high R2 with G1 does not mean G1 causes G3.
# G1 is just a very strong earlier indicator of later performance.
# This can help prediction, but it is less useful for very early intervention.