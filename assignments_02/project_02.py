import pandas as pd
import matplotlib.pyplot as plt
import os

# The dataset uses semicolon (;) as separator
# so we need to specify sep=";" when loading the file
base_dir = os.path.dirname(__file__)
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

plt.savefig(os.path.join(base_dir, "outputs", "g3_distribution.png"))
plt.close()

# Task 2: Preprocess the Data

print("Shape before:", df.shape)

df_clean = df[df["G3"] != 0]

print("Shape after:", df_clean.shape)