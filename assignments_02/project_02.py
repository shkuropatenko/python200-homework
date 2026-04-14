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