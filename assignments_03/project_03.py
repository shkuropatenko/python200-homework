import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

df = pd.read_csv(url, header=None)

print("Shape:", df.shape)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("X shape:", X.shape)
print("y distribution:")
print(y.value_counts())

features = [0, 52, 55]  # примерно соответствуют word_freq_free, char_freq_!, capital_run_length_total

for f in features:
    plt.figure()
    df.boxplot(column=f, by=y)
    plt.title(f"Feature {f} by spam (1) vs ham (0)")
    plt.suptitle("")
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_feature_{f}.png"))
    plt.close()