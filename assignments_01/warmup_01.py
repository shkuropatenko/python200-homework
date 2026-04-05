# --- Pandas ---
import pandas as pd

# Pandas Q1

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print("=" * 40)
print("First Three Rows")
print("=" * 40)
print(df.head(3))

print("\n" + "=" * 40)
print("Shape")
print("=" * 40)
print(df.shape)

print("\n" + "=" * 40)
print("Data Types")
print("=" * 40)
print(df.dtypes)

# Pandas Q2
print("\n" + "=" * 40)
print(df[(df["grade"] > 80) & (df["passed"] == True)])