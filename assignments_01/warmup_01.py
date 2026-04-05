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
print("Filter the rows to show only students who passed and have a grade above 80")
print(df[(df["grade"] > 80) & (df["passed"] == True)])

# Pandas Q3
print("\n" + "=" * 40)
print("Add a new column called 'grade_curved' that Adds 5 points to each student's grade")
df["grade_curved"] = df["grade"] + 5
print(df)

# Pandas Q4
print("\n" + "=" * 40)
print("name in uppercase")
df["name_upper"] = df["name"].str.upper().str.strip()
print(df[["name", "name_upper"]].head())

# Pandas Q5
print("\n" + "=" * 40)
print("compute the mean grade for each city")
print(df.groupby("city")["grade"].mean())