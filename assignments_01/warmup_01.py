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

# Pandas Q6
print("\n" + "=" * 40)
print("Replace the value 'Austin' in the 'city' column with 'Houston'. Print the 'name' and 'city' columns to confirm the change.")
df["city"] = df["city"].replace({"Austin": "Houston"})
print(df[["name", "city"]])

# Pandas Q7
print("\n" + "=" * 40)
print("Sort the DataFrame by 'grade' in descending order and print the top 3 rows.")
print(df.sort_values(by = "grade", ascending=True).head(3))

# --- NumPy ---
import numpy as np

# NumPy Question 1
print("\n" + "=" * 40)
print("Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.")
arr = np.array([10, 20, 30, 40, 50])
print(f"shape {arr.shape}")
print(f"dtype {arr.dtype}")
print(f"ndim {arr.ndim}")

# NumPy Question 2
print("\n" + "=" * 40)
print("Create the following 2D array and print its shape and size (total number of elements).")
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print(f"shape {arr.shape}")
print(f"size {arr.size}")

# NumPy Question 3
print("\n" + "=" * 40)
print("Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]].")
print(arr[:2, :2])

# NumPy Question 4
print("\n" + "=" * 40)
print("Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both.")
arr_0 = np.zeros((3, 4))
arr_1 = np.ones((2, 5))

print(f"3x4 \n {arr_0}")
print(f"2x5 \n {arr_1}")

# NumPy Question 5
print("\n" + "=" * 40)
print("Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like. Then, print the array, its shape, mean, sum, and standard deviation.")
arr = np.arange(0, 50, 5)
print(f"shape {arr.shape}")
print(f"mean {arr.mean()}")
print(f"sum {arr.sum()}")
print(f"std {arr.std()}")

# NumPy Question 6
print("\n" + "=" * 40)
print("Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal()). Print the mean and standard deviation of the result.")
arr = np.random.normal(0, 1, 200)
print(arr.mean())
print(arr.std())
