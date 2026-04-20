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


# --- Matplotlib ---
import matplotlib.pyplot as plt

# Matplotlib Q1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Matplotlib Q2
subjects = ["Math", "Science", "English", "History"]
scores  = [88, 92, 75, 83]

plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("subjects")
plt.ylabel("scores")

plt.show()

# Matplotlib Q3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color="blue", label="Dataset 1")
plt.scatter(x2, y2, color="red", label="Dataset 2")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()

# Matplotlib Q4

# Q1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Q2
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

fig, ax = plt.subplots(1, 2)

# left graphic
ax[0].plot(x, y)
ax[0].set_title("Squares")

# right graphic
ax[1].bar(subjects, scores)
ax[1].set_title("Subject Scores")

plt.tight_layout()
plt.show()

# --- Descriptive Stats ---
from statistics import mode

# Q1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Variance:", np.var(data))
print("Standard Deviation:", np.std(data))

# Q2

data = np.random.normal(65, 10, 500)

plt.hist(data, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")

plt.show()

# Q3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.show()

# Q4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.figure(figsize=(8, 5))
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.show()

# Q5
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print("data1 mean:", np.mean(data1))
print("data1 median:", np.median(data1))
print("data1 mode:", mode(data1))

print("data2 mean:", np.mean(data2))
print("data2 median:", np.median(data2))
print("data2 mode:", mode(data2))


# --- Hypothesis ---

from scipy import stats

# Q1
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]


t_stat, p_value = stats.ttest_ind(group_a, group_b)

print("t-statistic:", t_stat)
print("p-value:", p_value)

# Q2
if p_value < 0.05:
    print("Q2: The result is statistically significant at alpha = 0.05.")
else:
    print("Q2: The result is NOT statistically significant at alpha = 0.05.")

# Q3
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat_q3, p_value_q3 = stats.ttest_rel(before, after)
print("Q3 t-statistic:", t_stat_q3)
print("Q3 p-value:", p_value_q3)

# Q4
scores = [72, 68, 75, 70, 69, 74, 71, 73]

t_stat_q4, p_value_q4 = stats.ttest_1samp(scores, 70)
print("Q4 t-statistic:", t_stat_q4)
print("Q4 p-value:", p_value_q4)

# Q5
t_stat_q5, p_value_q5 = stats.ttest_ind(group_a, group_b, alternative="less")
print("Q5 one-tailed p-value:", p_value_q5)

# Q6
print("Q6: Group A scored lower on average than Group B, and this difference is unlikely to be due to chance.")

# --- Correlation ---
import seaborn as sns
from scipy.stats import pearsonr

# Q1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix_q1 = np.corrcoef(x, y)
print("Correlation Q1 matrix:")
print(corr_matrix_q1)
print("Correlation Q1 coefficient [0,1]:", corr_matrix_q1[0, 1])

# Q2
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

corr_q2, p_value_corr_q2 = pearsonr(x, y)
print("Correlation Q2 coefficient:", corr_q2)
print("Correlation Q2 p-value:", p_value_corr_q2)

# Q3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)

print("Correlation Q3 matrix:")
print(df.corr())

# Q4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Q5
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# --- Pipeline Question 1 ---

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr): 
  created_s = pd.Series(arr, name="values")
  return created_s


def clean_data(created_series):
  clean_s = created_series.dropna().reset_index(drop=True)
  return clean_s


def summarize_data(cleaned_data):
  return {
    "mean": cleaned_data.mean(),
    "median": cleaned_data.median(),
    "std": cleaned_data.std(),
    "mode": cleaned_data.mode()[0]
  }


def data_pipeline(arr):
  series = create_series(arr)
  cleaned = clean_data(series)
  summary = summarize_data(cleaned)

  for key, value in summary.items():
    print(key, value)

  return summary

data_pipeline(arr)
