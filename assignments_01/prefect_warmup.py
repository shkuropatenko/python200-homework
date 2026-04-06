import numpy as np
import pandas as pd

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr): 
  created_s = pd.Series(arr, name="values")
  return created_s

created_series = create_series(arr)

# print(created_series)

def clean_data(created_series):
  clean_s = created_series.dropna().reset_index(drop=True)
  return clean_s

cleaned_data = clean_data(created_series)

# print(cleaned_data)


def summarize_data(cleaned_data):
  return {
    "mean": cleaned_data.mean(),
    "median": cleaned_data.median(),
    "std": cleaned_data.std(),
    "mode": cleaned_data.mode()[0]
  }
summarized_data = summarize_data(cleaned_data)
print(summarized_data)