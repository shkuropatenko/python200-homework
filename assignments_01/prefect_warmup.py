import numpy as np
import pandas as pd

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr): 
  return pd.Series(arr, name="values")

create_series(arr)
print(create_series(arr))