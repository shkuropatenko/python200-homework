from dotenv import load_dotenv
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from smolagents import tool, CodeAgent, OpenAIServerModel

load_dotenv()

print("Project 07 started")

df = None

DATA_PATH = "assignments_01/outputs/merged_happiness.csv"


@tool
def load_happiness_data() -> dict:
  """Load the World Happiness dataset into memory.

  Returns:
      A dictionary with the dataset shape and column names.
  """
  global df

  path = Path(DATA_PATH)

  if path.exists():
    df = pd.read_csv(path)

    df.columns = (
      df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
    )

    df = df.rename(columns={
      "regional_indicator": "region"
    })

    numeric_columns = [
      "happiness_score",
      "gdp_per_capita",
      "social_support",
      "healthy_life_expectancy",
      "freedom_to_make_life_choices",
      "generosity",
      "perceptions_of_corruption",
      "ladder_score"
    ]

    for col in numeric_columns:
      if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )

        df[col] = pd.to_numeric(df[col], errors="coerce")

  else:
    return {"error": f"File not found: {DATA_PATH}"}

  return {
    "shape": df.shape,
    "columns": list(df.columns)
  }


@tool
def summarize_column(column: str) -> dict:
  """Return descriptive statistics for one column.

  Args:
    column: The column name to summarize.

  Returns:
    A dictionary with descriptive statistics for the column.
  """
  if df is None:
    return {"error": "No data loaded"}

  if column not in df.columns:
    return {"error": f"Column not found: {column}"}

  return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
  """Compute Pearson correlation between two numeric columns.

  Args:
    col1: The first numeric column.
    col2: The second numeric column.

  Returns:
    A dictionary with column names, Pearson correlation coefficient, and p-value.
  """
  if df is None:
      return {"error": "No data loaded"}

  if col1 not in df.columns:
    return {"error": f"Column not found: {col1}"}

  if col2 not in df.columns:
    return {"error": f"Column not found: {col2}"}

  clean_df = df[[col1, col2]].dropna()

  r, p = pearsonr(clean_df[col1], clean_df[col2])

  return {
    "col1": col1,
    "col2": col2,
    "pearson_r": round(float(r), 4),
    "p_value": round(float(p), 4)
  }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
  """Return the top N countries ranked by a column for a specific year.

  Args:
    column: The column used for ranking.
    year: The year to filter by.
    n: The number of countries to return.

  Returns:
    A dictionary with the top countries and their values.
  """
  if df is None:
    return {"error": "No data loaded"}

  if column not in df.columns:
    return {"error": f"Column not found: {column}"}

  if "year" not in df.columns:
    return {"error": "Column not found: year"}

  if "country" not in df.columns:
    return {"error": "Column not found: country"}

  filtered = df[df["year"] == year]

  if filtered.empty:
    return {"error": f"No rows found for year {year}"}

  top_rows = (
    filtered
    .sort_values(column, ascending=False)
    .head(n)
  )

  result = top_rows[["country", column]].to_dict(orient="records")

  return {"results": result}


model = OpenAIServerModel(
  model_id="gpt-4o-mini"
)

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient.
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
  tools=[
    load_happiness_data,
    summarize_column,
    compute_correlation,
    get_top_n_countries
  ],
  model=model,
  instructions=SYSTEM_PROMPT,
  additional_authorized_imports=[
    "pandas",
    "matplotlib.pyplot",
    "scipy.stats"
  ],
  max_steps=8,
)


if __name__ == "__main__":

  queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to assignments_07/outputs/happiness_by_region.png.",
  ]

  for query in queries:
    print(f"\n--- Query: {query} ---")
    response = agent.run(query, reset=False)
    print(response)

  # My query 1
  my_query_1 = "Which region has the highest average happiness_score overall?"
  response_1 = agent.run(my_query_1, reset=False)
  print("\nMy Query 1:")
  print(response_1)

  # Comment:
  # This triggered both tool usage and code generation.

  # My query 2
  my_query_2 = "Create a histogram of happiness_score and save it to assignments_07/outputs/happiness_histogram.png."
  response_2 = agent.run(my_query_2, reset=False)
  print("\nMy Query 2:")
  print(response_2)

  # Comment:
  # This mostly triggered code generation because plotting was not covered by a tool.
  # The CodeAgent struggled to access the underlying DataFrame directly.
  # It attempted multiple plotting strategies and generated several incorrect assumptions.
  # This demonstrates both the flexibility and instability of autonomous code-generation agents.


# --- Reflection ---
#
# 1. The agent used the p-value to explain statistical significance.
#    It correctly treated a very small p-value as statistically significant.
#    The agent appeared to use a standard threshold of p < 0.05.
#
# 2. One surprising thing was how the CodeAgent attempted multiple different
#    strategies when plotting failed. It showed persistence, but it also
#    hallucinated data structures and made incorrect assumptions.
#
# 3. One useful additional tool would be a plotting helper tool.
#    It could automatically generate charts from selected columns and save them.
#    This would help the agent create reliable visualizations without repeatedly
#    generating fragile plotting code.