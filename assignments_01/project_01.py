from prefect import flow, task, get_run_logger
import pandas as pd

# Task 1: Load Multiple Years of Data
@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data():
  logger = get_run_logger()

  base_url = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/main/assignments/resources/happiness_project"
  dfs = []

  for year in range(2015, 2025):
    url = f"{base_url}/world_happiness_{year}.csv"
    logger.info(f"Loading data for {year} from {url}")

    df = pd.read_csv(url, sep=";")
    df["year"] = year
    dfs.append(df)

  merged_df = pd.concat(dfs, ignore_index=True)
  logger.info(f"Merged shape: {merged_df.shape}")

  merged_df.to_csv("assignments_01/outputs/merged_happiness.csv", index=False)
  logger.info("Merged dataset saved to assignments_01/outputs/merged_happiness.csv")

  return merged_df

# Task 2: Descriptive Statistics
@task
def descriptive_stats(df):
  logger = get_run_logger()
  df["Happiness score"] = df["Happiness score"].str.replace(",", ".").astype(float)

  logger.info(f"Mean: {df['Happiness score'].mean()}")
  logger.info(f"Median: {df['Happiness score'].median()}")
  logger.info(f"Std: {df['Happiness score'].std()}")

  by_year = df.groupby("year")["Happiness score"].mean()
  logger.info(f"Mean by year:\n{by_year}")

  by_region = df.groupby("Regional indicator")["Happiness score"].mean()
  logger.info(f"Mean by region:\n{by_region}")

# Task 3: Visual Exploration
@task
def create_plots(df):
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  logger = get_run_logger()

  df["Happiness score"] = df["Happiness score"].astype(float)

  # Histogram
  plt.figure()
  df["Happiness score"].hist()
  plt.title("Happiness Score Distribution")
  plt.savefig("assignments_01/outputs/happiness_histogram.png")
  logger.info("Histogram saved")

  # Boxplot
  plt.figure()
  sns.boxplot(x="year", y="Happiness score", data=df)
  plt.title("Happiness by Year")
  plt.savefig("assignments_01/outputs/happiness_by_year.png")

  logger.info("Boxplot saved")

  # Scatter
  df["GDP per capita"] = df["GDP per capita"].str.replace(",", ".").astype(float)
  plt.figure()
  sns.scatterplot(x="GDP per capita", y="Happiness score", data=df)
  plt.title("GDP vs Happiness Score")
  plt.savefig("assignments_01/outputs/gdp_vs_happiness.png")
  logger.info("Scatter plot saved")

  # Heatmap
  numeric_cols = [
    "Happiness score",
    "GDP per capita",
    "Social support",
    "Healthy life expectancy",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption",
    "year",
  ]

  for col in numeric_cols:
    if col in df.columns:
      df[col] = (
        df[col]
          .astype(str)
          .str.replace(",", ".", regex=False)
      )
      df[col] = pd.to_numeric(df[col], errors="coerce")

  corr = df[numeric_cols].corr()

  plt.figure(figsize=(10, 8))
  sns.heatmap(corr, annot=True, cmap="coolwarm")
  plt.title("Correlation Heatmap")
  plt.savefig("assignments_01/outputs/correlation_heatmap.png")
  logger.info("Correlation heatmap saved")

# Task 4: Hypothesis Testing
@task
def hypothesis_test(df):
  from scipy.stats import ttest_ind
  
  logger = get_run_logger()

  df["Happiness score"] = (
    df["Happiness score"]
    .astype(str)
    .str.replace(",", ".")
    .astype(float)
  )

  before = df[df["year"] == 2019]["Happiness score"]
  after = df[df["year"] == 2020]["Happiness score"]

  t_stat, p_value = ttest_ind(before, after, nan_policy="omit")

  logger.info(f"T-statistic: {t_stat}")
  logger.info(f"P-value: {p_value}")
  logger.info(f"Mean 2019: {before.mean()}")
  logger.info(f"Mean 2020: {after.mean()}")

  if p_value < 0.05:
    logger.info("Statistically significant difference between 2019 and 2020")
  else:
    logger.info("No significant difference between 2019 and 2020")

@flow
def happiness_pipeline():
  df = load_and_merge_data()
  descriptive_stats(df)
  create_plots(df)

if __name__ == "__main__":
  happiness_pipeline()