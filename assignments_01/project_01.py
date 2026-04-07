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

@flow
def happiness_pipeline():
  df = load_and_merge_data()
  descriptive_stats(df)
  create_plots(df)

if __name__ == "__main__":
  happiness_pipeline()