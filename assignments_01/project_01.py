from prefect import flow, task, get_run_logger
import pandas as pd

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

@flow
def happiness_pipeline():
  load_and_merge_data()

if __name__ == "__main__":
  happiness_pipeline()