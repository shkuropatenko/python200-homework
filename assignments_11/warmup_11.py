from prefect import task, get_run_logger


# --- Prefect Orchestration ---

# Q1
# A @task is one step in a pipeline, like extract, transform, or load.
# A @flow controls the full pipeline and decides the order of tasks.
# I would not decorate a simple Celsius to Fahrenheit helper with @task
# because it is a pure in-memory calculation with no I/O or important side effect.

# Q2
@task(retries=3, retry_delay_seconds=30)
def call_api():
  pass


# Q3
# If extract is Completed, transform is Failed, and load never ran,
# I would open the failed flow run in the Prefect UI and inspect the transform task.
# I would look at task logs, error messages, stack traces, and inputs/outputs
# to understand why the transform step failed.


# --- Production Patterns ---

# Q1
# raise_for_status() raises an exception when an HTTP request returns an error status like 500.
# This is better than only printing an error because Prefect can mark the task as Failed.
# If the API returns 500 and we only print an error, downstream tasks might still run with bad data.
# With raise_for_status(), the pipeline stops correctly and the failure is visible in the UI.

# Q2
# overwrite=True protects me when I rerun the pipeline after a crash.
# If a previous run already created the same blob path, overwrite=True replaces it safely.
# Without overwrite=True, the upload could fail because the blob already exists.

# Q3
@task
def log_loaded_records(records: list, blob_path: str):
  logger = get_run_logger()
  logger.info(f"Loaded {len(records)} records to {blob_path}")