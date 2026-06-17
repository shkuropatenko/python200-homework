# Video Link: https://youtu.be/wwNB6qNrnQ0

import json
from datetime import date

from dotenv import load_dotenv
from openai import OpenAI
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import requests
from prefect import flow, task

load_dotenv()

ACCOUNT_URL = "https://dmytroctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

SYSTEM_PROMPT = (
  "You are classifying hourly weather conditions for outdoor running. "
  "Given a temperature in Celsius and a precipitation amount in mm, "
  "classify the conditions as exactly one of: good, marginal, or bad. "
  "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}


@task(retries=2, retry_delay_seconds=10)
def extract_weather():
  url = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=35.2271"
    "&longitude=-80.8431"
    "&hourly=temperature_2m,precipitation"
    "&forecast_days=7"
  )

  response = requests.get(url)
  response.raise_for_status()

  print("Extracted weather data from Open-Meteo API")
  return response.json()


@task
def transform_weather(weather_json: dict):
  client = OpenAI()

  hourly = weather_json["hourly"]

  records = []

  for time, temp, precip in zip(
    hourly["time"],
    hourly["temperature_2m"],
    hourly["precipitation"]
  ):
    records.append({
      "time": time,
      "temperature_2m": temp,
      "precipitation": precip
    })

  enriched_records = []

  for i, record in enumerate(records[:24], start=1):
    user_message = (
      f"Temperature: {record['temperature_2m']}C, "
      f"Precipitation: {record['precipitation']}mm"
    )

    response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
      ],
      temperature=0,
    )

    label = response.choices[0].message.content.strip().lower()

    if label not in VALID_LABELS:
      label = "unknown"

    enriched_records.append({
      **record,
      "conditions": label
    })

    if i % 6 == 0:
      print(f"Transformed {i} records")

  return enriched_records


@task
def load_weather(enriched_records: list):
  credential = DefaultAzureCredential()

  blob_service_client = BlobServiceClient(
    account_url=ACCOUNT_URL,
    credential=credential
  )

  container_client = blob_service_client.get_container_client(CONTAINER)

  today = date.today().isoformat()
  blob_path = f"final/{today}/weather_etl.json"

  json_bytes = json.dumps(enriched_records, indent=2).encode("utf-8")

  blob_client = container_client.get_blob_client(blob_path)

  blob_client.upload_blob(
    json_bytes,
    overwrite=True
  )

  print(f"Uploaded final blob: {blob_path}")
  print(f"Bytes uploaded: {len(json_bytes)}")

  return blob_path


@flow(log_prints=True)
def cloud_etl_pipeline():
  raw_weather = extract_weather()
  enriched_records = transform_weather(raw_weather)
  blob_path = load_weather(enriched_records)

  print(f"Pipeline completed successfully. Final blob: {blob_path}")


if __name__ == "__main__":
  cloud_etl_pipeline()