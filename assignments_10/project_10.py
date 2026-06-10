# Video link: https://youtu.be/JndwY6_xkfA

# Reflection:
# This task can be done with an LLM, but it is not the only option.
# A rule-based approach could also classify running conditions using clear thresholds.
# The LLM is more flexible, but deterministic code would be cheaper, faster, and more predictable.
# I would use an LLM if the rules were more subjective or harder to define.

import json
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


load_dotenv()

ACCOUNT_URL = "https://dmytroctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

OUTPUT_DIR = Path("assignments_10/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
  "You are classifying hourly weather conditions for outdoor running. "
  "Given a temperature in Celsius and a precipitation amount in mm, "
  "classify the conditions as exactly one of: good, marginal, or bad. "
  "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}


def classify_conditions(client, temperature, precipitation):
  user_message = f"Temperature: {temperature}C, Precipitation: {precipitation}mm"

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
    return "unknown"

  return label


def reshape_hourly_data(weather_json):
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

  return records


def main():
  openai_client = OpenAI()
  credential = DefaultAzureCredential()

  blob_service_client = BlobServiceClient(
    account_url=ACCOUNT_URL,
    credential=credential
  )

  container_client = blob_service_client.get_container_client(CONTAINER)

  today = date.today().isoformat()

  raw_blob_path = f"raw/{today}/weather.json"
  processed_blob_path = f"processed/{today}/weather_classified.json"

  raw_blob_client = container_client.get_blob_client(raw_blob_path)

  print(f"Reading raw blob: {raw_blob_path}")

  try:
    raw_bytes = raw_blob_client.download_blob().readall()
    weather_json = json.loads(raw_bytes.decode("utf-8"))
    print("Downloaded raw weather data from Blob Storage")

  except Exception:
    fallback_path = Path("assignments_09/outputs/weather_raw.json")
    print("Could not find today's blob. Using fallback file:", fallback_path)

    with open(fallback_path, "r", encoding="utf-8") as f:
      weather_json = json.load(f)

  records = reshape_hourly_data(weather_json)

  first_24_records = records[:24]

  enriched_records = []

  print("\nClassifying first 24 records...")

  for i, record in enumerate(first_24_records, start=1):
    label = classify_conditions(
      openai_client,
      record["temperature_2m"],
      record["precipitation"]
    )

    enriched_record = {
      **record,
      "conditions": label
    }

    enriched_records.append(enriched_record)

    if i % 6 == 0:
      print(f"Processed {i} records")

  processed_bytes = json.dumps(
    enriched_records,
    indent=2
  ).encode("utf-8")

  processed_blob_client = container_client.get_blob_client(processed_blob_path)

  processed_blob_client.upload_blob(
    processed_bytes,
    overwrite=True
  )

  print(f"\nUploaded processed blob: {processed_blob_path}")
  print(f"Bytes uploaded: {len(processed_bytes)}")

  downloaded_processed = processed_blob_client.download_blob().readall()
  processed_records = json.loads(downloaded_processed.decode("utf-8"))

  df = pd.DataFrame(processed_records)

  print("\nCondition counts:")
  print(df["conditions"].value_counts())

  print("\nFirst 5 rows:")
  print(df.head())

  first_10_path = OUTPUT_DIR / "first_10_records.json"

  with open(first_10_path, "w", encoding="utf-8") as f:
    json.dump(enriched_records[:10], f, indent=2)

  print(f"\nSaved first 10 records to {first_10_path}")


if __name__ == "__main__":
  main()