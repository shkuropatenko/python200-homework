# Video link:
# https://youtu.be/4oOBmHQgnIw

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


ACCOUNT_URL = "https://dmytroctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

OUTPUT_DIR = Path("assignments_09/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
  credential = DefaultAzureCredential()

  blob_service_client = BlobServiceClient(
    account_url=ACCOUNT_URL,
    credential=credential
  )

  container_client = blob_service_client.get_container_client(CONTAINER)

  print("Connected to Azure Blob Storage")

  # Step 1: Extract
  url = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=35.2271"
    "&longitude=-80.8431"
    "&hourly=temperature_2m,precipitation"
    "&forecast_days=7"
  )

  response = requests.get(url)
  response.raise_for_status()

  weather_data = response.json()
  print("Weather data extracted from Open-Meteo API")

  # Step 2: Serialize
  json_bytes = json.dumps(weather_data).encode("utf-8")

  # Step 3: Load
  today = date.today().isoformat()
  blob_path = f"raw/{today}/weather.json"

  blob_client = container_client.get_blob_client(blob_path)

  blob_client.upload_blob(
    json_bytes,
    overwrite=True
  )

  print(f"Uploaded blob: {blob_path}")
  print(f"Bytes uploaded: {len(json_bytes)}")

  # Step 4: Verify
  print("\nBlobs in container:")
  for blob in container_client.list_blobs():
    print(f"{blob.name} - {blob.size} bytes")

  # Step 5: Read Back
  downloaded_bytes = blob_client.download_blob().readall()

  downloaded_json = json.loads(downloaded_bytes.decode("utf-8"))

  output_file = OUTPUT_DIR / "weather_raw.json"

  with open(output_file, "wb") as f:
    f.write(downloaded_bytes)

  print(f"\nDownloaded JSON saved to {output_file}")

  hourly_df = pd.DataFrame(downloaded_json["hourly"])

  print("\nHourly DataFrame preview:")
  print(hourly_df.head())


if __name__ == "__main__":
  main()