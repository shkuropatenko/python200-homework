from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

ACCOUNT_URL = "https://dmytroctd2026sa.blob.core.windows.net"
CONTAINER_NAME = "pipeline-data"

credential = DefaultAzureCredential()

blob_service_client = BlobServiceClient(
    account_url=ACCOUNT_URL,
    credential=credential
)

container_client = blob_service_client.get_container_client(CONTAINER_NAME)

print("Connected to Azure Blob Storage")

blob_client = container_client.get_blob_client("test.txt")

blob_client.upload_blob(
    "Hello Azure Blob Storage!",
    overwrite=True
)

print("File uploaded successfully")