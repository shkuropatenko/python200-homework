# --- Azure Authentication ---

# Q1
# When I run a Python script locally with DefaultAzureCredential,
# it can use my Azure CLI login session.
# I need to run "az login" first.
# DefaultAzureCredential checks different authentication methods in order,
# and one of them is AzureCliCredential, which uses the active az login session.

# Q2
# A deployed pipeline on an Azure VM or container should not use az login
# because there is no human manually logging in.
# In production, it usually uses managed identity.
# The same Python code can work because DefaultAzureCredential checks the environment
# and uses the right credential method depending on where the code is running.

# Q3
# If I get an AuthenticationError, two likely causes are:
# 1. I did not run az login, or my Azure CLI session expired.
#    I would diagnose this by running "az account show".
# 2. I am logged into the wrong tenant or subscription.
#    I would check "az account list" and verify the active subscription.


# --- Blob Storage ---

# Q1
# Azure Blob Storage has three levels:
# Storage account -> container -> blob.
# An analogy is a filing cabinet:
# the storage account is the cabinet,
# the container is a drawer,
# and the blob is a file inside the drawer.

# Q2
# Scenario 1: I would use Blob Storage because raw JSON API responses are files
# that may need to be stored and reprocessed later.

# Scenario 2: I would use a relational database like Azure SQL because the analytics team
# needs to query structured transaction data by date and customer ID.

# Scenario 3: I would use Blob Storage because NumPy arrays or model artifacts are files
# that can be saved between pipeline runs.


# Q3
def list_container(container_client):
  """Print the name and size of every blob in a container."""
  blobs = container_client.list_blobs()

  for blob in blobs:
    print(f"{blob.name} - {blob.size} bytes")


# Q4
def upload_text(container_client, blob_name, text):
  """Upload a text string to Blob Storage as UTF-8."""
  data = text.encode("utf-8")

  container_client.upload_blob(
    name=blob_name,
    data=data,
    overwrite=True
  )