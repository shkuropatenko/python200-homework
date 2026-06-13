# --- LLMs as Transform ---

# Q1

# Parse the string "Jan 5th, 2024" into ISO format:
# Deterministic code, because date parsing follows fixed rules.

# Classify a customer support ticket:
# LLM, because the text can be written in many different ways.

# Calculate the average of a list of numbers:
# Deterministic code, because math should be exact.

# Extract the company name from a freeform job title:
# LLM, because job titles can have many inconsistent formats.

# Determine whether a review is more than 100 words long:
# Deterministic code, because counting words is straightforward.


# Q2

# The problem is that "a few sentences" produces inconsistent output.
# Different summaries may have different lengths and formats,
# making downstream parsing difficult.
#
# Better prompt:
#
# system = (
#     "Summarize the review in exactly one sentence. "
#     "Return plain text only."
# )


# Q3

# 50,000 records at 1 second each would take about 50,000 seconds
# (approximately 13.9 hours).
#
# One practical solution is parallel or batch processing
# so multiple requests run at the same time.


# --- Azure OpenAI ---

# Q1

# Organizations use Azure OpenAI because:
# 1. It integrates with Azure security and compliance controls.
# 2. Data remains inside the organization's Azure environment.


# Q2

# Azure-specific parameters:
#
# azure_endpoint:
# The URL of the Azure OpenAI resource.
#
# api_version:
# The API version to use.
#
# azure_deployment:
# The deployment configured inside Azure OpenAI.


# Q3

# The model parameter takes the deployment name,
# not a model name like "gpt-4o-mini".
#
# The deployment name is configured in Azure OpenAI Studio.