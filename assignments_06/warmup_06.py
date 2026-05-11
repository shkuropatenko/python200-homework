from dotenv import load_dotenv
import os

if load_dotenv():
  print("API key loaded successfully.")
else:
  print("Warning: could not load API key. Check your .env file.")


# --- RAG Concepts ---

# Concepts Question 1

# Scenario A:
# RAG would be the best choice because the legal documents are updated often.
# A retrieval system can search the latest PDFs without retraining the model.

# Scenario B:
# Fine-tuning would work best because the company wants a very specific writing style.
# They already have many examples of the tone they want.

# Scenario C:
# Prompt engineering or simple context injection is enough because it is only one short report.
# There is no need for a full retrieval system or fine-tuning.


# Concepts Question 2

# A confident hallucination is dangerous because people may trust the answer even when it is wrong.
# For example, a medical AI giving incorrect dosage instructions could seriously harm a patient.
# The tone matters because confident language makes the response sound trustworthy.


# Concepts Question 3

# Correct RAG pipeline order:

# 1. Receive the user's query
# The system first gets a question from the user.

# 2. Extract text from source documents
# Text is collected from PDFs or other files.

# 3. Split text into chunks
# Large documents are broken into smaller pieces.

# 4. Convert text chunks into embeddings
# Each chunk is converted into a vector representation.

# 5. Embed the user's query
# The user's question is also converted into a vector.

# 6. Retrieve the most relevant chunks
# The system finds chunks most similar to the query.

# 7. Inject retrieved chunks into the prompt
# Relevant chunks are added to the model input.

# 8. Generate a response from the LLM
# The model creates a final answer using the retrieved context.

print("Warmup 06 started")