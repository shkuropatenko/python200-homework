from dotenv import load_dotenv
import os
import string

if load_dotenv():
  print("API key loaded successfully.")
else:
  print("Warning: could not load API key. Check your .env file.")

print("Warmup 06 started")


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


# --- Keyword RAG ---

def simple_keyword_retrieval(query, documents, verbose=True):
  """Keyword retrieval using token overlap scoring."""

  stopwords = {
    "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
    "are", "was", "were", "by", "with", "at", "from", "that", "this",
    "as", "be", "it", "its", "their", "they", "we", "you", "our",
    "what", "your", "do", "i", "how", "have"
  }

  translator = str.maketrans("", "", string.punctuation)

  query_words = {
    w.translate(translator)
    for w in query.lower().split()
    if w not in stopwords
  }

  if verbose:
    print(f"\nQuery tokens (filtered): {sorted(query_words)}")

  scores = []

  for name, content in documents.items():

    content_words = {
      w.translate(translator)
      for w in content.lower().split()
      if w not in stopwords
    }

    overlap = query_words & content_words
    score = len(overlap)

    scores.append((score, name, content))

    if verbose:
      print(f"[{name}] overlap={score} -> {sorted(overlap)}")

  scores.sort(reverse=True)

  best = next(
    ((name, content) for score, name, content in scores if score > 0),
    None
  )

  if best:
    if verbose:
      print(f"\nSelected best match: {best[0]}")
    return [best]

  else:
    if verbose:
      print("\nNo overlapping keywords found.")

    return [("None found", "No relevant content.")]


documents = {
  "menu.txt": (
    "We serve espresso, lattes, cappuccinos, and cold brew. "
    "Pastries include croissants and muffins baked fresh daily. "
    "Oat milk and almond milk are available."
  ),

  "hours.txt": (
    "We are open Monday through Friday from 7am to 7pm. "
    "On weekends we open at 8am and close at 5pm. "
    "We are closed on Thanksgiving and Christmas Day."
  ),

  "hiring.txt": (
    "We are currently hiring baristas and shift supervisors. "
    "Send your resume to jobs@groundworkcoffee.com."
  ),

  "loyalty.txt": (
    "Join our loyalty program to earn one point per dollar spent. "
    "Redeem 100 points for a free drink of your choice."
  ),
}


print("\nDocuments loaded:")
print(documents.keys())


# Keyword Question 1

query = "What are your hours on weekends?"

result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nKeyword Q1 selected document:")
print(result[0][0])

# The selected document is hours.txt because the query uses "hours" and "weekends",
# which overlap with the business hours document.

# Keyword Question 2

query = "Do you have anything without caffeine?"

result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nKeyword Q2 selected document:")
print(result[0][0])

# Keyword RAG may not get this right because the menu does not contain the exact word "caffeine".
# Semantic RAG would work better because it can understand meaning and related ideas.


# Keyword Question 3

# Prediction:
# I think loyalty.txt may be selected because rewards are related to loyalty programs.
# However, keyword search only checks exact words, so it may fail.

query = "How do I sign up for rewards?"

result = simple_keyword_retrieval(query, documents, verbose=True)

print("\nKeyword Q3 selected document:")
print(result[0][0])

# Keyword retrieval has limitations because it depends on exact word overlap.
# Semantic retrieval would better understand that rewards and loyalty are related.

# Semantic Question 2

# Keyword RAG:
# - compares exact words
# - retrieves full documents
# - cannot handle synonyms well
# - stores plain text
# - uses keyword overlap scores

# Semantic RAG:
# - compares meaning using embeddings
# - retrieves relevant chunks
# - can understand synonyms and related ideas
# - stores vectors / embeddings
# - uses cosine similarity or vector similarity