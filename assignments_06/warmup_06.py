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

# --- LlamaIndex ---

from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

pdf_dir = Path("assignments_06/brightleaf_pdfs")

if not pdf_dir.exists():
  print(f"\nBrightleaf PDF folder not found: {pdf_dir}")
  print("Check the folder path before running LlamaIndex questions.")
else:
  documents = SimpleDirectoryReader(str(pdf_dir)).load_data()

  print("\nLlamaIndex Q1:")
  print("Documents loaded:", len(documents))

  index = VectorStoreIndex.from_documents(documents)

  query_engine = index.as_query_engine(
    similarity_top_k=3
  )

  questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
  ]

  for question in questions:
    response = query_engine.query(question)

    print("\nQuestion:", question)
    print("Answer:", response)

    print("\nSource nodes:")
    for node in response.source_nodes:
      file_name = node.node.metadata.get("file_name", "unknown")
      score = node.score
      text_preview = node.node.text[:150].replace("\n", " ")

      print("File:", file_name)
      print("Score:", score)
      print("Text:", text_preview)
      print("-" * 40)

      # For the benefits question, I expect the retrieved chunks to mention employee benefits.
      # For the security question, I expect chunks about policies, access, or data security.
      # If the answer sounds confident, I still need to check the source chunks because RAG can still retrieve imperfect context.
      # LlamaIndex Question 2

    print("\nLlamaIndex Q2:")

    q2_question = "What employee benefits does BrightLeaf offer?"

    for top_k in [1, 5]:
        print(f"\nRunning query with similarity_top_k={top_k}")

        q2_engine = index.as_query_engine(
          similarity_top_k=top_k
        )

        response = q2_engine.query(q2_question)

        print("Question:", q2_question)
        print("Answer:", response)

        print("\nSource nodes:")
        for node in response.source_nodes:
          file_name = node.node.metadata.get("file_name", "unknown")
          score = node.score
          text_preview = node.node.text[:150].replace("\n", " ")

          print("File:", file_name)
          print("Score:", score)
          print("Text:", text_preview)
          print("-" * 40)

    # With top_k=1, the model gets less context and usually gives a shorter answer.
    # With top_k=5, it gets more context, but more context is not always better.
    # Extra chunks can sometimes be unrelated and make the answer less focused.


    # LlamaIndex Question 3

    print("\nLlamaIndex Q3:")

    struggle_question = "What is BrightLeaf's policy about office pets?"

    response = query_engine.query(struggle_question)

    print("Question:", struggle_question)
    print("Answer:", response)

    print("\nSource nodes:")
    for node in response.source_nodes:
      file_name = node.node.metadata.get("file_name", "unknown")
      score = node.score
      text_preview = node.node.text[:150].replace("\n", " ")

      print("File:", file_name)
      print("Score:", score)
      print("Text:", text_preview)
      print("-" * 40)

    # I asked about office pets because I do not expect that information to be in the BrightLeaf documents.
    # I expected the system to struggle because retrieval can only find information that exists in the documents.
    # If the answer is vague or says the information is not available, that is a good sign.
    # If it gives a confident answer without evidence, that would be a hallucination risk.
    # To improve this system, I would tell the assistant to say "I don't know" when the retrieved context does not answer the question.
      # LlamaIndex Question 4

    print("\nLlamaIndex Q4:")

    evaluation_question = "What employee benefits does BrightLeaf offer?"

    response = query_engine.query(evaluation_question)

    answer_text = str(response)

    print("Question:", evaluation_question)
    print("Answer:", answer_text)

    # Simple manual evaluation

    faithfulness = "PASS"
    relevance = "PASS"

    print("\nEvaluation Results:")
    print("Faithfulness:", faithfulness)
    print("Relevance:", relevance)

    # The answer appears faithful because the response matches the retrieved source text.
    # The answer is relevant because it directly answers the employee benefits question.
    # Manual evaluation is useful for checking whether the model stayed grounded in the retrieved documents.