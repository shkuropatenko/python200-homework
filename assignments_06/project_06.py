from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


if load_dotenv():
  print("API key loaded successfully.")
else:
  print("Warning: could not load API key. Check your .env file.")


Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


# Step 1: Setup
docs_dir = Path("assignments_06/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"


# Step 2: Load the Documents
documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print("\nDocuments loaded:", len(documents))

for doc in documents:
  file_name = doc.metadata.get("file_name", "unknown")
  print("File:", file_name)


# Step 3: Build the Index and Query Engine
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
  similarity_top_k=3
)

print("\nIndex built successfully. Ready to answer questions.")


# Step 4: Query the Assistant
questions = [
  "What are Groundwork's hours on weekends?",
  "Do you offer any dairy-free milk options?",
  "How does the loyalty program work?",
  "How did Groundwork Coffee get started?",
  "Do you offer catering or wholesale orders?",
]

for question in questions:
  response = query_engine.query(question)

  print("\nQuestion:", question)
  print("Answer:", response)

  if response.source_nodes:
    top_node = response.source_nodes[0]
    file_name = top_node.node.metadata.get("file_name", "unknown")
    score = top_node.score
    text_preview = top_node.node.text[:200].replace("\n", " ")

    print("\nTop source node:")
    print("File:", file_name)
    print("Score:", score)
    print("Text:", text_preview)

  print("-" * 50)

# The assistant sounded mostly confident because it had documents to retrieve from.
# I still need to check the source text because a RAG system can retrieve the wrong chunk.
# The answers are only trustworthy if the retrieved source node matches the question.


# Step 5: Find a Failure
failure_question = "Does Groundwork Coffee offer birthday discounts for customers?"

response = query_engine.query(failure_question)

print("\nFailure Question:", failure_question)
print("Answer:", response)

print("\nRetrieved source nodes:")
for node in response.source_nodes:
  file_name = node.node.metadata.get("file_name", "unknown")
  score = node.score
  text_preview = node.node.text[:200].replace("\n", " ")

  print("File:", file_name)
  print("Score:", score)
  print("Text:", text_preview)
  print("-" * 50)

# I asked about birthday discounts because I do not expect that information to be in the documents.
# If the retrieval does not find a relevant chunk, the model should say that it does not know.
# If it guesses confidently, that would be a hallucination risk.
# To improve the system, I would add a stronger instruction telling the assistant to only answer from the retrieved context.


# Step 6: Reflection

# LlamaIndex made the RAG implementation much shorter than building semantic RAG manually.
# Instead of writing separate code for chunking, embeddings, indexing, and retrieval,
# I only needed a few lines to load documents, build an index, and create a query engine.

# Another use case would be an HR assistant for a company.
# It could answer employee questions using internal documents like benefits policies,
# PTO rules, onboarding guides, and security policies.

# One failure mode RAG cannot fully prevent is bad or incomplete source documents.
# If the documents are outdated or missing information, the system may still return a weak answer.