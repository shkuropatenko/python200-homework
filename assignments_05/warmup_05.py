from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}
    ]
)

# result
text = response.choices[0].message.content
model = response.model
tokens = response.usage.total_tokens

print("Response:", text)
print("Model:", model)
print("Tokens used:", tokens)

# I noticed that Python is beginner-friendly because the syntax is simple and readable.
# The response was clear and easy to understand.

# Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for t in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=t
    )

    text = response.choices[0].message.content
    print(f"\nTemperature {t}:")
    print(text)

# At temperature 0, the response is more consistent and predictable.
# At higher temperature, the answers become more creative but less consistent.
# I would use low temperature when I need stable results.

# Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}
    ],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices, 1):
    print(f"\nOption {i}:")
    print(choice.message.content)

# I see that using n=3 gives multiple answers in one request.
# This is useful when I want different options without making multiple API calls.

# Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain how neural networks work."}
    ],
    max_tokens=15
)

text = response.choices[0].message.content

print("Short response:")
print(text)

# The response is very short and cuts off the explanation.
# max_tokens limits how long the answer can be.
# This can be useful to control cost or keep responses concise.

# System Q1 - Sarcastic persona
messages = [
    {"role": "system", "content": "You are a sarcastic and impatient programmer who gives short answers."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSarcastic response:")
print(response.choices[0].message.content)

# The sarcastic response was short and direct.
# The tutor response was more friendly and easier to understand.

# System Q1 - Tutor persona
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nTutor response:")
print(response.choices[0].message.content)


# System Q2

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("Memory test response:")
print(response.choices[0].message.content)

# The model knows the name because we included it in the messages list.
# Even though the model is stateless, it can use the conversation history we provide.

# Prompt Q1

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, 1):
    prompt = f"Classify the sentiment (positive, negative, or mixed): {review}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content

    print(f"\nReview {i}:")
    print("Text:", review)
    print("Sentiment:", text)

# The model can classify sentiment even without examples.
# The output may vary in format since we did not specify it clearly.

# --- Prompt Engineering ---

# Prompt Q2 - One-Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = """
Classify each review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Now classify these reviews:
"""

for i, review in enumerate(reviews, 1):
    prompt += f'\nReview {i}: "{review}"\nSentiment:'

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q2 - One-Shot:")
print(response.choices[0].message.content)

# Adding one example helped the model understand the format better.
# The output looked more consistent than zero-shot.

# Prompt Q3 - Few-Shot
prompt = """
Classify each review as positive, negative, or mixed.

Examples:
Review: "The setup was easy and everything worked well."
Sentiment: positive

Review: "The app freezes every time I try to upload a file."
Sentiment: negative

Review: "The design looks nice, but the checkout process is confusing."
Sentiment: mixed

Now classify these reviews:
"""

for i, review in enumerate(reviews, 1):
    prompt += f'\nReview {i}: "{review}"\nSentiment:'

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q3 - Few-Shot:")
print(response.choices[0].message.content)

# Zero-shot is useful when the task is simple.
# One-shot helps show the format I want.
# Few-shot is better when I want more consistent answers.

# Prompt Q4 - Chain of Thought
prompt = """
Solve this problem step by step, then give a final answer clearly.

A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q4 - Chain of Thought:")
print(response.choices[0].message.content)

# Asking the model to reason step by step can help because it breaks the problem into smaller parts.
# This makes it easier to catch mistakes in calculations.

# Prompt Q5 - Structured Output
import json

review = "I've been using this tool for three months. It handles large datasets well, but the UI is clunky and the export options are limited."

prompt = f"""
Analyze the review below.

Return ONLY valid JSON with these keys:
- sentiment
- confidence
- reason

The confidence should be a float from 0 to 1.
The reason should be one sentence.

Review:
```{review}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

raw_response = response.choices[0].message.content

print("\nPrompt Q5 - Raw JSON Response:")
print(raw_response)

try:
    data = json.loads(raw_response)

    print("\nParsed JSON:")
    print("Sentiment:", data["sentiment"])
    print("Confidence:", data["confidence"])
    print("Reason:", data["reason"])

except json.JSONDecodeError:
    print("The response was not valid JSON.")
    print("Raw response:", raw_response)

# Prompt Q6 - Delimiters
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q6 - Instructions Text:")
print(response.choices[0].message.content)


regular_text = "Data engineering is an important part of working with modern data systems. It helps move and organize data for analysis."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{regular_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q6 - Regular Text:")
print(response.choices[0].message.content)

# Delimiters help separate the user's text from the instructions.
# This makes the prompt clearer and helps prevent the model from mixing them together.

# --- Ollama Question 1 ---

# Ollama output:
"""
I tried to run Ollama, but Windows returned:
'ollama' is not recognized as an internal or external command,
operable program or batch file.

This means Ollama is probably not installed or not added to PATH on my system yet.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain what a large language model is in two sentences."}
    ]
)

print("\nOllama Q1 - OpenAI Response:")
print(response.choices[0].message.content)

# I noticed that the OpenAI response was more polished and easier to read.
# The local Ollama model can run on my own computer, which is an advantage because it does not need an API call.
# One disadvantage is that the smaller local model may give lower quality or less detailed answers.