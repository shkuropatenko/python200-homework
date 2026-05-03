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

# I noticed that with temperature 0 the answers are more predictable.
# With higher temperature, the names become more creative and different.
# I would use low temperature when I need consistent results.

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

# Q5
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

# I noticed that the system message changes how the model responds.
# The tutor response was more friendly and detailed.
# The second response was shorter and more sarcastic.

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