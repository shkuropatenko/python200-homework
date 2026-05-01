from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

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