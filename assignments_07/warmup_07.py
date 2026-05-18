from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

print("Warmup 07 started")


# --- Lesson 02: Tool Definitions and ReAct Loop ---

# Q1

def celsius_to_fahrenheit(celsius: float) -> str:
  """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
  fahrenheit = (celsius * 9 / 5) + 32
  return f"{celsius}°C is {fahrenheit}°F"


celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "Temperature in Celsius"
                }
            },
            "required": ["celsius"]
        }
    }
}


print("\nQ1 direct function calls:")
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))

# This tool is simple: it takes one number in Celsius and returns a formatted Fahrenheit result.

import json
from datetime import datetime


def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}


def run_agent(user_message: str) -> str:
    tools = [get_current_time_schema]

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools only when needed."},
        {"role": "user", "content": user_message}
    ]

    first_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_message = first_response.choices[0].message

    if not assistant_message.tool_calls:
        return assistant_message.content

    messages.append(assistant_message)

    for tool_call in assistant_message.tool_calls:
        if tool_call.function.name == "get_current_time":
            tool_result = get_current_time()
        else:
            tool_result = "Unknown tool"

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })

    second_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return second_response.choices[0].message.content


# Q2 Prediction:
# I do not think this will trigger a tool call because the only available tool gets the current time.
# The agent does not have the Celsius conversion tool yet.
# I expect one API call if the model answers directly, or two API calls if it incorrectly calls a tool.

print("\nQ2:")
result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(result)

# My prediction was that the Celsius tool would not be called because it was not available yet.
# The prediction was correct.
# The agent did not call a tool because only the time tool was available.
# The model answered directly using its own knowledge, so this took one API call.

# Q3

def run_agent_with_two_tools(user_message: str) -> str:
    tools = [
      get_current_time_schema,
      celsius_to_fahrenheit_schema
    ]

    messages = [
      {"role": "system", "content": "You are a helpful assistant. Use tools when they are useful."},
      {"role": "user", "content": user_message}
    ]

    first_response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=messages,
      tools=tools,
      tool_choice="auto"
    )

    assistant_message = first_response.choices[0].message

    if not assistant_message.tool_calls:
      return assistant_message.content

    messages.append(assistant_message)

    for tool_call in assistant_message.tool_calls:
      tool_name = tool_call.function.name
      arguments = json.loads(tool_call.function.arguments)

      if tool_name == "get_current_time":
        tool_result = get_current_time()

      elif tool_name == "celsius_to_fahrenheit":
        tool_result = celsius_to_fahrenheit(arguments["celsius"])

      else:
        tool_result = "Unknown tool"

      messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result
      })

    second_response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=messages
    )

    return second_response.choices[0].message.content


print("\nQ3:")

response_a = run_agent_with_two_tools("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)

# A tool should be called here because the Celsius conversion tool is available
# and the user is asking for a temperature conversion.

response_b = run_agent_with_two_tools("What is the boiling point of water in plain English?")
print("Response B:", response_b)

# A tool may not be needed here because this is a general knowledge question.
# The model can answer in plain English without converting a specific input.

# Response A used the Celsius conversion tool because the question asked for a specific conversion.
# Response B probably did not need a tool because the model already knows the boiling point of water.