from dotenv import load_dotenv
from openai import OpenAI
from smolagents import tool, ToolCallingAgent, CodeAgent, OpenAIServerModel

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

# --- Lesson 03: Multi-Tool Agent ---

import pandas as pd
from scipy.stats import pearsonr


# Q4

class CsvManager:
  def __init__(self):
    self.df = None
    self.path = None

  def load_csv(self, path: str):
    try:
      self.df = pd.read_csv(path)
      self.path = path
      return {
        "message": "CSV loaded successfully",
        "path": path,
        "shape": self.df.shape,
        "columns": list(self.df.columns)
      }
    except Exception as e:
      return {"error": str(e)}

  def list_columns(self):
    if self.df is None:
      return {"error": "No CSV loaded"}
    return {"columns": list(self.df.columns)}

  def summarize_column(self, column: str):
    if self.df is None:
      return {"error": "No CSV loaded"}
    if column not in self.df.columns:
      return {"error": f"Column not found: {column}"}
    return self.df[column].describe().to_dict()

  def compute_correlation(self, col1: str, col2: str):
    """
    Compute the Pearson correlation between two columns in the loaded DataFrame.
    Returns the correlation coefficient and p-value.
    """
    if self.df is None:
      return {"error": "No CSV loaded"}

    if col1 not in self.df.columns:
      return {"error": f"Column not found: {col1}"}

    if col2 not in self.df.columns:
      return {"error": f"Column not found: {col2}"}

    clean_df = self.df[[col1, col2]].dropna()

    r, p = pearsonr(clean_df[col1], clean_df[col2])

    return {
      "col1": col1,
      "col2": col2,
      "pearson_r": round(float(r), 4),
      "p_value": round(float(p), 4)
    }


csv_manager = CsvManager()


tools_schema = [
  {
    "type": "function",
    "function": {
      "name": "load_csv",
      "description": "Load a CSV file into memory.",
      "parameters": {
        "type": "object",
        "properties": {
            "path": {
              "type": "string",
              "description": "Path to the CSV file"
            }
          },
        "required": ["path"]
        }
    }
  },
  {
    "type": "function",
    "function": {
        "name": "list_columns",
        "description": "List columns in the loaded CSV.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
  },
  {
    "type": "function",
    "function": {
        "name": "summarize_column",
        "description": "Return summary statistics for a column.",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "Column name to summarize"
                }
            },
            "required": ["column"]
        }
    }
  },
  {
    "type": "function",
    "function": {
        "name": "compute_correlation",
        "description": "Compute Pearson correlation between two numeric columns.",
        "parameters": {
            "type": "object",
            "properties": {
                "col1": {
                    "type": "string",
                    "description": "First numeric column"
                },
                "col2": {
                    "type": "string",
                    "description": "Second numeric column"
                }
            },
            "required": ["col1", "col2"]
        }
    }
  }
]


node_tools = {
  "load_csv": csv_manager.load_csv,
  "list_columns": csv_manager.list_columns,
  "summarize_column": csv_manager.summarize_column,
  "compute_correlation": csv_manager.compute_correlation
}


SYSTEM_PROMPT = """
You are a data analysis agent.
You can load CSV files, inspect columns, summarize columns, and compute correlations.
Use tools when needed.
Be concise and explain the result in plain English.
"""


def run_agent_cycle(messages, user_input, max_tool_rounds=5):
  messages.append({"role": "user", "content": user_input})

  for _ in range(max_tool_rounds):
    response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=messages,
      tools=tools_schema,
      tool_choice="auto"
    )

    assistant_message = response.choices[0].message
    messages.append(assistant_message.model_dump())

    if not assistant_message.tool_calls:
      return assistant_message.content

    for tool_call in assistant_message.tool_calls:
      tool_name = tool_call.function.name
      arguments = json.loads(tool_call.function.arguments)

      if tool_name in node_tools:
        tool_result = node_tools[tool_name](**arguments)
      else:
        tool_result = {"error": f"Unknown tool: {tool_name}"}

      messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(tool_result)
      })

  return "Agent stopped because it reached the tool-round limit."


# Q5

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

result = run_agent_cycle(
  messages,
  "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
)

print("\nQ5:")
print(result)


# Q6

# In the ReAct loop:
# system gives the agent rules and behavior.
# user gives the task.
# assistant decides what to do and may request tool calls.
# tool contains the result returned by a Python function.

print("\nQ6 full messages:")
print(json.dumps(messages, indent=2, default=str))

# --- Lesson 04: smolagents ---

# Q7

@tool
def compute_correlation_tool(col1: str, col2: str) -> dict:
  """Compute the Pearson correlation between two numeric columns.

  Args:
      col1: The name of the first numeric column.
      col2: The name of the second numeric column.

  Returns:
      A dictionary with the column names, Pearson correlation coefficient, and p-value.
  """
  return csv_manager.compute_correlation(col1, col2)


print("\nQ7:")
print(compute_correlation_tool.description)

# smolagents automatically generated a tool description from the function name,
# parameters, type hints, and docstring.

# In Q4, I manually wrote the JSON schema myself.
# With smolagents, the framework handles most of that automatically.

# To generate a good description, smolagents still depends on clear function names,
# parameter names, type hints, and a useful docstring written by the developer.


# Q8

model = OpenAIServerModel(
    model_id="gpt-4o-mini"
)

TOOLS = [
    compute_correlation_tool
]

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    additional_authorized_imports=["pandas", "matplotlib.pyplot"]
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

print("\nQ8:")

response_tool = tool_agent.run(prompt)
print("\nToolCallingAgent response:")
print(response_tool)

response_code = code_agent.run(
    prompt,
    additional_args={"csv_manager": csv_manager}
)
print("\nCodeAgent response:")
print(response_code)

# The ToolCallingAgent can only use the tools that were provided.
# Since I did not provide a plotting tool, it may not be able to create the scatter plot or set green dots.
# The CodeAgent is more flexible because it can write Python code and use matplotlib.
# This shows that CodeAgent is better for open-ended coding tasks, while ToolCallingAgent is better for controlled tasks with clear tools.


# The ToolCallingAgent could not create the plot because it only had access
# to the provided tools and no plotting tool existed.

# The CodeAgent was more flexible and attempted to write Python code,
# but it struggled because the CsvManager object did not expose the raw DataFrame directly.

# This demonstrates that CodeAgent can attempt more complex tasks,
# but it can also hallucinate methods or misunderstand object structures.


# Q9

# ToolCallingAgent is safer and more predictable because it can only use approved tools.
# It is better for controlled workflows and production systems.

# CodeAgent is more flexible because it can generate and execute Python code.
# It is useful for open-ended analysis and experimentation.

# However, CodeAgent is riskier because it can hallucinate functions,
# misunderstand data structures, or generate invalid code.