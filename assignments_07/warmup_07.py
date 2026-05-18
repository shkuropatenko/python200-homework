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