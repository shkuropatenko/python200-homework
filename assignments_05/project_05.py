from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


YOUR_SYSTEM_PROMPT = """
You are a job application coach helping career changers improve their job materials.

Stay focused on resumes, cover letters, interview preparation, and job applications.
Give practical suggestions that are clear and not too long.
Do not invent experience, skills, job titles, or numbers that the user did not provide.
Always remind the user to review and edit your output before submitting it.
You may not know every industry norm, so remind the user to use their own judgment.
"""

# I made the system prompt specific to job applications so the assistant does not drift into unrelated advice.
# I also told it not to invent facts because resume and cover letter writing needs to stay honest.


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.

Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
Use strong action verbs.
Do not invent facts that are not implied by the original.

Return ONLY a valid JSON list.
Each item should have:
- "original"
- "improved"

Bullet points:
```{bullet_text}```
"""

    messages = [{"role": "user", "content": prompt}]
    raw_response = get_completion(messages)

    # Sometimes the model wraps JSON in ```json code blocks, so I clean it before parsing.
    if "```" in raw_response:
        raw_response = raw_response.split("```")[1]
        raw_response = raw_response.replace("json", "").strip()

    try:
        rewritten = json.loads(raw_response)

        for item in rewritten:
            print("\nOriginal:", item["original"])
            print("Improved:", item["improved"])

        return rewritten

    except json.JSONDecodeError:
        print("The response was not valid JSON.")
        print(raw_response)
        return []


bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)

# These bullets are weak because they are too general.
# The model tried to make them more specific and action-oriented.