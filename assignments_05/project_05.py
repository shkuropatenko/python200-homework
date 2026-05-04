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

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.
The paragraph should be 3-5 sentences: confident, specific, and not generic.

Here are two examples of the style and tone:

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I've spent my career making decisions under pressure using incomplete information, which turns out to be excellent training for data analysis. I recently completed a data analytics program where I built dashboards tracking patient outcomes across departments. I'm excited to bring that combination of clinical context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: I spent a decade on the operations side of banking, watching technology decisions get made by people who had never processed a wire transfer or resolved a failed ACH batch. That frustration turned into curiosity, and two years of self-teaching Python later, I'm ready to be on the other side of those decisions. I'm applying to [Company] because your work on payment infrastructure is exactly where my domain expertise and new technical skills intersect.

Now write an opening paragraph for this person:
Role: {job_title}
Background: {background}
Opening:
"""

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages)

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed a Python course and built data pipelines using Prefect and Pandas."

cover_letter = generate_cover_letter(job_title, background)

print("\nCover Letter Opening:")
print(cover_letter)

# I chose these examples because they show career changers connecting old experience to a new technical role.
# Few-shot prompting helps control the tone and makes the output less generic.


def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print("Input was flagged. Please rephrase.")
        return False

    return True

print("\nModeration test:")
print(is_safe("Hello, how are you?"))  # should be True
print(is_safe("I want to harm someone"))  # might be flagged

# The moderation check helps filter unsafe input before sending it to the model.