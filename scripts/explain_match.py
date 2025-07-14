import os
from dotenv import load_dotenv
import openai

# ✅ Load from .env file
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.base_url = os.getenv("GROQ_API_BASE_URL")


def explain_match(resume, job, job_description="", job_skills=""):
    try:
        response = openai.chat.completions.create(
            model="llama3-8b-8192",  # ✅ Updated model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains why a given job is a good match for a resume."},
                {"role": "user", "content": f"""Given this resume:

{resume}

And this job:

Title: {job['title']}
Company: {job['company']}
Location: {job['location']}
Work Type: {job['work_type']}
Experience: {job['experience']}
Description: {job_description}
Skills: {job_skills}

Explain why this job is a good match for the resume in simple points.
"""}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {e}"
