import os
from dotenv import load_dotenv
import openai

# ✅ Load from .env file
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.base_url = os.getenv("GROQ_API_BASE_URL")


def analyze_resume_with_llm(resume_text):
    try:
        response = openai.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful career coach reviewing resumes."},
                {"role": "user", "content": f"""
Review the following resume and provide honest suggestions to improve it. Include anything missing like skills, projects, or structure.

Resume:
\"\"\"
{resume_text}
\"\"\"
"""}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {e}"
