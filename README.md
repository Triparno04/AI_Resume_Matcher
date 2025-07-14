# AI Resume Matcher

AI Resume Matcher is a GenAI + ML-powered web application built using Python and Streamlit. It intelligently matches candidate resumes to relevant job listings by leveraging semantic similarity (FAISS), skill overlap, and a trained resume-job classifier. Additionally, it offers AI-based feedback to help improve resumes.

## Features

- ✅ Resume parsing from PDF
- ✅ Intelligent job matching using:
  - FAISS (vector similarity)
  - Skill match ratio
  - Trained classifier (fit / average / not-fit)
- ✅ Match explanation using LLM (via Groq API)
- ✅ Radar and bar charts for visualizing fit
- ✅ AI Resume Analyzer with actionable improvement suggestions
- ✅ Filtering by location, work type, salary, and region (India/global)

## Tech Stack

- Python (Backend logic, ML integration)
- Streamlit (Web UI)
- FAISS (Semantic similarity search)
- Scikit-learn (Classifier model)
- HuggingFace Transformers (Embeddings)
- Groq + LLM API (Resume feedback and match explanation)
- Plotly (Radar + bar charts)
- PyPDF2 (Resume parsing)

## Folder Structure

ai_resume_matcher/
│
├── app/
│ └── main_ui.py # Streamlit app UI
│
├── data/ # Job/resume datasets, embeddings (not tracked in Git)
│ ├── job_metadata.csv
│ ├── job_embeddings.npy
│ └── ...
│
├── scripts/ # All backend scripts
│ ├── embed_jobs.py
│ ├── parse_jobs.py
│ ├── parse_resume.py
│ ├── resume_matcher.py
│ ├── resume_analyzer_ui.py
│ ├── explain_match.py
│ └── ...
│
├── .env # API keys (excluded via .gitignore)
├── .gitignore
├── requirements.txt
└── README.md


## Setup Instructions

1. Clone the repository:

git clone https://github.com/Triparno04/AI_Resume_Matcher.git
cd AI_Resume_Matcher

2. Create and activate a virtual environment:
   
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

3. Install dependencies:

cd "C:\Users\Triparno\Downloads\VS Code\ai_resume_matcher"

4. Set up your environment variables in a .env file:

OPENAI_API_KEY=your-groq-key
OPENAI_BASE_URL=https://api.groq.com/openai/v1/

5. Run the application:

streamlit run app/main_ui.py
