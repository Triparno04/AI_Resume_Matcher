from scripts.embed_jobs import generate_embeddings
import pandas as pd
from scripts.resume_matcher import match_resume_to_jobs

DATA_PATH = "C:/Users/Triparno/Downloads/VS Code/ai_resume_matcher/data"

# Optional: Uncomment this if you want to re-embed after CSV changes
# generate_embeddings(
#     input_csv=f"{DATA_PATH}/job_metadata.csv",
#     output_dir=DATA_PATH
# )

# Preview data
df = pd.read_csv(f"{DATA_PATH}/job_metadata.csv", low_memory=False)
print("📄 Columns in job_metadata.csv:\n", df.columns.tolist())
print("\n🔍 Sample Job Posting:")
print(df[["job_id", "title", "location"]].tail(10))

# Match a sample resume
resume_input = """
Highly motivated Computer Science student with experience in Python, Streamlit, and backend APIs.
Built ML-based salary predictor using Scikit-learn. Familiar with LLMs, prompt engineering, and Gen AI tools.
Interned as full stack developer with exposure to cloud (AWS), version control, and agile workflows.
"""

print("\n🧠 Matching Resume to Jobs...\n")
matches = match_resume_to_jobs(resume_input)

for idx, job in enumerate(matches, 1):
    print(f"{idx}. {job['title']} @ {job['company']} ({job['location']})")
    print(f"   Type: {job['work_type']} | Exp: {job['experience']} | Similarity: {job['score']}\n")
