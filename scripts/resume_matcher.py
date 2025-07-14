import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import re
import os
import joblib

DATA_DIR = "C:/Users/Triparno/Downloads/VS Code/ai_resume_matcher/data/"

# ✅ Load classifier model
try:
    clf = joblib.load(os.path.join("scripts", "resume_classifier_model.pkl"))
    vectorizer = joblib.load(os.path.join("scripts", "resume_vectorizer.pkl"))
except:
    clf, vectorizer = None, None

def load_skill_mapping():
    mapping_df = pd.read_csv(os.path.join(DATA_DIR, "skill_mapping.csv"))
    return dict(zip(mapping_df["skill_abr"], mapping_df["skill_name"]))

def load_job_skills():
    df = pd.read_csv(os.path.join(DATA_DIR, "job_skills.csv"))
    return df.groupby("job_id")["skill_abr"].apply(set).to_dict()

def extract_keywords(text):
    return set(re.findall(r'\b\w{3,}\b', str(text).lower())) if text else set()

def load_faiss_index(embedding_file=os.path.join(DATA_DIR, "job_embeddings.npy"), logs=None):
    embeddings = np.load(embedding_file)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    if logs is not None:
        logs.append("📦 Loaded job embeddings and built FAISS index.")
    return index, embeddings

indian_keywords = [
    "india", "kolkata", "mumbai", "delhi", "new delhi", "gurgaon", "hyderabad", "chennai", "bangalore", "bengaluru",
    "pune", "ahmedabad", "nagpur", "lucknow", "surat", "varanasi", "patna", "kanpur", "faridabad", "bhopal", "agra",
    "meerut", "visakhapatnam", "vadodara", "rajkot", "ludhiana"
]

def is_indian_location(loc):
    loc = str(loc).lower().strip()
    loc_clean = re.sub(r"[^\w\s]", "", loc)
    loc_clean = re.sub(r"\s+", " ", loc_clean)
    tokens = loc_clean.split()
    if loc_clean in indian_keywords:
        return True
    if len(tokens) == 1 and tokens[0] in indian_keywords:
        return True
    elif len(tokens) == 2 and all(tok in indian_keywords for tok in tokens):
        return True
    return False

def match_resume_to_jobs(resume_text, top_k=5, return_logs=False):
    logs = [] if return_logs else None
    if not resume_text.strip():
        if logs: logs.append("❌ Empty resume text.")
        return ([], logs) if return_logs else []

    try:
        jobs_df = pd.read_csv(os.path.join(DATA_DIR, "job_metadata.csv"))
        jobs_df["location"] = jobs_df["location"].astype(str).str.strip()
    except Exception as e:
        if logs: logs.append(f"❌ Metadata load failed: {e}")
        return ([], logs) if return_logs else []

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        resume_embedding = model.encode([resume_text])
    except Exception as e:
        if logs: logs.append(f"❌ Embedding error: {e}")
        return ([], logs) if return_logs else []

    try:
        index, _ = load_faiss_index(logs=logs)
        distances, indices = index.search(resume_embedding, top_k)
    except Exception as e:
        if logs: logs.append(f"❌ FAISS search failed: {e}")
        return ([], logs) if return_logs else []

    job_skills_map = load_job_skills()
    skill_names_map = load_skill_mapping()
    resume_text_lower = resume_text.lower()

    results = []
    for i in indices[0]:
        if i >= len(jobs_df): continue
        job = jobs_df.iloc[i]
        job_id = job.get("job_id")

        matched_skills, missing_skills = [], []
        if job_id in job_skills_map:
            for abr in job_skills_map[job_id]:
                skill = skill_names_map.get(abr, abr)
                (matched_skills if skill.lower() in resume_text_lower else missing_skills).append(skill)
        else:
            for kw in extract_keywords(job.get("title", "")):
                if kw in resume_text_lower:
                    matched_skills.append(kw)

        location = str(job.get("location", "")).strip()
        is_indian = is_indian_location(location)
        base_score = float(1 - distances[0][list(indices[0]).index(i)])
        boost = (base_score * 0.15) + 0.05 if is_indian else 0
        final_score = round(min(base_score + boost, 1.0), 4)

        # ✅ Predict Fit using classifier (convert safely to string)
        job_title = str(job.get("title", "") or "")
        job_desc = str(job.get("description", "") or "")
        combo_text = resume_text + " " + job_title + " " + job_desc
        pred = "unknown"
        if clf and vectorizer:
            try:
                pred = clf.predict(vectorizer.transform([combo_text]))[0]
            except:
                pred = "unknown"

        results.append({
            "title": job_title,
            "company": str(job.get("company_name", "")),
            "location": location,
            "work_type": str(job.get("formatted_work_type", "")),
            "experience": str(job.get("formatted_experience_level", "")),
            "score": final_score,
            "skills_matched": matched_skills,
            "skills_missing": missing_skills,
            "fit_label": pred
        })

        if logs: logs.append(f"✅ Match: {job_title} | Score: {final_score} | Fit: {pred}")

    return (results, logs) if return_logs else results
