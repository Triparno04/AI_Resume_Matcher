import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

def generate_embeddings(input_csv, output_dir):
    # ✅ Load data
    df = pd.read_csv(input_csv)

    # ✅ Ensure required columns exist
    for col in ["title", "description", "skills_desc", "company_name", "location"]:
        if col not in df.columns:
            df[col] = ""

    df = df.fillna("")

    # ✅ Create combined text field for embedding
    df["combined_text"] = (
        df["title"] + " " +
        df["description"] + " " +
        df["skills_desc"] + " " +
        df["company_name"] + " " +
        df["location"]
    )

    df = df[df["combined_text"].str.strip() != ""]

    # ✅ Generate embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("🔄 Generating embeddings...")
    embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)

    # ✅ Save embeddings and updated metadata
    np.save(os.path.join(output_dir, "job_embeddings.npy"), embeddings)
    df.to_csv(os.path.join(output_dir, "job_metadata.csv"), index=False)
    print("✅ Done! Metadata and embeddings saved.")

# ✅ Run script directly
if __name__ == "__main__":
    input_csv = r"C:\Users\Triparno\Downloads\VS Code\ai_resume_matcher\data\job_metadata.csv"
    output_dir = r"C:\Users\Triparno\Downloads\VS Code\ai_resume_matcher\data"
    generate_embeddings(input_csv=input_csv, output_dir=output_dir)
