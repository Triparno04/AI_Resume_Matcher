import pandas as pd

# Load both datasets
df1 = pd.read_csv("data/diverse_resume_job_dataset.csv")
df2 = pd.read_csv("data/diverse_resume_job_dataset_v2.csv")

# Concatenate them
df_combined = pd.concat([df1, df2], ignore_index=True)

# Drop any exact duplicate rows (if resume, job, and label are the same)
df_combined = df_combined.drop_duplicates()

# Save the merged dataset
df_combined.to_csv("data/combined_resume_job_dataset.csv", index=False)

print("✅ Merged dataset saved as: data/combined_resume_job_dataset.csv")
print("✅ Total rows after merging:", len(df_combined))
