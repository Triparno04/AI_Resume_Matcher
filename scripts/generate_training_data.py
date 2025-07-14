import random
import pandas as pd

# Sample resume templates
resume_templates = {
    "software_engineer": [
        "Experienced software engineer skilled in Python, Java, and cloud computing.",
        "Backend developer with Node.js, Express, and MongoDB experience.",
        "Full-stack developer with React, Flask, and PostgreSQL expertise."
    ],
    "data_scientist": [
        "Data scientist skilled in machine learning, Python, and data visualization.",
        "ML engineer with TensorFlow, Keras, and large-scale data experience.",
        "Analytics professional with SQL, pandas, and predictive modeling."
    ],
    "graphic_designer": [
        "Graphic designer with skills in Photoshop, Illustrator, and UI/UX design.",
        "Designer experienced in branding, typography, and mobile app design.",
        "UX/UI expert with knowledge in Figma and user-centered design."
    ],
    "product_manager": [
        "Product manager experienced in agile, JIRA, and stakeholder communication.",
        "PM with roadmap planning and user story writing experience.",
        "Product owner skilled in cross-functional team leadership."
    ]
}

# Sample job descriptions and labels
job_templates = {
    "software_engineer": [
        ("Hiring backend engineer with Python and AWS experience.", "fit"),
        ("Looking for Java developer for microservices and DevOps.", "fit"),
        ("Looking for frontend dev with React and HTML/CSS.", "average"),
        ("Hiring sales rep with CRM and outreach experience.", "not")
    ],
    "data_scientist": [
        ("Looking for ML engineer with Scikit-learn and Python.", "fit"),
        ("Data analyst needed with Excel and BI tools.", "average"),
        ("Hiring graphic designer for branding and visual identity.", "not")
    ],
    "graphic_designer": [
        ("Hiring UI/UX designer skilled in Figma and prototyping.", "fit"),
        ("Looking for creative with Adobe XD and branding skills.", "fit"),
        ("Software engineer with Python and Flask needed.", "not")
    ],
    "product_manager": [
        ("Hiring PM with agile, JIRA, and stakeholder coordination.", "fit"),
        ("Looking for project coordinator for documentation work.", "average"),
        ("ML research engineer needed for NLP project.", "not")
    ]
}

# Generate 1000+ examples
records = []
for _ in range(1050):
    role = random.choice(list(resume_templates.keys()))
    resume = random.choice(resume_templates[role])
    job, label = random.choice(job_templates[role])
    records.append({
        "resume_text": resume,
        "job_description": job,
        "label": label
    })

# Save to CSV
df = pd.DataFrame(records)
df.to_csv("data/diverse_resume_job_dataset.csv", index=False)
print("✅ Data saved to data/diverse_resume_job_dataset.csv")
