import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import PyPDF2
import joblib
import plotly.graph_objects as go

from scripts.resume_matcher import match_resume_to_jobs, is_indian_location
from scripts.explain_match import explain_match
from scripts.resume_analyzer_ui import render_resume_analyzer_tab

# Load model + vectorizer
try:
    clf = joblib.load(os.path.join("scripts", "resume_classifier_model.pkl"))
    vectorizer = joblib.load(os.path.join("scripts", "resume_vectorizer.pkl"))
except Exception as e:
    clf, vectorizer = None, None
    st.error(f"Could not load model/vectorizer: {e}")

st.set_page_config(page_title="AI Resume Matcher", layout="centered")
st.title("AI Resume Matcher")
st.markdown("Upload your resume to find best job matches and get personalized feedback using Gen AI.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
location_filter = st.text_input("Location Filter")
work_type_filter = st.selectbox("Work Type", ["All", "Full-time", "Part-time", "Remote"])
min_salary = st.number_input("Minimum Expected Salary (optional)", min_value=0, step=1000)
only_indian_jobs = st.toggle("Show Only Indian Jobs", value=False)

def draw_charts(job, resume_text):
    skill_score = round(len(job["skills_matched"]) / (len(job["skills_matched"]) + len(job["skills_missing"]) + 1e-5), 2)
    location_match = 1.0 if "india" in job["location"].lower() else 0.5
    work_type_match = 1.0 if job["work_type"].lower() == "full-time" else 0.6
    exp = job.get("experience", "")
    exp_str = str(exp).lower() if isinstance(exp, str) else ""
    experience_match = 1.0 if exp_str in resume_text.lower() else 0.4
    faiss_score = round(job["score"], 2)

    labels = ["Skill Match", "Location Match", "Work Type", "Experience", "FAISS Score"]
    values = [skill_score, location_match, work_type_match, experience_match, faiss_score]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        line=dict(color='#1f77b4'),
        name="Radar Chart"
    ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=360,
        margin=dict(t=20, b=20, l=20, r=20)
    )

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color='#1f77b4',
        name="Bar Chart"
    ))
    bar_fig.update_layout(
        xaxis=dict(range=[0, 1]),
        height=360,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False
    )

    return radar_fig, bar_fig

if resume_file:
    reader = PyPDF2.PdfReader(resume_file)
    resume_text = "".join([page.extract_text() for page in reader.pages])

    tab1, tab2 = st.tabs(["Job Matches", "Resume Analyzer"])

    with tab1:
        if st.button("Match Jobs"):
            status = st.empty()
            status.info("Matching resume to jobs...")

            with st.spinner("Processing..."):
                matches, logs = match_resume_to_jobs(
                    resume_text,
                    top_k=1000,
                    return_logs=True
                )

            with st.expander("Debug Logs"):
                for line in logs:
                    st.text(line)

            st.subheader("Top Matches")

            if not matches:
                st.warning("No matches found.")
            else:
                matches.sort(key=lambda x: x["score"], reverse=True)
                filtered_matches = []

                for job in matches:
                    loc = job.get("location", "")
                    if location_filter and location_filter.lower() not in loc.lower():
                        continue
                    if only_indian_jobs and not is_indian_location(loc):
                        continue
                    if work_type_filter != "All" and job["work_type"] != work_type_filter:
                        continue
                    filtered_matches.append(job)

                filtered_matches = filtered_matches[:20]

                if not filtered_matches:
                    st.warning("Matches found but none passed your filters.")
                else:
                    for i, job in enumerate(filtered_matches):
                        matched = ", ".join(job["skills_matched"]) if job["skills_matched"] else "None"
                        missing = ", ".join(job["skills_missing"]) if job["skills_missing"] else "None"

                        # Classifier prediction
                        if clf and vectorizer:
                            combo_text = resume_text + " " + str(job.get("title", "")) + " " + str(job.get("description", ""))
                            probs = clf.predict_proba(vectorizer.transform([combo_text]))[0]
                            labels = clf.classes_
                            label_probs = dict(zip(labels, probs))

                            if label_probs.get("fit", 0) > 0.6:
                                fit_label = "🟢Strong Fit"
                            elif label_probs.get("not", 0) > 0.6:
                                fit_label = "🔴Weak Fit"
                            else:
                                fit_label = "🟡Average Fit"

                            confidence = max(label_probs.values())
                            fit_label += f" ({int(confidence * 100)}%)"

                            # Confidence breakdown string
                            label_probs_str = " | ".join([
                                f"{label.capitalize()}: {round(prob * 100, 1)}%"
                                for label, prob in label_probs.items()
                            ])

                            # Skill match %
                            skill_match_percent = round(
                                len(job["skills_matched"]) /
                                (len(job["skills_matched"]) + len(job["skills_missing"]) + 1e-5) * 100, 1
                            )
                        else:
                            fit_label = "Unknown"
                            label_probs_str = "N/A"
                            skill_match_percent = 0

                        # 1. Full-width Job Info
                        st.markdown(f"""
                            <div style="padding: 14px 18px; border: 1px solid #ddd; border-radius: 12px; margin-bottom: 10px;">
                                <h4 style="margin-bottom: 6px;">{i+1}. {job['title']}</h4>
                                <p style="margin: 0;"><strong>Company:</strong> {job['company']}</p>
                                <p style="margin: 0;"><strong>Location:</strong> {job['location']}</p>
                                <p style="margin: 0;"><strong>Work Type:</strong> {job['work_type']}</p>
                                <p style="margin: 0;"><strong>Experience:</strong> {job['experience'] or "Not specified"}</p>
                                <p style="margin: 0;"><strong>Match Score:</strong> {job['score']}</p>
                                <p style="margin-top: 10px;"><strong>✅ Skills Matched:</strong> {matched}</p>
                                <p style="margin-top: 4px;"><strong>❌ Skills Missing:</strong> {missing}</p>
                            </div>
                        """, unsafe_allow_html=True)

                        # 2. Charts side-by-side
                        radar, bar = draw_charts(job, resume_text)
                        chart1, chart2 = st.columns(2)
                        with chart1:
                            st.markdown("**Radar Chart:**")
                            st.plotly_chart(radar, use_container_width=True, key=f"radar_{i}")
                        with chart2:
                            st.markdown("**Bar Chart:**")
                            st.plotly_chart(bar, use_container_width=True, key=f"bar_{i}")

                        # 3. Classifier Fit
                        st.markdown(f"""
                            <div style="padding: 10px 14px; border: 1px solid #ccc;
                                        border-radius: 8px; font-size: 15px; margin-top: 10px;
                                        margin-bottom: 20px;">
                                <strong>Classifier Fit:</strong> {fit_label}<br>
                                <strong>Classifier Confidence:</strong> {label_probs_str}<br>
                                <strong>FAISS Similarity Score:</strong> {round(job['score'], 3)}<br>
                            </div>
                        """, unsafe_allow_html=True)

                        # 4. Match Explanation
                        explanation = explain_match(
                            resume_text,
                            job,
                            job_description="",
                            job_skills=", ".join(job.get("skills_matched", []) + job.get("skills_missing", []))
                        )
                        st.markdown(f"""
                            <div style="padding: 14px 18px; border: 1px solid #ccc; border-radius: 10px; margin-bottom: 40px;">
                                <pre style="color: #444; font-size: 15px; white-space: pre-wrap; word-wrap: break-word;">{explanation}</pre>
                            </div>
                        """, unsafe_allow_html=True)

    # ---------------- TAB 2: Resume Analyzer ------------------
    with tab2:
        render_resume_analyzer_tab(resume_text)
