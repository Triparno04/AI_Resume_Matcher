import streamlit as st
from scripts.analyze_resume import analyze_resume_with_llm
import re

def render_resume_analyzer_tab(resume_text):
    st.info("Get AI-generated feedback to improve your resume.")

    if st.button("Analyze Resume"):
        with st.spinner("Sending to LLM for analysis..."):
            suggestion = analyze_resume_with_llm(resume_text)

        # ✅ Clean up extra spacing (too many newlines)
        suggestion = re.sub(r"\n{3,}", "\n\n", suggestion.strip())

        st.markdown("**AI Resume Suggestions:**")

        # ✅ Scrollable, styled output box (preserves line breaks without too much spacing)
        st.markdown("""
            <style>
            .suggestion-box {
                border: 1px solid #555;
                border-radius: 10px;
                padding: 16px;
                margin-top: 10px;
                background-color: rgba(0, 0, 0, 0.03);
                color: inherit;
                max-height: 400px;
                overflow-y: auto;
                white-space: pre-wrap;
                font-size: 15px;
            }
            </style>
        """, unsafe_allow_html=True)

        # ✅ Split into chunks if too long (safely render all content)
        max_chunk_length = 3000  # safety for Streamlit's HTML rendering
        chunks = [suggestion[i:i+max_chunk_length] for i in range(0, len(suggestion), max_chunk_length)]
        for chunk in chunks:
            st.markdown(f'<div class="suggestion-box">{chunk}</div>', unsafe_allow_html=True)
