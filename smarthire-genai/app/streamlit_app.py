"""
MODULE 6 — Streamlit Portal.
Run with: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # allow `from src...` imports

import streamlit as st

from src.parsing.loader import load_resume
from src.parsing.resume_parser import parse_resume, ResumeParseError
from src.search.job_search import search_jobs
from src.generate.cv_suggestions import generate_cv_suggestions
from src.mentor.rag_chain import ask_mentor

st.set_page_config(page_title="SmartHire GenAI", page_icon="💼", layout="wide")
st.title("💼 SmartHire GenAI — Resume Matching & AI Career Mentor")

tab_upload, tab_mentor = st.tabs(["📄 Upload & Match", "🤖 AI Career Mentor"])

# ---------------------------------------------------------------------
# TAB 1 — Upload, parse, match, suggest
# ---------------------------------------------------------------------
with tab_upload:
    uploaded = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

    if uploaded is not None:
        tmp_path = Path("data/resumes") / uploaded.name
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.getbuffer())

        with st.spinner("Reading and parsing your resume..."):
            try:
                text = load_resume(tmp_path)
                profile = parse_resume(text)
            except ResumeParseError as e:
                st.error(f"Couldn't parse resume: {e}")
                st.stop()

        st.subheader("📋 Parsed Profile")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {profile.name}")
            st.markdown(f"**Email:** {profile.email}")
            st.markdown(f"**Target role:** {profile.target_role}")
        with col2:
            st.markdown("**Skills:**")
            st.write(", ".join(profile.skills) if profile.skills else "—")

        st.subheader("🎯 Matched Jobs")
        with st.spinner("Searching job corpus..."):
            jobs = search_jobs(profile.as_search_text())
        for job in jobs:
            with st.expander(f"{job.get('title', 'Untitled role')} — score {job['match_score']:.2f}"):
                st.write(job.get("description", ""))

        st.subheader("✍️ CV Improvement Suggestions")
        if jobs:
            target_job = jobs[0]
            if st.button("Generate suggestions for top match"):
                with st.spinner("Generating suggestions..."):
                    suggestions = generate_cv_suggestions(text, target_job.get("description", ""))
                st.markdown("**Missing skills:** " + ", ".join(suggestions.get("missing_skills", [])))
                for wb in suggestions.get("weak_bullets", []):
                    st.markdown(f"- *Original:* {wb.get('original')}\n  *Why weak:* {wb.get('why_weak')}")
                st.markdown("**Rewritten summary:**")
                st.info(suggestions.get("rewritten_summary", ""))

# ---------------------------------------------------------------------
# TAB 2 — RAG mentor chat
# ---------------------------------------------------------------------
with tab_mentor:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    question = st.chat_input("Ask the Career Mentor anything about jobs, skills, or careers...")
    if question:
        st.session_state.chat_history.append({"role": "human", "content": question})
        with st.chat_message("human"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_mentor(question, chat_history=st.session_state.chat_history[:-1])
            st.write(result["answer"])
            if result["sources"]:
                st.caption("Sources: " + ", ".join(result["sources"]))

        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
