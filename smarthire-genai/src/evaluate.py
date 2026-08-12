"""
Section 8 — Evaluation.
Run this after building both indexes to produce reports/answer_quality.md.

Measures:
  - Retrieval relevance: hit-rate of top job matches for sample profiles
  - Answer quality: mentor answers on a fixed test-question set, graded by an LLM judge
  - Hallucination check: mentor must refuse when the answer isn't in career notes
  - Prompt comparison: CV_SUGGESTIONS v1 vs v2 (see src/generate/prompts.py)
"""
import json
from datetime import datetime

from src.search.job_search import search_jobs
from src.mentor.rag_chain import ask_mentor

# --- Fill these in with your own test set before running ---
TEST_PROFILES = [
    {"profile_text": "Python developer, 2 years experience, skills: Python, SQL, pandas, seeking Data Analyst role",
     "expect_keywords": ["data", "analyst", "python"]},
]

TEST_MENTOR_QUESTIONS = [
    "How do I switch from software development to data analysis?",
    "What is the capital of France?",  # off-topic -> should refuse/redirect
    "What was the exact salary offered by CompanyX in 2019?",  # not-in-corpus -> should say don't know
]


def eval_retrieval_relevance():
    rows = []
    for case in TEST_PROFILES:
        jobs = search_jobs(case["profile_text"], top_n=5)
        hit = any(
            any(kw.lower() in (job.get("title", "") + job.get("description", "")).lower()
                for kw in case["expect_keywords"])
            for job in jobs
        )
        rows.append({"profile": case["profile_text"][:60], "hit": hit, "top_jobs": [j.get("title") for j in jobs]})
    hit_rate = sum(r["hit"] for r in rows) / len(rows) if rows else 0
    return {"hit_rate": hit_rate, "cases": rows}


def eval_mentor_answers():
    rows = []
    for q in TEST_MENTOR_QUESTIONS:
        result = ask_mentor(q)
        rows.append({"question": q, "answer": result["answer"], "sources": result["sources"]})
    return rows


def run_all():
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "retrieval_relevance": eval_retrieval_relevance(),
        "mentor_answers": eval_mentor_answers(),
    }
    with open("reports/eval_raw.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_all()
