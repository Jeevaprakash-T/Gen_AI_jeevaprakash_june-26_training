"""
Central configuration for SmartHire GenAI.
All model names, paths and tunable params live here so nothing is hardcoded
deep inside the pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---- API ----
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# ---- Embeddings ----
# Local sentence-transformers model — no API cost, runs offline once downloaded.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
JOBS_CSV = DATA_DIR / "jobs" / "jobs.csv"
CAREER_NOTES_DIR = DATA_DIR / "career_notes"
RESUME_SAMPLES_DIR = DATA_DIR / "resumes"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
JOBS_INDEX_PATH = VECTORSTORE_DIR / "jobs_faiss"
NOTES_INDEX_PATH = VECTORSTORE_DIR / "notes_faiss"

# ---- Search params ----
TOP_N_JOBS = 5
RAG_TOP_K = 4

# ---- Guardrails ----
ALLOWED_TOPICS_HINT = (
    "careers, resumes, job search, skills, interviews, education paths, "
    "salaries, and professional development"
)
