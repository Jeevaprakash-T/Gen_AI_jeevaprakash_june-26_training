"""
MODULE 2 — Semantic Job Search.
Builds a FAISS index over the job corpus and exposes a top-N similarity
search used both by the matching feature and the RAG mentor's job lookups.
"""
import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import pandas as pd

from src.config import JOBS_CSV, JOBS_INDEX_PATH, TOP_N_JOBS
from src.search.embed import embed_texts, embed_text


def _job_to_text(row: pd.Series) -> str:
    return f"{row.get('title','')}. Skills: {row.get('skills','')}. {row.get('description','')}"


def build_job_index(jobs_csv: Path = JOBS_CSV, out_path: Path = JOBS_INDEX_PATH) -> None:
    """
    MODULE 2 build step: embed every job description, store vectors in FAISS,
    and persist the row metadata alongside it so search results are human-readable.
    """
    df = pd.read_csv(jobs_csv)
    texts = [_job_to_text(row) for _, row in df.iterrows()]
    vectors = embed_texts(texts)

    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine since vectors are normalized
    index.add(vectors)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path) + ".index")
    df.to_json(str(out_path) + ".meta.json", orient="records")
    print(f"Indexed {len(df)} jobs -> {out_path}.index")


def load_job_index(path: Path = JOBS_INDEX_PATH):
    index = faiss.read_index(str(path) + ".index")
    with open(str(path) + ".meta.json") as f:
        meta = json.load(f)
    return index, meta


def search_jobs(query_text: str, top_n: int = TOP_N_JOBS, index_path: Path = JOBS_INDEX_PATH) -> List[Dict]:
    """Embed the query (usually a candidate profile) and return top-N matching jobs with scores."""
    index, meta = load_job_index(index_path)
    query_vec = embed_text(query_text).reshape(1, -1)
    scores, ids = index.search(query_vec, top_n)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        job = dict(meta[idx])
        job["match_score"] = float(score)
        results.append(job)
    return results
