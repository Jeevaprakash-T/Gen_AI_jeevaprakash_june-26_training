"""
MODULE 3 — CV Improvement Generator.
Given a resume and a target job description, produces missing-skills,
weak-bullet feedback and a rewritten summary as structured JSON.
"""
import json
import re

import anthropic

from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from src.generate.prompts import CV_SUGGESTIONS_PROMPT_V2

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _extract_json(raw: str) -> dict:
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def generate_cv_suggestions(resume_text: str, job_description: str) -> dict:
    prompt = CV_SUGGESTIONS_PROMPT_V2.format(
        resume_text=resume_text, job_description=job_description
    )
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text
    return _extract_json(raw)
