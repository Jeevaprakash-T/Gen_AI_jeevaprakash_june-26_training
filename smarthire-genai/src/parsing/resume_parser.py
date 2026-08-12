"""
MODULE 1 — Resume Parser.
Sends raw resume text to the LLM with a strict JSON-only prompt, then
validates the response against a schema before it's trusted anywhere else
in the pipeline.
"""
import json
import re
from dataclasses import dataclass, field
from typing import List

import anthropic

from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

PARSE_SYSTEM_PROMPT = """You are a resume parsing engine. You extract structured data from raw resume text.

Rules:
- Output ONLY valid JSON. No markdown fences, no preamble, no explanation.
- If a field is not present in the resume, use an empty string or empty list — never invent data.
- "skills" must be a flat list of individual skill strings (split combined skills apart).
- "experience" is a list of objects: {title, company, duration, description}.
- "education" is a list of objects: {degree, institution, year}.

Return exactly this JSON shape:
{
  "name": "",
  "email": "",
  "phone": "",
  "skills": [],
  "experience": [{"title": "", "company": "", "duration": "", "description": ""}],
  "education": [{"degree": "", "institution": "", "year": ""}],
  "target_role": ""
}
"""


class ResumeParseError(Exception):
    pass


@dataclass
class ResumeProfile:
    name: str = ""
    email: str = ""
    phone: str = ""
    skills: List[str] = field(default_factory=list)
    experience: List[dict] = field(default_factory=list)
    education: List[dict] = field(default_factory=list)
    target_role: str = ""

    def as_search_text(self) -> str:
        """Flatten the profile into a single string suitable for embedding."""
        exp = " ".join(
            f"{e.get('title','')} at {e.get('company','')}: {e.get('description','')}"
            for e in self.experience
        )
        return f"{self.target_role}. Skills: {', '.join(self.skills)}. Experience: {exp}"


REQUIRED_KEYS = {"name", "email", "phone", "skills", "experience", "education", "target_role"}


def _extract_json(raw: str) -> dict:
    """Strip accidental markdown fences and parse JSON, with one retry-friendly regex fallback."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def validate_profile(data: dict) -> None:
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ResumeParseError(f"LLM output missing required keys: {missing}")
    if not isinstance(data["skills"], list):
        raise ResumeParseError("`skills` must be a list")
    if not isinstance(data["experience"], list):
        raise ResumeParseError("`experience` must be a list")
    if not isinstance(data["education"], list):
        raise ResumeParseError("`education` must be a list")


def parse_resume(resume_text: str) -> ResumeProfile:
    """Call the LLM once, validate strictly, return a typed ResumeProfile."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1500,
        system=PARSE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Resume text:\n\n{resume_text}"}],
    )
    raw = response.content[0].text
    data = _extract_json(raw)
    validate_profile(data)
    return ResumeProfile(**data)
