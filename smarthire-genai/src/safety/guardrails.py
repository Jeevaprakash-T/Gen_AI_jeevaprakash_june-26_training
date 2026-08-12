"""
MODULE 5 — Guardrails Layer.
Two-stage check applied before every LLM call in the mentor flow:
  1. Fast local heuristics (empty input, obvious injection patterns, length caps)
  2. LLM-based topic/safety classifier for anything that passes stage 1
"""
import re

import anthropic

from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ALLOWED_TOPICS_HINT
from src.generate.prompts import GUARDRAIL_PROMPT

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MAX_INPUT_CHARS = 4000

INJECTION_PATTERNS = [
    r"ignore (all|previous|prior) instructions",
    r"you are now",
    r"system prompt",
    r"act as (?!a career)",
    r"disregard your rules",
]


class GuardrailViolation(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def local_checks(message: str) -> None:
    if not message or not message.strip():
        raise GuardrailViolation("Empty message.")
    if len(message) > MAX_INPUT_CHARS:
        raise GuardrailViolation(f"Message too long (max {MAX_INPUT_CHARS} chars).")
    lowered = message.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            raise GuardrailViolation("Message looks like a prompt-injection attempt.")


def llm_topic_check(message: str) -> str:
    """Returns 'SAFE', 'OFF_TOPIC', or 'UNSAFE'."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": GUARDRAIL_PROMPT.format(message=message)}],
    )
    verdict = response.content[0].text.strip().upper()
    if "UNSAFE" in verdict:
        return "UNSAFE"
    if "OFF_TOPIC" in verdict:
        return "OFF_TOPIC"
    return "SAFE"


def check_message(message: str) -> None:
    """
    Raises GuardrailViolation if the message should be blocked.
    Call this before every LLM call in the mentor pipeline.
    """
    local_checks(message)
    verdict = llm_topic_check(message)
    if verdict == "UNSAFE":
        raise GuardrailViolation("This request isn't something I can help with.")
    if verdict == "OFF_TOPIC":
        raise GuardrailViolation(
            f"I'm focused on {ALLOWED_TOPICS_HINT}. Try asking me something in that area!"
        )
