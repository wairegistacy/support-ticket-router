import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class LLMDecision:
    category: str
    priority: str
    queue: str
    confidence: float
    needs_human_review: bool
    rationale: List[str]
    raw_text: str


def _build_prompt(
    ticket_text: str,
    allowed_categories: List[str],
    category_to_queue: Dict[str, str],
) -> str:
    # Keep prompt deterministic and schema-first.
    # IMPORTANT: LLM must output JSON only.
    return f"""
You are a support ticket routing assistant. Your job is to classify and prioritize a ticket.
Return ONLY valid JSON (no markdown, no commentary).

Allowed categories:
{json.dumps(allowed_categories, ensure_ascii=False)}

Queue mapping (category -> queue):
{json.dumps(category_to_queue, ensure_ascii=False)}

Priority definitions:
- P0: account blocked, billing failure/fraud/charge issues, security, urgent access loss
- P1: major product breakage, ads not serving, disapproved impacting business, severe degradation
- P2: standard issue requiring investigation
- P3: how-to / guidance / minor issue

Output JSON schema:
{{
  "category": "<one of Allowed categories>",
  "priority": "P0|P1|P2|P3",
  "queue": "<must match mapping for category>",
  "confidence": <number between 0 and 1>,
  "needs_human_review": <true|false>,
  "rationale": ["<1-2 short bullets>"]
}}

Rules:
- If uncertain, set needs_human_review=true and lower confidence.
- Confidence should reflect certainty about correct routing (not writing quality).
- queue must exactly match the mapping for the selected category.

Ticket:
{ticket_text}
""".strip()


def _safe_json_loads(s: str) -> Optional[dict]:
    # Attempt direct JSON parse; if the model wrapped JSON with text, try to extract.
    try:
        return json.loads(s)
    except Exception:
        # Extract the first {...} block
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                return None
        return None


def _validate_decision(
    obj: Dict[str, Any],
    allowed_categories: List[str],
    category_to_queue: Dict[str, str],
) -> Optional[LLMDecision]:
    required = ["category", "priority", "queue", "confidence", "needs_human_review", "rationale"]
    if not all(k in obj for k in required):
        return None

    cat = str(obj["category"])
    if cat not in allowed_categories:
        return None

    pr = str(obj["priority"])
    if pr not in {"P0", "P1", "P2", "P3"}:
        return None

    expected_queue = category_to_queue.get(cat)
    q = str(obj["queue"])
    if expected_queue is None or q != expected_queue:
        return None

    try:
        conf = float(obj["confidence"])
    except Exception:
        return None
    if not (0.0 <= conf <= 1.0):
        return None

    needs = bool(obj["needs_human_review"])
    rat = obj["rationale"]
    if not isinstance(rat, list) or not all(isinstance(x, str) for x in rat):
        return None

    # Keep rationale short
    rat = [r.strip() for r in rat if r.strip()][:3]

    return LLMDecision(
        category=cat,
        priority=pr,
        queue=q,
        confidence=conf,
        needs_human_review=needs,
        rationale=rat,
        raw_text=json.dumps(obj, ensure_ascii=False),
    )


def call_llm(prompt: str) -> str:
    """
    Provider-agnostic stub.

    For now:
    - If LLM is not configured, raise a clear error.
    - Later you can implement OpenAI or Gemini here.
    """
    raise RuntimeError(
        "LLM not configured. Set up a provider implementation in src/llm_adjudicator.py::call_llm()."
    )


def adjudicate_with_llm(
    ticket_text: str,
    allowed_categories: List[str],
    category_to_queue: Dict[str, str],
) -> Optional[LLMDecision]:
    """
    Returns a validated LLMDecision or None if:
    - provider not configured
    - parsing/validation fails
    """
    prompt = _build_prompt(ticket_text, allowed_categories, category_to_queue)

    try:
        raw = call_llm(prompt)
    except Exception:
        return None

    obj = _safe_json_loads(raw)
    if obj is None:
        return None

    decision = _validate_decision(obj, allowed_categories, category_to_queue)
    if decision is None:
        return None

    decision.raw_text = raw
    return decision
