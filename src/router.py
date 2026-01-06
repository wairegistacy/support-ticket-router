from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.config import AUTO_ROUTE_MIN_CONFIDENCE, CATEGORY_TO_QUEUE, HIGH_RISK_CATEGORIES
from src.model import TextClassifierBundle
from src.calibration import calibrated_proba_from_bundle
from src.llm_adjudicator import adjudicate_with_llm


@dataclass
class RouteResult:
    category: str
    queue: str
    confidence: float
    needs_human_review: bool
    top3: List[Dict[str, float]]
    priority: str
    priority_reason: str
    llm_used: bool = False
    llm_rationale: Optional[List[str]] = None


def heuristic_priority(text: str, category: str) -> (str, str):
    t = text.lower()

    if any(k in t for k in ["account suspended", "suspended", "charged multiple times", "fraud", "cannot access account"]):
        return "P0", "Contains urgent keywords indicating account/billing risk."

    if any(k in t for k in ["ads not serving", "impressions dropped", "spend dropped", "campaign stopped", "disapproved"]):
        return "P1", "Indicates major delivery/performance interruption."

    if category in {"Account & Billing"}:
        return "P1", "High-risk category; prioritize faster handling."

    return "P2", "Standard issue based on text signals."


def route_ticket(
    text: str,
    category_model: TextClassifierBundle,
    T: Optional[float] = None,
    threshold: Optional[float] = None,
    enable_llm_fallback: bool = False,
) -> RouteResult:
    # 1) Get probabilities (raw or calibrated)
    if T is None:
        proba = category_model.predict_proba([text])[0]
    else:
        proba = calibrated_proba_from_bundle(category_model, [text], T=T)[0]

    # 2) Compute top-k + defaults (ALWAYS set these)
    idx_sorted = np.argsort(-proba)[:3]
    top3_pairs = [(category_model.labels[i], float(proba[i])) for i in idx_sorted]

    category = top3_pairs[0][0]
    conf = top3_pairs[0][1]
    queue = CATEGORY_TO_QUEUE.get(category, "General Support")

    min_conf = threshold if threshold is not None else AUTO_ROUTE_MIN_CONFIDENCE

    # human review logic
    needs_human = conf < min_conf
    if category in HIGH_RISK_CATEGORIES and conf < (min_conf + 0.10):
        needs_human = True

    priority, why = heuristic_priority(text, category)

    llm_used = False
    llm_rationale = None

    # 3) Optional LLM fallback (only if ML says "needs human")
    if enable_llm_fallback and needs_human:
        llm_decision = adjudicate_with_llm(
            ticket_text=text,
            allowed_categories=category_model.labels,
            category_to_queue=CATEGORY_TO_QUEUE,
        )
        if llm_decision is not None:
            llm_used = True
            llm_rationale = llm_decision.rationale

            # Override with validated LLM decision
            category = llm_decision.category
            queue = llm_decision.queue
            priority = llm_decision.priority
            conf = float(llm_decision.confidence)
            needs_human = bool(llm_decision.needs_human_review)

            # Put LLM choice at the top for display
            top3_pairs = [(category, conf)] + top3_pairs[:2]

    # 4) Return result (always defined)
    return RouteResult(
        category=category,
        queue=queue,
        confidence=float(conf),
        needs_human_review=bool(needs_human),
        top3=[{"label": lbl, "prob": float(p)} for lbl, p in top3_pairs],
        priority=priority,
        priority_reason=why,
        llm_used=llm_used,
        llm_rationale=llm_rationale,
    )
