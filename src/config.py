from dataclasses import dataclass
from typing import Dict, List


CATEGORIES = [
    "Campaign Setup",
    "Technical Issue",
    "Account & Billing",
    "General Inquiry",
]

PRIORITIES: List[str] = ["P0", "P1", "P2", "P3"]

CATEGORY_TO_QUEUE: Dict[str, str] = {
    "Billing & Payments": "Billing Ops",
    "Policy & Disapprovals": "Policy Support",
    "Campaign Setup": "Onboarding Support",
    "Performance": "Optimization Support",
    "Conversion Tracking": "Measurement/Tagging",
    "Account Access": "Account Recovery",
    "Reporting & Analytics": "Analytics Support",
    "Creative & Assets": "Creative Support",
    "Technical Bug": "Tech Escalations",
    "Other": "General Support",
}

# Confidence threshold for auto-routing
AUTO_ROUTE_MIN_CONFIDENCE = 0.70

# High-risk categories: require more caution
HIGH_RISK_CATEGORIES = {"Billing & Payments", "Policy & Disapprovals"}
