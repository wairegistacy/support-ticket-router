import os
import pandas as pd
from datasets import load_dataset

from src.preprocessing import normalize

# Map dataset queues to our portfolio categories
QUEUE_TO_CATEGORY = {
    "Technical Support": "Technical Issue",
    "IT Support": "Technical Issue",
    "Service Outages and Maintenance": "Technical Issue",
    "Product Support": "Campaign Setup",
    "Customer Service": "Account & Billing",
    "Billing and Payments": "Account & Billing",
    "General Inquiry": "General Inquiry",
}
# Map dataset priority to our P0–P3
PRIORITY_TO_P = {
    "critical": "P0",
    "high": "P1",
    "medium": "P2",
    "low": "P3",
}

OUT_PATH = "data/tickets.csv"

def build_text(row: dict) -> str:
    subj = str(row.get("subject") or "").strip()
    body = str(row.get("body") or "").strip()
    text = (subj + "\n\n" + body).strip()
    return normalize(text)

def main():
    os.makedirs("data", exist_ok=True)

    # Downloads from Hugging Face the first time you run it
    ds = load_dataset("Tobi-Bueck/customer-support-tickets")

    # Some datasets have only "train"; if there are multiple splits, we concatenate.
    frames = []
    for split in ds.keys():
        frames.append(ds[split].to_pandas())
    df = pd.concat(frames, ignore_index=True)

    # Keep English tickets first (simpler); we can add multilingual later.
    if "language" in df.columns:
        df = df[df["language"].astype(str).str.lower().isin(["en", "english"])].copy()

    # Build text
    df["text"] = df.apply(lambda r: build_text(r.to_dict()), axis=1)

    # Basic fields
    df["orig_queue"] = df.get("queue", "General Inquiry").astype(str)
    df["orig_priority"] = df.get("priority", "medium").astype(str).str.lower()

    # Map to our schema
    df["category"] = df["orig_queue"].map(QUEUE_TO_CATEGORY).fillna("Other")
    df["priority"] = df["orig_priority"].map(PRIORITY_TO_P).fillna("P2")
    df["queue"] = df["category"].map({
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
    }).fillna("General Support")

    # Drop empty texts
    df = df[df["text"].str.len() >= 20].copy()

    # Keep only what we need
    out = df[["text", "category", "priority", "queue", "orig_queue", "orig_priority"]].copy()

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out):,} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()
