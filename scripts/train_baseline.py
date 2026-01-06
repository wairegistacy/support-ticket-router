import os
import pandas as pd

from src.preprocessing import normalize
from src.model import train_tfidf_logreg, save_bundle
from src.config import CATEGORIES

MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "category_baseline.joblib")

def main():
    df = pd.read_csv("data/tickets.csv")
    df["text"] = df["text"].astype(str).apply(normalize)
    df["category"] = df["category"].astype(str)

    bundle = train_tfidf_logreg(
        texts=df["text"].tolist(),
        y=df["category"].tolist(),
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_bundle(bundle, MODEL_PATH)
    print(f"Saved baseline model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
