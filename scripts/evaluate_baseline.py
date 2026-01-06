import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
)
from sklearn.model_selection import train_test_split

from src.preprocessing import normalize
from src.model import train_tfidf_logreg
from src.config import CATEGORIES, AUTO_ROUTE_MIN_CONFIDENCE


@dataclass
class SplitData:
    X_train: list
    X_val: list
    X_test: list
    y_train: list
    y_val: list
    y_test: list


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "category",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> SplitData:
    # First split off test
    X = df["text"].tolist()
    y = df[label_col].tolist()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Then split train/val from remaining
    # val_size is fraction of the ORIGINAL data, so convert to fraction of trainval
    val_frac_of_trainval = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_frac_of_trainval,
        random_state=seed,
        stratify=y_trainval,
    )

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


def auto_route_metrics(
    y_true: list,
    proba: np.ndarray,
    labels: list,
    min_conf: float,
) -> Tuple[float, float, float]:
    """
    Returns:
      coverage: fraction of tickets auto-routed
      precision_on_covered: precision among covered tickets (overall)
      accuracy_on_covered: accuracy among covered tickets (overall)
    """
    pred_idx = np.argmax(proba, axis=1)
    pred_labels = [labels[i] for i in pred_idx]
    confidences = np.max(proba, axis=1)

    covered_mask = confidences >= min_conf
    if covered_mask.sum() == 0:
        return 0.0, 0.0, 0.0

    y_true_cov = np.array(y_true)[covered_mask]
    y_pred_cov = np.array(pred_labels)[covered_mask]

    # For multi-class, "precision" can be defined in many ways.
    # Here we compute micro-precision on the covered set (same as accuracy when one label per item),
    # but we’ll report both explicitly to avoid confusion in interviews.
    acc_cov = float((y_true_cov == y_pred_cov).mean())

    # micro precision equals accuracy in single-label multi-class.
    prec_cov = acc_cov

    coverage = float(covered_mask.mean())
    return coverage, prec_cov, acc_cov


def save_confusion_matrix_png(cm: np.ndarray, labels: list, out_path: str) -> None:
    # Keep matplotlib optional: only import when needed
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Test)",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    data_path = "data/tickets.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Missing {data_path}. Run: python -m scripts.build_dataset"
        )

    df = pd.read_csv(data_path)
    if "text" not in df.columns or "category" not in df.columns:
        raise ValueError("data/tickets.csv must contain 'text' and 'category' columns.")

    df["text"] = df["text"].astype(str).apply(normalize)
    df["category"] = df["category"].astype(str)

    # Optional: restrict to your category set (drops unexpected labels)
    df = df[df["category"].isin(CATEGORIES)].copy()

    # Drop very small classes (prevents stratify errors)
    min_count = 10
    counts = df["category"].value_counts()
    keep = counts[counts >= min_count].index
    df = df[df["category"].isin(keep)].copy()

    print(f"Rows after filtering: {len(df):,}")
    print("Class distribution (top 20):")
    print(df["category"].value_counts().head(20))

    split = stratified_split(df, label_col="category", test_size=0.15, val_size=0.15, seed=42)

    bundle = train_tfidf_logreg(split.X_train, split.y_train)

    # Evaluate on val and test
    for name, X, y in [
        ("VAL", split.X_val, split.y_val),
        ("TEST", split.X_test, split.y_test),
    ]:
        proba = bundle.predict_proba(X)
        pred = bundle.predict(X)

        acc = accuracy_score(y, pred)
        f1_macro = f1_score(y, pred, average="macro")
        f1_weighted = f1_score(y, pred, average="weighted")

        print("\n" + "=" * 80)
        print(f"{name} METRICS")
        print("=" * 80)
        print(f"Accuracy:     {acc:.4f}")
        print(f"F1 macro:     {f1_macro:.4f}")
        print(f"F1 weighted:  {f1_weighted:.4f}")

        # Auto-routing metrics (coverage vs precision tradeoff)
        coverage, prec_cov, acc_cov = auto_route_metrics(
            y_true=y,
            proba=proba,
            labels=bundle.labels,
            min_conf=AUTO_ROUTE_MIN_CONFIDENCE,
        )
        print(f"\nAuto-route threshold: {AUTO_ROUTE_MIN_CONFIDENCE:.2f}")
        print(f"Coverage (auto-routed %): {coverage*100:.2f}%")
        print(f"Precision on covered:      {prec_cov:.4f}")
        print(f"Accuracy on covered:       {acc_cov:.4f}")

        print("\nClassification report:")
        print(classification_report(y, pred, digits=4))

        if name == "TEST":
            labels = bundle.labels
            cm = confusion_matrix(y, pred, labels=labels)
            out_png = "artifacts/confusion_matrix_test.png"
            save_confusion_matrix_png(cm, labels, out_png)
            print(f"\nSaved confusion matrix to: {out_png}")


if __name__ == "__main__":
    main()
