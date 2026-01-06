import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.preprocessing import normalize
from src.model import train_tfidf_logreg


# -----------------------------
# Utilities
# -----------------------------
def stratified_split(
    texts: List[str],
    labels: List[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    val_frac_of_trainval = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_frac_of_trainval,
        random_state=seed,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def one_hot(y: np.ndarray, class_to_idx: Dict[str, int]) -> np.ndarray:
    oh = np.zeros((len(y), len(class_to_idx)), dtype=np.float64)
    for i, label in enumerate(y):
        oh[i, class_to_idx[label]] = 1.0
    return oh


# -----------------------------
# Temperature Scaling
# -----------------------------
@dataclass
class TemperatureScaler:
    T: float  # temperature

    def transform_proba_from_logits(self, logits: np.ndarray) -> np.ndarray:
        return softmax(logits / self.T)


def fit_temperature_scaling(
    logits: np.ndarray,
    y_true: List[str],
    labels: List[str],
    T_grid: np.ndarray = None,
) -> TemperatureScaler:
    """
    Simple grid-search temperature scaling minimizing NLL on validation.
    Works well and avoids extra dependencies.
    """
    if T_grid is None:
        T_grid = np.linspace(0.5, 5.0, 91)  # 0.5..5.0 step 0.05

    class_to_idx = {c: i for i, c in enumerate(labels)}
    y = np.array(y_true)
    y_oh = one_hot(y, class_to_idx)

    best_T = 1.0
    best_nll = float("inf")

    for T in T_grid:
        proba = softmax(logits / T)
        # NLL
        eps = 1e-12
        nll = -np.mean(np.sum(y_oh * np.log(np.clip(proba, eps, 1.0)), axis=1))
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)

    return TemperatureScaler(T=best_T)


# -----------------------------
# Calibration metrics
# -----------------------------
def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    ECE for top-1 confidence. Bins confidence, compares avg confidence vs accuracy.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def save_reliability_plot(conf: np.ndarray, correct: np.ndarray, out_path: str, n_bins: int = 10) -> None:
    import matplotlib.pyplot as plt

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_acc = []
    bin_conf = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_acc.append(correct[mask].mean())
        bin_conf.append(conf[mask].mean())
        bin_counts.append(mask.sum())

    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1])  # perfect calibration line
    ax.plot(bin_conf, bin_acc, marker="o")
    ax.set_xlabel("Average predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability Diagram (Top-1)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Threshold tuning for auto-routing
# -----------------------------
def auto_route_precision_coverage(
    y_true: List[str],
    y_pred: List[str],
    conf: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:
    """
    Returns (precision_on_covered, coverage)
    For single-label multi-class, precision on covered == accuracy on covered.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    covered = conf >= threshold
    if covered.sum() == 0:
        return 0.0, 0.0

    precision = float((y_true[covered] == y_pred[covered]).mean())
    coverage = float(covered.mean())
    return precision, coverage


def find_threshold_for_target_precision(
    y_true: List[str],
    y_pred: List[str],
    conf: np.ndarray,
    target_precision: float = 0.95,
) -> Dict[str, float]:
    thresholds = np.linspace(0.0, 1.0, 201)  # step 0.005
    best = {"threshold": 1.0, "precision": 0.0, "coverage": 0.0}

    # Choose max coverage among thresholds meeting precision target
    for t in thresholds:
        p, c = auto_route_precision_coverage(y_true, y_pred, conf, float(t))
        if p >= target_precision:
            if c > best["coverage"]:
                best = {"threshold": float(t), "precision": float(p), "coverage": float(c)}

    # If none meet target, pick best precision threshold with non-trivial coverage
    if best["coverage"] == 0.0:
        for t in thresholds:
            p, c = auto_route_precision_coverage(y_true, y_pred, conf, float(t))
            # prefer higher precision, then coverage
            if (p > best["precision"]) or (np.isclose(p, best["precision"]) and c > best["coverage"]):
                best = {"threshold": float(t), "precision": float(p), "coverage": float(c)}

    return best


# -----------------------------
# Main
# -----------------------------
def main():
    data_path = "data/tickets.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Missing data/tickets.csv. Run: python -m scripts.build_dataset")

    df = pd.read_csv(data_path)
    df["text"] = df["text"].astype(str).apply(normalize)
    df["category"] = df["category"].astype(str)

    # Drop tiny classes to keep stratified splits stable
    min_count = 50
    vc = df["category"].value_counts()
    keep = vc[vc >= min_count].index
    df = df[df["category"].isin(keep)].copy()

    print(f"Rows after filtering: {len(df):,}")
    print("Class distribution:")
    print(df["category"].value_counts())

    texts = df["text"].tolist()
    labels = df["category"].tolist()

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(texts, labels)

    # Train baseline
    bundle = train_tfidf_logreg(X_train, y_train)
    model_labels = bundle.labels
    print("\nModel label order:", model_labels)

    # Get logits on VAL
    Xv = bundle.vectorizer.transform(X_val)
    logits_val = bundle.clf.decision_function(Xv)  # shape: (n, k)

    # Fit temperature scaling on VAL
    scaler = fit_temperature_scaling(logits_val, y_val, model_labels)
    print(f"\nBest temperature T: {scaler.T:.3f}")

    # Evaluate calibration on VAL (before vs after)
    # Before: use model's predict_proba (already softmax internally for multinomial LR)
    proba_val_before = bundle.clf.predict_proba(Xv)
    pred_val = bundle.predict(X_val)

    conf_before = np.max(proba_val_before, axis=1)
    correct = (np.array(pred_val) == np.array(y_val))

    ece_before = expected_calibration_error(conf_before, correct, n_bins=15)

    proba_val_after = scaler.transform_proba_from_logits(logits_val)
    conf_after = np.max(proba_val_after, axis=1)
    ece_after = expected_calibration_error(conf_after, correct, n_bins=15)

    print(f"VAL ECE before: {ece_before:.4f}")
    print(f"VAL ECE after : {ece_after:.4f}")

    # Tune threshold on VAL for a target precision
    target_precision = 0.95
    best = find_threshold_for_target_precision(y_val, pred_val, conf_after, target_precision=target_precision)

    print("\nThreshold tuning on VAL (using calibrated confidence):")
    print(f"Target precision: {target_precision:.2f}")
    print(f"Chosen threshold: {best['threshold']:.3f}")
    print(f"Precision:        {best['precision']:.4f}")
    print(f"Coverage:         {best['coverage']*100:.2f}%")

    # Save reliability plot
    out_plot = "artifacts/reliability_val.png"
    save_reliability_plot(conf_after, correct.astype(float), out_plot, n_bins=10)
    print(f"\nSaved reliability plot to: {out_plot}")

    # Quick sanity on TEST using the VAL-chosen threshold + temperature
    Xt = bundle.vectorizer.transform(X_test)
    logits_test = bundle.clf.decision_function(Xt)
    proba_test_cal = scaler.transform_proba_from_logits(logits_test)

    pred_test = bundle.predict(X_test)
    conf_test = np.max(proba_test_cal, axis=1)

    p_test, c_test = auto_route_precision_coverage(y_test, pred_test, conf_test, threshold=best["threshold"])
    print("\nTEST auto-routing using VAL-chosen threshold:")
    print(f"Precision: {p_test:.4f}")
    print(f"Coverage:  {c_test*100:.2f}%")

    # Save calibration params so the app can use them later
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/calibration.txt", "w", encoding="utf-8") as f:
        f.write(f"T={scaler.T}\n")
        f.write(f"threshold={best['threshold']}\n")
        f.write(f"target_precision={target_precision}\n")
    print("\nSaved calibration params to: artifacts/calibration.txt")


if __name__ == "__main__":
    main()

