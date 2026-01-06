import re
import numpy as np

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)

def load_calibration_params(path: str) -> dict:
    """
    Reads artifacts/calibration.txt with lines like:
      T=0.55
      threshold=0.865
    """
    params = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            params[k.strip()] = float(v.strip())
    return params

def calibrated_proba_from_bundle(bundle, texts, T: float) -> np.ndarray:
    X = bundle.vectorizer.transform(texts)
    logits = bundle.clf.decision_function(X)  # (n, k)
    return softmax(logits / T)
