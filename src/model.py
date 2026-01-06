from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class TextClassifierBundle:
    vectorizer: TfidfVectorizer
    clf: LogisticRegression
    labels: List[str]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X)

    def predict(self, texts: List[str]) -> List[str]:
        proba = self.predict_proba(texts)
        idx = np.argmax(proba, axis=1)
        return [self.labels[i] for i in idx]

    def topk(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        proba = self.predict_proba([text])[0]
        order = np.argsort(-proba)[:k]
        return [(self.labels[i], float(proba[i])) for i in order]


def train_tfidf_logreg(texts: list, y: list) -> TextClassifierBundle:
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        stop_words="english",
    )
    X = vec.fit_transform(texts)

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(X, y)

    # IMPORTANT: use the model's learned class order
    labels = list(clf.classes_)

    return TextClassifierBundle(
        vectorizer=vec,
        clf=clf,
        labels=labels,
    )

def save_bundle(bundle: TextClassifierBundle, path: str) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: str) -> TextClassifierBundle:
    return joblib.load(path)
