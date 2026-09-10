"""
baselines.py
------------
Three baselines for intent classification comparison:

  1. MajorityClass  — always predicts the most-frequent class
  2. TFIDFLogReg    — TF-IDF features + Logistic Regression
  3. TFIDFSVMRBF    — TF-IDF features + SVM (RBF kernel)

All baselines expose fit(X_train, y_train) and predict(X_test) interfaces
compatible with scikit-learn.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

BASELINE_DIR = Path(__file__).parent.parent / "data" / "baselines"


# ── 1. Majority Class ─────────────────────────────────────────────────────────
class MajorityClassBaseline:
    """Always predicts the most frequent class in training data."""
    name = "majority_class"

    def __init__(self):
        self.majority_label: Optional[str] = None

    def fit(self, X: List[str], y: List[str]) -> "MajorityClassBaseline":
        from collections import Counter
        self.majority_label = Counter(y).most_common(1)[0][0]
        logger.info(f"[MajorityClass] Majority label: {self.majority_label}")
        return self

    def predict(self, X: List[str]) -> List[str]:
        return [self.majority_label] * len(X)

    def predict_proba_dict(self, X: List[str]) -> List[dict]:
        return [{"label": self.majority_label, "prob": 1.0}] * len(X)


# ── 2. TF-IDF + Logistic Regression ──────────────────────────────────────────
class TFIDFLogRegBaseline:
    """
    Classic TF-IDF (char + word n-grams) + L2-regularised Logistic Regression.
    A strong simple baseline that often beats neural models on short text.
    """
    name = "tfidf_logreg"

    def __init__(self, max_features: int = 30_000, C: float = 4.0):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                max_features=max_features,
                sublinear_tf=True,
                strip_accents="unicode",
                lowercase=True,
            )),
            ("clf", LogisticRegression(
                C=C,
                max_iter=1000,
                solver="lbfgs",
                multi_class="multinomial",
                class_weight="balanced",
                n_jobs=-1,
            )),
        ])

    def fit(self, X: List[str], y: List[str]) -> "TFIDFLogRegBaseline":
        logger.info("[TFIDFLogReg] Fitting pipeline …")
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: List[str]) -> List[str]:
        return self.pipeline.predict(X).tolist()

    def predict_proba(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X)


# ── 3. TF-IDF + LinearSVC ────────────────────────────────────────────────────
class TFIDFSVMBaseline:
    """
    TF-IDF (word n-grams) + LinearSVC wrapped in Platt scaling for probabilities.
    Usually the strongest classical baseline on text classification.
    """
    name = "tfidf_svm"

    def __init__(self, max_features: int = 50_000, C: float = 1.0):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                max_features=max_features,
                sublinear_tf=True,
                strip_accents="unicode",
                lowercase=True,
                min_df=2,
            )),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=C, max_iter=5000, class_weight="balanced"),
                cv=3,
            )),
        ])

    def fit(self, X: List[str], y: List[str]) -> "TFIDFSVMBaseline":
        logger.info("[TFIDF-SVM] Fitting pipeline …")
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: List[str]) -> List[str]:
        return self.pipeline.predict(X).tolist()

    def predict_proba(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X)


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate_baseline(
    model,
    X_test: List[str],
    y_test: List[str],
    labels: Optional[List[str]] = None,
) -> dict:
    """Return a metrics dict for a baseline."""
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "macro_f1":  f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "per_class": report,
        "y_pred":    y_pred,
    }


# ── Persistence ───────────────────────────────────────────────────────────────
def save_baseline(model, name: Optional[str] = None) -> Path:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    n = name or model.name
    p = BASELINE_DIR / f"{n}.pkl"
    with open(p, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved baseline → {p}")
    return p


def load_baseline(name: str):
    p = BASELINE_DIR / f"{name}.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Quick self-test with synthetic data
    from intent_taxonomy import LABELS
    import random

    random.seed(42)
    X_train = [
        "where is my order, it hasn't arrived",
        "I want a refund for this broken item",
        "my account is locked, can't login",
        "I was charged twice on my credit card",
        "cancel my prime membership please",
        "the product arrived damaged",
        "my package says delivered but I don't have it",
        "how do I gift wrap an order?",
    ] * 50
    y_train = [
        "ORDER_STATUS", "REFUND_RETURN", "ACCOUNT_ACCESS", "BILLING_CHARGE",
        "PRIME_MEMBERSHIP", "PRODUCT_ISSUE", "DELIVERY_PROBLEM", "GENERAL_INQUIRY",
    ] * 50

    X_test  = X_train[:16]
    y_test  = y_train[:16]

    for Cls in [MajorityClassBaseline, TFIDFLogRegBaseline, TFIDFSVMBaseline]:
        model = Cls()
        model.fit(X_train, y_train)
        metrics = evaluate_baseline(model, X_test, y_test, labels=LABELS)
        print(f"\n{model.name}: acc={metrics['accuracy']:.3f} | macro_f1={metrics['macro_f1']:.3f}")
