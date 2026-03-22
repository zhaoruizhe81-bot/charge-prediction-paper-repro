"""传统机器学习模型定义。"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=120000,
        sublinear_tf=True,
    )


def build_model(name: str, seed: int = 42) -> Pipeline:
    model_name = name.lower()
    tfidf = build_tfidf()

    if model_name in {"lr", "logistic", "logistic_regression"}:
        clf = LogisticRegression(
            max_iter=1200,
            solver="saga",
            class_weight="balanced",
            C=2.0,
            n_jobs=-1,
            random_state=seed,
        )
    elif model_name in {"svm", "linear_svm", "linearsvc"}:
        clf = LinearSVC(C=1.2, class_weight="balanced", random_state=seed)
    elif model_name in {"sgd", "sgd_log"}:
        clf = SGDClassifier(
            loss="log_loss",
            alpha=1e-6,
            max_iter=2000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed,
        )
    elif model_name in {"pa", "passive_aggressive"}:
        clf = PassiveAggressiveClassifier(
            C=0.8,
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported model: {name}")

    return Pipeline([
        ("tfidf", tfidf),
        ("clf", clf),
    ])
