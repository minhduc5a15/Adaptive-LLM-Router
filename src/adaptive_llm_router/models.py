"""Factories for the two algorithms required by the course."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def decision_tree(random_state: int = 42) -> DecisionTreeClassifier:
    """Train on the DataFrame returned by ``structural_features``."""
    return DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=random_state,
    )


def text_svm() -> Pipeline:
    """Fit TF-IDF on training prompts only, inside the pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("svm", LinearSVC(class_weight="balanced")),
        ]
    )
