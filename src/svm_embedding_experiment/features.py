"""Transformer trong pipeline để TF-IDF được fit riêng trong mỗi fold."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted

from src.embedding_experiment.common import DIMENSION


def input_frame(prompts, vectors=None):
    prompts = list(prompts)
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Cần prompt không rỗng")
    frame = pd.DataFrame({"prompt_text": prompts})
    if vectors is not None:
        vectors = np.asarray(vectors)
        if vectors.shape != (len(prompts), DIMENSION) or not np.isfinite(vectors).all():
            raise ValueError("Embedding không hợp lệ")
        frame["embedding"] = list(vectors)
    return frame


class SVMFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, variant="tfidf", max_features=5000):
        self.variant = variant
        self.max_features = max_features

    def fit(self, X, y=None):
        if self.variant not in ("tfidf", "embedding", "combined"):
            raise ValueError("Variant không hợp lệ")
        if self.variant != "embedding":
            self.vectorizer_ = TfidfVectorizer(max_features=self.max_features)
            self.vectorizer_.fit(X.prompt_text)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "fitted_")
        words = self.vectorizer_.transform(X.prompt_text) if self.variant != "embedding" else None
        return self._assemble(X, words)

    def fit_transform(self, X, y=None, **fit_params):
        # Giữ đúng fit_transform của v1: fit rồi transform riêng có thể đổi
        # thứ tự phần tử CSR, gây sai khác số học nhỏ trong bộ giải LinearSVC.
        if self.variant not in ("tfidf", "embedding", "combined"):
            raise ValueError("Variant không hợp lệ")
        words = None
        if self.variant != "embedding":
            self.vectorizer_ = TfidfVectorizer(max_features=self.max_features)
            words = self.vectorizer_.fit_transform(X.prompt_text)
        self.fitted_ = True
        return self._assemble(X, words)

    def _assemble(self, X, words):
        if self.variant == "tfidf":
            return words
        vectors = np.stack(X.embedding).astype(np.float64)
        if vectors.shape != (len(X), DIMENSION) or not np.isfinite(vectors).all():
            raise ValueError("Embedding không hợp lệ")
        if self.variant == "embedding":
            return vectors
        # Hai block có norm 1. Ghép rồi L2-normalize để norm tổng vẫn bằng 1.
        # Giữ sparse, không chuyển 5000 chiều TF-IDF thành dense.
        return normalize(sparse.hstack([words, sparse.csr_matrix(vectors)], format="csr"))
