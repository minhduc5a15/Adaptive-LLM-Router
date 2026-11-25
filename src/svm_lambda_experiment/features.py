"""X = normalize([TF-IDF, embedding_weight * MiniLM])."""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from src.embedding_experiment.common import DIMENSION
from src.svm_embedding_experiment.features import SVMFeatures


class WeightedSVMFeatures(SVMFeatures):
    def __init__(self, embedding_weight=1.0, max_features=5000):
        super().__init__(variant='combined', max_features=max_features)
        self.embedding_weight = embedding_weight

    def _assemble(self, X, words):
        if not np.isfinite(self.embedding_weight) or self.embedding_weight <= 0:
            raise ValueError('embedding_weight phải hữu hạn và > 0')
        vectors = np.stack(X.embedding).astype(np.float64)
        if vectors.shape != (len(X), DIMENSION) or not np.isfinite(vectors).all():
            raise ValueError('Embedding không hợp lệ')
        weighted = sparse.csr_matrix(self.embedding_weight * vectors)
        return normalize(sparse.hstack([words, weighted], format='csr'))
