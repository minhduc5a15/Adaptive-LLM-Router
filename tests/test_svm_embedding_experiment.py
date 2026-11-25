"""Kiểm tra ranh giới fold và cách ghép hai biểu diễn cho SVM."""
import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer

pytest.importorskip('onnxruntime')
pytest.importorskip('tokenizers')
from src.svm_embedding_experiment.features import input_frame, SVMFeatures


def test_tfidf_fit_transform_preserves_baseline_csr_order():
    frame = input_frame(['binary search tree', 'linear search', 'tree algorithm'])
    baseline = TfidfVectorizer(max_features=5000).fit_transform(frame.prompt_text)
    actual = SVMFeatures('tfidf').fit_transform(frame)
    np.testing.assert_array_equal(actual.indptr, baseline.indptr)
    np.testing.assert_array_equal(actual.indices, baseline.indices)
    np.testing.assert_array_equal(actual.data, baseline.data)


def test_tfidf_does_not_learn_held_out_vocabulary():
    train = input_frame(['binary search', 'linear search'])
    validation = input_frame(['unseenvalidationword search'])
    feature = clone(SVMFeatures('tfidf')).fit(train)
    vocabulary = feature.vectorizer_.vocabulary_.copy()
    matrix = feature.transform(validation)
    assert 'unseenvalidationword' not in feature.vectorizer_.vocabulary_
    assert feature.vectorizer_.vocabulary_ == vocabulary
    assert matrix.shape == (1, 3)
    assert matrix.nnz == 1


def test_combined_is_sparse_normalized_and_keeps_both_blocks():
    embeddings = np.zeros((2, 384), dtype=np.float32)
    embeddings[0, 0] = 1
    embeddings[1, 1] = 1
    frame = input_frame(['binary search', 'linear search'], embeddings)
    feature = SVMFeatures('combined').fit(frame)
    combined = feature.transform(frame)
    words = feature.vectorizer_.transform(frame.prompt_text)
    assert sparse.isspmatrix_csr(combined)
    assert combined.shape == (2, 387)
    np.testing.assert_allclose(combined[:, :3].toarray(), words.toarray() / np.sqrt(2))
    np.testing.assert_allclose(combined[:, 3:].toarray(), embeddings / np.sqrt(2))
    np.testing.assert_allclose(np.asarray(combined.multiply(combined).sum(axis=1)).ravel(), 1)
    np.testing.assert_array_equal(SVMFeatures('embedding').fit_transform(frame), embeddings)
    with pytest.raises(ValueError):
        input_frame(['one'], embeddings)
