"""Kiểm tra phủ token, pooling, cache và phép ghép đặc trưng."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

from src.embedding_experiment.common import DIMENSION, ENCODER_DIR
from src.embedding_experiment.encoder import (
    FrozenEncoder, cached_embeddings, masked_mean, normalize_rows, split_token_ids,
)
from src.embedding_experiment.inference import feature_matrix, route_scores


def test_chunks_preserve_all_tokens_and_order():
    ids = list(range(700))
    chunks = split_token_ids(ids)
    assert [len(chunk) for chunk in chunks] == [254, 254, 192]
    assert [token for chunk in chunks for token in chunk] == ids
    with pytest.raises(ValueError):
        split_token_ids([])


def test_masked_pooling_drops_padding_and_l2_normalizes():
    tokens = np.array([[[1., 2.], [3., 4.], [1000., 1000.]]])
    pooled = masked_mean(tokens, np.array([[1, 1, 0]]))
    np.testing.assert_allclose(pooled, [[2., 3.]])
    np.testing.assert_allclose(np.linalg.norm(normalize_rows(pooled), axis=1), [1.])
    with pytest.raises(ValueError):
        normalize_rows(np.zeros((1, 2)))


def test_combined_features_keep_structure_then_embedding():
    embeddings = np.arange(2 * DIMENSION).reshape(2, DIMENSION)
    features = feature_matrix(["a b\nc", "x"], embeddings, "combined")
    assert features.shape == (2, 387)
    np.testing.assert_array_equal(features[:, :3], [[5, 3, 2], [1, 1, 1]])
    np.testing.assert_array_equal(features[:, 3:], embeddings)
    with pytest.raises(ValueError):
        feature_matrix(["x"], np.ones((2, DIMENSION)), "embedding")


def test_threshold_and_frozen_svm_mixing():
    tree, svm = np.array([.2, .8]), np.array([.4, .4])
    scores, routes = route_scores(tree, svm, {"tree_weight": .5, "threshold": .5})
    np.testing.assert_allclose(scores, [.3, .6])
    np.testing.assert_array_equal(routes, [0, 1])
    _, routes = route_scores(tree, svm, {"tree_weight": 0., "threshold": .4})
    np.testing.assert_array_equal(routes, [1, 1])
    _, routes = route_scores(tree, svm, {"tree_weight": 1., "threshold": 1.01})
    np.testing.assert_array_equal(routes, [0, 0])


class FakeEncoder:
    manifest = {"fake_model": True}

    def encode(self, prompts):
        return np.full((len(prompts), DIMENSION), 1 / np.sqrt(DIMENSION), dtype=np.float32), {}


def test_cache_rejects_prompt_reordering_and_corrupt_arrays(tmp_path):
    data = pd.DataFrame({"sample_id": ["1", "2"], "prompt_text": ["one", "two"]})
    encoder = FakeEncoder()
    first, _ = cached_embeddings(encoder, data, tmp_path, "train")
    second, _ = cached_embeddings(encoder, data, tmp_path, "train")
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError):
        cached_embeddings(encoder, data.iloc[::-1], tmp_path, "train")
    np.save(tmp_path / "train.npy", first * 2)
    with pytest.raises(ValueError):
        cached_embeddings(encoder, data, tmp_path, "train")


@pytest.mark.skipif(not (ENCODER_DIR / "manifest.json").exists(), reason="Chưa tải encoder")
def test_real_encoder_batch_invariance_and_long_prompt_tail():
    encoder = FrozenEncoder()
    prompts = ["Explain binary search.", "cat " * 254 + "dog " * 254, "cat " * 508]
    batch, stats = encoder.encode(prompts, batch_size=3, progress=False)
    individual, _ = encoder.encode(prompts, batch_size=1, progress=False)
    np.testing.assert_allclose(batch, individual, atol=2e-6, rtol=2e-4)
    np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1., atol=1e-6)
    assert stats["multi_chunk_prompts"] == 2
    assert stats["truncated_content_tokens"] == 0
    assert not np.allclose(batch[1], batch[2], atol=1e-4)  # Phần cuối có tác động.


def test_label_errors_distinguish_wasted_cost_from_lost_correctness():
    from src.embedding_experiment.evaluate import error_counts, compare_paired

    data = pd.DataFrame({
        "prompt_text": ["a", "b", "c", "d"], "label": [0, 0, 0, 1],
        "small_score": [0, 1, 1, 0], "large_score": [0, 1, 0, 1],
        "small_cost": [1., 1., 1., 1.], "large_cost": [2., 2., 2., 2.],
    })
    routes = np.array([1, 1, 1, 0])
    errors = error_counts(data, routes)
    assert errors["fn"] == 1 and errors["fp"] == 3
    assert errors["fp_both_wrong"] == errors["fp_both_correct"] == errors["fp_harmful"] == 1
    comparison = compare_paired(data, routes, data.label.to_numpy(), repetitions=100)
    assert comparison["gained_correct"] == 0 and comparison["lost_correct"] == 2
    assert comparison["answer_delta"] == -.5
    assert comparison["saving_delta"] == -.25
    same = compare_paired(data, routes, routes, repetitions=100)
    assert same["answer_delta_95ci"] == [0., 0.]
    assert same["saving_delta_95ci"] == [0., 0.]
