"""Chung phép dựng đặc trưng và định tuyến cho thí nghiệm, CLI và đánh giá."""

import numpy as np

from src.features import extract_structure_features
from src.embedding_experiment.common import DIMENSION
from src.router import MODEL_NAMES


def feature_matrix(prompts, embeddings, variant):
    structural = extract_structure_features(prompts).to_numpy(dtype=np.float32)
    if variant == "structure":
        return structural
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (len(structural), DIMENSION) or not np.isfinite(embeddings).all():
        raise ValueError("Embedding sai kích thước hoặc có giá trị không hữu hạn")
    if variant == "embedding":
        return embeddings
    if variant == "combined":
        return np.hstack([structural, embeddings])
    raise ValueError(f"Variant không hợp lệ: {variant}")


def positive_probability(model, features):
    return model.predict_proba(features)[:, list(model.classes_).index(1)]


def route_scores(tree_probability, svm_probability, policy):
    weight = policy["tree_weight"]
    if not 0 <= weight <= 1 or not 0 <= policy["threshold"] <= 1.01:
        raise ValueError("Trọng số hoặc ngưỡng không hợp lệ")
    scores = weight * tree_probability + (1 - weight) * svm_probability
    return scores, (scores >= policy["threshold"]).astype(int)


def predict(bundle, prompts, variant, mode="tuned_router", encoder=None):
    prompts = list(prompts)
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Cần prompt không rỗng")
    vectors = None
    if variant != "structure":
        if encoder is None:
            raise ValueError("Cần encoder cho variant embedding")
        if encoder.manifest != bundle["encoder"]:
            raise ValueError("Encoder khác encoder lúc training")
        vectors, _ = encoder.encode(prompts, progress=False)
    matrix = feature_matrix(prompts, vectors, variant)
    tree = positive_probability(bundle["trees"][variant], matrix)
    svm = positive_probability(bundle["svm"], prompts)
    policy = bundle["policies"][variant][mode]
    scores, routes = route_scores(tree, svm, policy)
    return {"tree_probability": tree, "svm_probability": svm, "score": scores,
            "route": routes, "model": np.array([MODEL_NAMES[int(r)] for r in routes])}
