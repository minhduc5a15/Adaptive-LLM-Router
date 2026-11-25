"""Suy luận chỉ nhận prompt, không nhận điểm đúng hay chi phí thực của LLM."""

import numpy as np

from src.experiments.models import positive_probability
from src.router import MODEL_NAMES


def predict_components(bundle, prompts):
    prompts = list(prompts)
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Cần ít nhất một prompt không rỗng")
    return {f"{family}_{target}": positive_probability(model, prompts)
            for family, models in bundle["models"].items() for target, model in models.items()}


def combine_scores(components, target, tree_weight):
    """Cùng trọng số cho hai LLM; gain nằm trong [-1, 1], không phải xác suất."""
    if not 0 <= tree_weight <= 1:
        raise ValueError("Trọng số cây phải nằm trong [0, 1]")
    targets = ["label"] if target == "routing" else ["small_score", "large_score"]
    mixed = {name: tree_weight * components[f"tree_{name}"]
             + (1 - tree_weight) * components[f"svm_{name}"] for name in targets}
    score = mixed["label"] if target == "routing" else mixed["large_score"] - mixed["small_score"]
    return score, mixed


def route_prompts(bundle, prompts, policy=None):
    policy = bundle["selected_policy"] if policy is None else policy
    components = predict_components(bundle, prompts)
    score, mixed = combine_scores(components, bundle["experiment"]["target"], policy["tree_weight"])
    routes = (score >= policy["threshold"]).astype(int)
    return {"score": score, "estimates": mixed, "components": components,
            "route": routes, "model": np.array([MODEL_NAMES[int(r)] for r in routes])}
