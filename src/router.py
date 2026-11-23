"""Suy luận từ mô hình đã lưu; không huấn luyện lại khi dự đoán."""

from pathlib import Path
import joblib
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "router.joblib"
MODEL_NAMES = {0: "mistralai/mistral-7b-chat", 1: "mistralai/mixtral-8x7b-chat"}


def load_router(path=DEFAULT_MODEL_PATH):
    # Chỉ nạp artifact do chính dự án tạo ra, vì joblib dùng pickle.
    return joblib.load(path)


def predict_probabilities(bundle, prompts):
    prompts = list(prompts)
    result = {}
    for name in ["tree", "svm"]:
        model = bundle[name]
        positive_column = list(model.classes_).index(1)
        result[name] = model.predict_proba(prompts)[:, positive_column]
    return result


def combine_probabilities(probabilities, tree_weight):
    if not 0 <= tree_weight <= 1:
        raise ValueError("Trọng số cây phải nằm trong [0, 1]")
    return (tree_weight * probabilities["tree"]
            + (1 - tree_weight) * probabilities["svm"])


def route_prompts(bundle, prompts, policy=None, threshold=None):
    prompts = list(prompts)
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Cần ít nhất một prompt không rỗng")
    policy = bundle["selected_policy"] if policy is None else policy
    threshold = policy["threshold"] if threshold is None else threshold
    if not 0 <= threshold <= 1:
        raise ValueError("Ngưỡng phải nằm trong [0, 1]")
    probabilities = predict_probabilities(bundle, prompts)
    combined = combine_probabilities(probabilities, policy["tree_weight"])
    routes = (combined >= threshold).astype(int)
    return {
        "tree_probability": probabilities["tree"],
        "svm_probability": probabilities["svm"],
        "probability": combined, "route": routes,
        "model": np.array([MODEL_NAMES[int(route)] for route in routes]),
    }
