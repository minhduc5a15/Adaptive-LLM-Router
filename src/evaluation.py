"""Đo nhãn Router, điểm câu trả lời và chi phí thành ba đại lượng riêng."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)


def selected_outcomes(data, routes):
    routes = np.asarray(routes)
    if routes.shape != (len(data),) or not np.isin(routes, [0, 1]).all():
        raise ValueError("Mỗi dòng dữ liệu cần đúng một nhãn 0 hoặc 1")
    scores = np.where(routes == 1, data["large_score"], data["small_score"])
    costs = np.where(routes == 1, data["large_cost"], data["small_cost"])
    return scores, costs


def routing_metrics(data, routes, probabilities=None):
    routes = np.asarray(routes)
    scores, costs = selected_outcomes(data, routes)
    y = data["label"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, routes, labels=[0, 1]).ravel()
    result = {
        "n_samples": len(data),
        "accuracy": float(accuracy_score(y, routes)),
        "balanced_accuracy": float(balanced_accuracy_score(y, routes)),
        "precision": float(precision_score(y, routes, zero_division=0)),
        "recall": float(recall_score(y, routes, zero_division=0)),
        "f1": float(f1_score(y, routes, zero_division=0)),
        "answer_score": float(scores.mean()),
        "mean_cost": float(costs.mean()), "total_cost": float(costs.sum()),
        "large_rate": float(routes.mean()),
        "cost_saving": float(1 - costs.sum() / data["large_cost"].sum()),
        "answer_delta_vs_large": float(scores.mean() - data["large_score"].mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if probabilities is not None:
        result.update({
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "brier": float(brier_score_loss(y, probabilities)),
            "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        })
    return result


def sweep_thresholds(data, probabilities, tree_weight, name):
    """Lưới ngưỡng cố định trước khi xem test."""
    rows = []
    for threshold in np.linspace(0, 1, 101):
        routes = (probabilities >= threshold).astype(int)
        scores, costs = selected_outcomes(data, routes)
        rows.append({
            "name": name, "tree_weight": float(tree_weight),
            "threshold": float(threshold), "answer_score": float(scores.mean()),
            "mean_cost": float(costs.mean()), "large_rate": float(routes.mean()),
        })
    return rows


def select_policy(rows, minimum_score):
    """Chi phí nhỏ nhất thỏa ràng buộc trên validation, sau đó ưu tiên điểm cao."""
    eligible = [r for r in rows if r["answer_score"] >= minimum_score - 1e-12]
    if not eligible:
        raise ValueError("Không có chính sách đạt điểm yêu cầu trên validation")
    return dict(min(eligible, key=lambda r: (
        r["mean_cost"], -r["answer_score"], r["tree_weight"], r["threshold"],
    )))


def bootstrap_vs_large(data, routes, repetitions=1000, seed=42):
    """Bootstrap ghép cặp theo nhóm prompt, giữ quan hệ giữa bản ghi trùng."""
    scores, costs = selected_outcomes(data, routes)
    groups, inverse = np.unique(data["prompt_text"].to_numpy(), return_inverse=True)
    group_size = np.bincount(inverse)
    score_diff = np.bincount(inverse, weights=scores - data["large_score"].to_numpy())
    group_cost = np.bincount(inverse, weights=costs)
    large_cost = np.bincount(inverse, weights=data["large_cost"].to_numpy())
    rng = np.random.default_rng(seed)
    differences, savings = [], []
    for _ in range(repetitions):
        sample = rng.integers(0, len(groups), size=len(groups))
        differences.append(score_diff[sample].sum() / group_size[sample].sum())
        savings.append(1 - group_cost[sample].sum() / large_cost[sample].sum())
    return {
        "answer_delta_95ci": np.quantile(differences, [0.025, 0.975]).tolist(),
        "cost_saving_95ci": np.quantile(savings, [0.025, 0.975]).tolist(),
        "repetitions": repetitions,
    }
