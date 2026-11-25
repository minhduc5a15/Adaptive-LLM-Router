"""Chọn ngưỡng cố định bằng validation. Chi phí không phải feature inference."""

import numpy as np
import pandas as pd

from src.evaluation import selected_outcomes
from src.experiments.config import TREE_WEIGHTS
from src.experiments.inference import combine_scores


def summarize_routes(data, routes):
    scores, costs = selected_outcomes(data, routes)
    return {"n_samples": len(data), "correct_count": int(scores.sum()),
            "answer_score": float(scores.mean()), "mean_cost": float(costs.mean()),
            "total_cost": float(costs.sum()),
            "cost_saving": float(1 - costs.sum() / data.large_cost.sum()),
            "large_rate": float(np.mean(routes))}


def sweep_policies(data, components, target):
    # Độ phân giải giống v1. 1.01 biểu diễn luôn Small kể cả score == 1.
    thresholds = np.r_[np.linspace(0, 1, 101) if target == "routing"
                       else np.linspace(-1, 1, 201), 1.01]
    rows = []
    for weight in TREE_WEIGHTS:
        score, _ = combine_scores(components, target, weight)
        name = "SVM" if weight == 0 else "Decision Tree" if weight == 1 else "Ensemble"
        for threshold in thresholds:
            metrics = summarize_routes(data, (score >= threshold).astype(int))
            rows.append({"name": name, "tree_weight": weight, "threshold": float(threshold), **metrics})
    return pd.DataFrame(rows)


def select_policy(table, objective, target_saving, minimum_score):
    if objective == "fixed_budget":
        eligible = table.loc[table.cost_saving >= target_saving - 1e-12]
        columns, ascending = ["answer_score", "mean_cost", "tree_weight", "threshold"], [False, True, True, True]
    elif objective == "quality_constraint":
        eligible = table.loc[table.answer_score >= minimum_score - 1e-12]
        columns, ascending = ["mean_cost", "answer_score", "tree_weight", "threshold"], [True, False, True, True]
    else:
        raise ValueError(f"Objective không hợp lệ: {objective}")
    if eligible.empty:
        raise ValueError(f"Không có chính sách khả thi cho {objective}")
    return eligible.sort_values(columns, ascending=ascending, kind="stable").iloc[0].to_dict()
