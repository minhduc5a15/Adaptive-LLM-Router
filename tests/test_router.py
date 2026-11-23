import numpy as np
import pandas as pd
import pytest

from src.evaluation import routing_metrics, select_policy
from src.features import extract_structure_features
from src.router import combine_probabilities, route_prompts


def test_structure_features_keep_blank_lines_and_whitespace_definition():
    features = extract_structure_features(["Hello 123\nNext line", "a b\n\nc", "中文"])
    assert features.values.tolist() == [[19, 4, 2], [6, 3, 3], [2, 1, 1]]


def test_router_accuracy_and_answer_quality_are_different():
    data = pd.DataFrame({
        "label": [0, 1, 0, 0], "small_score": [0, 0, 1, 1],
        "large_score": [0, 1, 0, 1], "small_cost": [1., 1., 1., 1.],
        "large_cost": [3., 3., 3., 3.],
    })
    small = routing_metrics(data, [0, 0, 0, 0])
    assert small["accuracy"] == 0.75
    assert small["answer_score"] == 0.5
    routed = routing_metrics(data, [0, 1, 0, 1])
    assert routed["answer_score"] == 0.75
    assert routed["total_cost"] == 8
    assert routed["cost_saving"] == pytest.approx(1 / 3)
    assert (routed["tn"], routed["fp"], routed["fn"], routed["tp"]) == (2, 1, 0, 1)


def test_policy_must_meet_quality_constraint_before_cost_minimization():
    rows = [
        {"name": "too cheap", "answer_score": 0.3, "mean_cost": 1, "tree_weight": 0., "threshold": .8},
        {"name": "good", "answer_score": .55, "mean_cost": 2, "tree_weight": .5, "threshold": .3},
        {"name": "expensive", "answer_score": .6, "mean_cost": 3, "tree_weight": 1., "threshold": .1},
    ]
    assert select_policy(rows, .54)["name"] == "good"
    with pytest.raises(ValueError):
        select_policy(rows, .9)


class FixedProbabilities:
    # Kiểm tra lấy đúng cột nhãn 1, không giả định thứ tự classes_ của mọi model.
    classes_ = np.array([1, 0])

    def __init__(self, positive):
        self.positive = positive

    def predict_proba(self, prompts):
        return np.tile([self.positive, 1 - self.positive], (len(prompts), 1))


def test_combination_threshold_and_input_validation():
    bundle = {"tree": FixedProbabilities(.3), "svm": FixedProbabilities(.8),
              "selected_policy": {"tree_weight": .5, "threshold": .55}}
    result = route_prompts(bundle, ["hello"])
    assert result["probability"][0] == pytest.approx(.55)
    assert result["route"].tolist() == [1]
    assert route_prompts(bundle, ["hello"], threshold=.56)["route"].tolist() == [0]
    for prompts in [[], ["  "], [None]]:
        with pytest.raises(ValueError):
            route_prompts(bundle, prompts)
    with pytest.raises(ValueError):
        route_prompts(bundle, ["hello"], threshold=1.1)
    with pytest.raises(ValueError):
        combine_probabilities({"tree": np.array([.3]), "svm": np.array([.8])}, -1)
