"""Kiểm thử các rủi ro của v2: lợi ích âm, ngân sách, rò rỉ và artifact."""

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.experiments.data import check_disjoint
from src.experiments.evaluate import paired_comparison
from src.experiments.features import extract_rich_features, make_text_features
from src.experiments.inference import combine_scores, route_prompts
from src.experiments.models import make_folds
from src.experiments.policy import select_policy, summarize_routes, sweep_policies


class FixedModel:
    classes_ = np.array([1, 0])  # Cố ý đảo cột để phát hiện giả định [:, 1].

    def __init__(self, positive):
        self.positive = positive

    def predict_proba(self, prompts):
        return np.tile([self.positive, 1 - self.positive], (len(prompts), 1))


def example_data():
    return pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"], "prompt_text": ["A", "B", "C", "D"],
        "label": [0, 0, 1, 0], "small_score": [0, 1, 0, 1], "large_score": [0, 0, 1, 1],
        "small_cost": [1., 1., 1., 1.], "large_cost": [3., 3., 3., 3.],
    })


def test_gain_distinguishes_four_outcomes_and_can_be_negative():
    data = example_data()
    components = {f"{family}_{target}": data[target].to_numpy()
                  for family in ["tree", "svm"] for target in ["small_score", "large_score"]}
    gain, mixed = combine_scores(components, "quality", .3)
    np.testing.assert_allclose(gain, [0, -1, 1, 0])
    routes = (gain >= .01).astype(int)
    result = summarize_routes(data, routes)
    assert result["correct_count"] == 3
    assert result["answer_score"] == .75
    assert result["cost_saving"] == .5


def test_budget_selection_and_negative_thresholds():
    data = example_data()
    components = {"tree_small_score": np.array([.3, .8, .2, .9]),
                  "svm_small_score": np.array([.3, .8, .2, .9]),
                  "tree_large_score": np.array([.2, .2, .9, .8]),
                  "svm_large_score": np.array([.2, .2, .9, .8])}
    table = sweep_policies(data, components, "quality")
    assert (table.threshold < 0).any()
    assert table.large_rate.min() == 0 and table.large_rate.max() == 1
    chosen = select_policy(table, "fixed_budget", .2, .48)
    assert chosen["answer_score"] == .75
    assert chosen["cost_saving"] >= .2
    # Không để chính sách đúng cao nhưng quá ngân sách thắng.
    impossible = table.assign(answer_score=1., cost_saving=-.1)
    with pytest.raises(ValueError):
        select_policy(impossible, "fixed_budget", .2, .48)


def test_rich_features_handle_empty_text_and_are_literal_counts():
    features = extract_rich_features(["", "Hi 12?\n```x+1```", "中文"])
    assert np.isfinite(features.to_numpy()).all()
    assert features.iloc[1].code_fence_count == 2
    assert features.iloc[1].math_symbol_count == 1
    assert features.iloc[1].question_mark_count == 1
    assert features.iloc[1].digit_ratio == pytest.approx(3 / 16)
    assert features.iloc[2].word_count == 1


def test_text_profiles_stay_sparse_and_transform_does_not_learn_validation():
    texts = ["binary search example"] * 3 + ["write python function"] * 3
    transformer = make_text_features("word_char")
    train = transformer.fit_transform(texts)
    vocabulary = transformer.transformer_list[0][1].vocabulary_.copy()
    validation = transformer.transform(["unseenvalidationword"])
    assert sparse.issparse(train) and sparse.issparse(validation)
    assert train.shape[1] == validation.shape[1]
    assert transformer.transformer_list[0][1].vocabulary_ == vocabulary
    assert "unseenvalidationword" not in vocabulary


def test_grouped_calibration_keeps_duplicate_prompts_together():
    data = pd.concat([example_data().assign(prompt_text=lambda d: d.prompt_text + str(i),
                                          sample_id=lambda d: d.sample_id + str(i))
                      for i in range(8)], ignore_index=True)
    data = pd.concat([data, data.iloc[[0]].assign(sample_id="duplicate")], ignore_index=True)
    folds, assignment = make_folds(data)
    assert assignment[0] == assignment[-1]
    for fit, held in folds:
        assert set(data.iloc[fit].prompt_text).isdisjoint(data.iloc[held].prompt_text)
    with pytest.raises(ValueError):
        check_disjoint(data.iloc[:1], data.iloc[-1:])


def test_quality_artifact_round_trip_and_prompt_only_inference(tmp_path):
    bundle = {"experiment": {"target": "quality"},
              "models": {family: {"small_score": FixedModel(.8), "large_score": FixedModel(.3)}
                         for family in ["tree", "svm"]},
              "selected_policy": {"tree_weight": .3, "threshold": -.1}}
    path = tmp_path / "quality.joblib"
    joblib.dump(bundle, path)
    output = route_prompts(joblib.load(path), ["A new prompt"])
    assert output["score"][0] == pytest.approx(-.5)
    assert output["route"].tolist() == [0]
    assert set(output["estimates"]) == {"small_score", "large_score"}
    for prompts in [[], [""], [None]]:
        with pytest.raises(ValueError):
            route_prompts(bundle, prompts)


def test_paired_comparison_identical_and_known_net_gain():
    data = example_data()
    identity = paired_comparison(data, [0, 0, 1, 0], [0, 0, 1, 0], repetitions=50)
    assert identity["answer_delta_95ci"] == [0, 0]
    assert identity["saving_delta_95ci"] == [0, 0]
    comparison = paired_comparison(data, [0, 0, 1, 0], [1, 1, 1, 1], repetitions=50)
    assert comparison["gained_correct"] == 1
    assert comparison["lost_correct"] == 0
    assert comparison["answer_delta"] == .25
    assert comparison["saving_delta"] == .5
