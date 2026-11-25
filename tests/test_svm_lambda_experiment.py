"""Kiểm tra trọng số trước normalization, mọi ngưỡng và tích phân budget envelope."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from src.svm_embedding_experiment.features import input_frame, SVMFeatures
from src.svm_lambda_experiment.features import WeightedSVMFeatures
from src.svm_lambda_experiment.curves import threshold_curve, best_for_budgets, normalized_area, frontier
from src.evaluation import selected_outcomes


@pytest.mark.parametrize('weight', [.25, .5, 1., 2.])
def test_weight_is_applied_before_joint_normalization(weight):
    embeddings = np.zeros((2, 384))
    embeddings[:, 0] = 1.
    frame = input_frame(['binary search', 'linear search'], embeddings)
    model = clone(WeightedSVMFeatures(embedding_weight=weight))
    matrix = model.fit_transform(frame)
    width = len(model.vectorizer_.vocabulary_)
    words = np.linalg.norm(matrix[:, :width].toarray(), axis=1)
    semantic = np.linalg.norm(matrix[:, width:].toarray(), axis=1)
    np.testing.assert_allclose(semantic / words, weight)
    np.testing.assert_allclose(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel(), 1.)
    if weight == 1:
        previous = SVMFeatures('combined').fit_transform(frame)
        np.testing.assert_array_equal(matrix.toarray(), previous.toarray())


def test_curves_match_brute_force_ties_endpoints_and_nonmonotone_costs():
    data = pd.DataFrame({'small_score': [1, 0, 0, 1], 'large_score': [0, 1, 1, 1],
                         'small_cost': [1., 1., 4., 1.], 'large_cost': [3., 3., 2., 3.]})
    scores = np.array([.2, .2, .8, .9])
    curve = threshold_curve(data, scores)
    for row in curve.itertuples():
        correct, costs = selected_outcomes(data, (scores >= row.threshold).astype(int))
        assert row.answer_score == pytest.approx(correct.mean())
        assert row.total_cost == pytest.approx(costs.sum())
    assert curve.iloc[0].large_rate == 1
    assert curve.iloc[-1].large_rate == 0
    assert not (curve.large_rate == .75).any()  # Không tách đôi score bằng nhau.
    assert (curve.cost_saving.diff().dropna() < 0).any()
    custom = threshold_curve(data, scores, [.0, .21, 1.01])
    assert custom.large_rate.tolist() == [1., .5, 0.]


def test_exact_step_area_and_policy_dominance():
    curve = pd.DataFrame({'threshold': [0., .2, .4, .6, 1.01],
                          'cost_saving': [0., .2, .3, .4, .6],
                          'answer_score': [.5, .6, .45, .4, .3]})
    # Q*(b): .6 on (0,.2], .45 on (.2,.3], .4 on (.3,.4], .3 on (.4,.6].
    assert normalized_area(curve, .6) == pytest.approx((.2*.6 + .1*.45 + .1*.4 + .2*.3)/.6)
    chosen = best_for_budgets(curve, [0., .2, .21, .6])
    assert chosen.threshold.tolist() == [.2, .2, .4, 1.01]
    assert 0. not in frontier(curve).threshold.tolist()
    with pytest.raises(ValueError):
        best_for_budgets(curve, [.7])
