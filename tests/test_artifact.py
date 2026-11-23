"""Kiểm tra artifact sau khi đã chạy thí nghiệm, không huấn luyện trong test."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.router import DEFAULT_MODEL_PATH, load_router, route_prompts

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="Chạy train_models.py để tạo artifact")
def test_saved_model_reproduces_report_and_keeps_features_separate():
    bundle = load_router()
    selection = json.loads((ROOT / "reports/selection.json").read_text())
    assert bundle["selected_policy"] == selection["selected_policy"]
    tree = bundle["tree"].calibrated_classifiers_[0].estimator.named_steps["classifier"]
    assert tree.n_features_in_ == 3
    assert list(tree.feature_names_in_) == ["char_count", "word_count", "line_count"]
    svm = bundle["svm"].calibrated_classifiers_[0].estimator
    assert "tfidf" in svm.named_steps
    test = pd.read_csv(ROOT / "data/processed/routerbench_0shot_test.csv")
    saved = pd.read_csv(ROOT / "reports/test_predictions.csv")
    result = route_prompts(bundle, test["prompt_text"])
    np.testing.assert_allclose(result["probability"], saved["selected_probability"], rtol=1e-12)
    np.testing.assert_array_equal(result["route"], saved["route"])
