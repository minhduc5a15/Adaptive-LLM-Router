"""Huấn luyện trên train, chọn trên validation, khóa cấu hình rồi mới đánh giá test."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

PROJECT_DIR = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_DIR))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.features import extract_structure_features
from src.evaluation import select_policy, sweep_thresholds
from src.router import combine_probabilities, predict_probabilities

DATA_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports"
MODEL_PATH = PROJECT_DIR / "models" / "router.joblib"
SEED = 42


def load_split(name):
    return pd.read_csv(DATA_DIR / f"routerbench_0shot_{name}.csv")


def split_hash(name):
    return hashlib.sha256((DATA_DIR / f"routerbench_0shot_{name}.csv").read_bytes()).hexdigest()


def make_candidates():
    for depth in [3, 5, 8, 12]:
        for leaf_size in [20, 50]:
            config = {"max_depth": depth, "min_samples_leaf": leaf_size}
            yield "tree", config, Pipeline([
                ("features", FunctionTransformer(extract_structure_features)),
                ("classifier", DecisionTreeClassifier(**config, random_state=SEED)),
            ])
    for c in [0.1, 1.0, 10.0]:
        config = {"C": c, "max_features": 5000}
        yield "svm", config, Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("classifier", LinearSVC(C=c, max_iter=10000, random_state=SEED)),
        ])


def fit_candidates(train, validation):
    # Cùng prompt phải ở cùng fold, kể cả trong bước hiệu chỉnh xác suất.
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = list(splitter.split(train["prompt_text"], train["label"], groups=train["prompt_text"]))
    for fit_indices, calibration_indices in folds:
        assert set(train.iloc[fit_indices]["prompt_text"]).isdisjoint(
            train.iloc[calibration_indices]["prompt_text"])

    best_models, best_losses, best_configs, rows = {}, {}, {}, []
    for family, config, pipeline in make_candidates():
        started = perf_counter()
        # ensemble=False: hiệu chỉnh từ dự đoán ngoài fold, sau đó fit mô hình
        # cơ sở trên toàn train. TF-IDF nằm TRONG pipeline nên fit riêng từng fold.
        calibrated = CalibratedClassifierCV(
            pipeline, method="sigmoid", cv=folds, ensemble=False, n_jobs=1,
        )
        calibrated.fit(train["prompt_text"], train["label"])
        probabilities = calibrated.predict_proba(validation["prompt_text"])[:, 1]
        loss = float(log_loss(validation["label"], probabilities))
        rows.append({
            "family": family, "config": json.dumps(config), "validation_log_loss": loss,
            "training_seconds": perf_counter() - started,
        })
        print(f'{family}: {config}, validation log loss={loss:.5f}', flush=True)
        if loss < best_losses.get(family, float("inf")):
            best_models[family], best_losses[family], best_configs[family] = calibrated, loss, config
    return best_models, best_configs, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-quality-drop", type=float, default=0.02,
                        help="Mức giảm điểm cho phép trên validation; 0.02 = 2 điểm phần trăm")
    args = parser.parse_args()
    if not 0 <= args.max_quality_drop <= 1:
        parser.error("--max-quality-drop phải nằm trong [0, 1]")
    started = perf_counter()
    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    train, validation = load_split("train"), load_split("validation")
    assert set(train["prompt_text"]).isdisjoint(validation["prompt_text"])
    print(f'Train: {len(train):,}; validation: {len(validation):,}. Chưa nạp test.', flush=True)
    models, configs, candidates = fit_candidates(train, validation)
    candidates.to_csv(REPORT_DIR / "validation_candidates.csv", index=False)
    probabilities = predict_probabilities(models, validation["prompt_text"])

    sweeps, policies = [], []
    minimum_score = float(validation["large_score"].mean() - args.max_quality_drop)
    for name, weights in [("Decision Tree", [1.0]), ("SVM", [0.0]),
                          ("Ensemble", np.linspace(0.1, 0.9, 9))]:
        family_rows = []
        for weight in weights:
            combined = combine_probabilities(probabilities, weight)
            family_rows.extend(sweep_thresholds(validation, combined, weight, name))
        sweeps.extend(family_rows)
        policies.append(select_policy(family_rows, minimum_score))
    selected = select_policy(policies, minimum_score)
    pd.DataFrame(sweeps).to_csv(REPORT_DIR / "validation_thresholds.csv", index=False)

    selection = {
        "seed": SEED, "max_quality_drop": args.max_quality_drop,
        "validation_minimum_answer_score": minimum_score,
        "model_selection": "Minimum validation log loss after grouped 3-fold sigmoid calibration",
        "policy_selection": "Minimum validation cost subject to answer-score constraint",
        "best_configs": configs, "policies": policies, "selected_policy": selected,
        "train_sha256": split_hash("train"), "validation_sha256": split_hash("validation"),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "training_and_selection_seconds": perf_counter() - started,
    }
    # Lưu quyết định và mô hình TRƯỚC khi nạp test. Không refit bằng validation.
    bundle = {**models, **selection}
    joblib.dump(bundle, MODEL_PATH)
    (REPORT_DIR / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f'Đã khóa cấu hình: {selected["name"]}, alpha={selected["tree_weight"]:.2f}, '
          f'ngưỡng={selected["threshold"]:.2f}. Bắt đầu đánh giá test.', flush=True)

    # Nạp lại artifact để kiểm tra chính mô hình được sử dụng trong CLI/demo.
    bundle = joblib.load(MODEL_PATH)
    test = load_split("test")
    assert set(test["prompt_text"]).isdisjoint(train["prompt_text"])
    assert set(test["prompt_text"]).isdisjoint(validation["prompt_text"])
    from src.reporting import evaluate_and_report
    evaluate_and_report(bundle, train, validation, test, REPORT_DIR, split_hash("test"))


if __name__ == "__main__":
    main()
