"""Huấn luyện từng mục tiêu nhị phân; TF-IDF luôn nằm trong fold train."""

import json
from time import perf_counter

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.features import extract_structure_features
from src.experiments.config import N_FOLDS, SEED, SVM_C_VALUES, TREE_DEPTHS, TREE_LEAF_SIZES
from src.experiments.features import extract_rich_features, make_text_features


def make_folds(train):
    """Dùng cùng fold như v1 cho mọi mục tiêu để so sánh công bằng."""
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(train.prompt_text, train.label, groups=train.prompt_text))
    assignment = np.full(len(train), -1, dtype=int)
    for fold_id, (fit_indices, held_indices) in enumerate(folds):
        fit, held = train.iloc[fit_indices], train.iloc[held_indices]
        if not set(fit.prompt_text).isdisjoint(held.prompt_text):
            raise ValueError("Prompt bị trùng giữa các fold")
        for target in ["label", "small_score", "large_score"]:
            if fit[target].nunique() != 2 or held[target].nunique() != 2:
                raise ValueError(f"Fold {fold_id} thiếu một lớp của {target}")
        assignment[held_indices] = fold_id
    if (assignment < 0).any():
        raise ValueError("Có dòng chưa được phân fold")
    return folds, assignment


def candidates(family, profile):
    if family == "tree":
        feature_function = extract_structure_features if profile == "basic" else extract_rich_features
        for depth in TREE_DEPTHS:
            for leaf in TREE_LEAF_SIZES:
                config = {"max_depth": depth, "min_samples_leaf": leaf}
                yield config, Pipeline([
                    ("features", FunctionTransformer(feature_function)),
                    ("classifier", DecisionTreeClassifier(**config, random_state=SEED)),
                ])
    elif family == "svm":
        for c in SVM_C_VALUES:
            yield {"C": c}, Pipeline([
                ("tfidf", make_text_features(profile)),
                ("classifier", LinearSVC(C=c, max_iter=10000, random_state=SEED)),
            ])
    else:
        raise ValueError(f"Family không hợp lệ: {family}")


def positive_probability(model, prompts):
    column = list(model.classes_).index(1)
    return model.predict_proba(prompts)[:, column]


def fit_best(train, validation, folds, family, profile, target):
    best_model, best_config, best_loss = None, None, float("inf")
    rows = []
    for config, pipeline in candidates(family, profile):
        started = perf_counter()
        model = CalibratedClassifierCV(pipeline, method="sigmoid", cv=folds,
                                       ensemble=False, n_jobs=1)
        model.fit(train.prompt_text, train[target])
        probability = positive_probability(model, validation.prompt_text)
        loss = float(log_loss(validation[target], probability, labels=[0, 1]))
        rows.append({"family": family, "profile": profile, "target": target,
                     "config": json.dumps(config, sort_keys=True), "validation_log_loss": loss,
                     "validation_brier": float(brier_score_loss(validation[target], probability)),
                     "seconds": perf_counter() - started})
        print(f"  {family}/{profile}/{target} {config}: log_loss={loss:.5f}", flush=True)
        if loss < best_loss:
            best_model, best_config, best_loss = model, config, loss
    return best_model, best_config, rows
