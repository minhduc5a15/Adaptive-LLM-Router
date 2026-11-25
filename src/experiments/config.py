"""Thiết kế thí nghiệm được ghi ra protocol.json trước khi huấn luyện."""

from dataclasses import asdict, dataclass

import numpy as np

SEED = 42
N_FOLDS = 3
TREE_DEPTHS = [3, 5, 8, 12]
TREE_LEAF_SIZES = [20, 50]
SVM_C_VALUES = [0.1, 1.0, 10.0]
TREE_WEIGHTS = [float(x) for x in np.linspace(0, 1, 11)]


@dataclass(frozen=True)
class Experiment:
    name: str
    tree_features: str
    text_features: str
    target: str
    description: str

    @property
    def targets(self):
        return ["label"] if self.target == "routing" else ["small_score", "large_score"]


# Hai bản *_word tách ảnh hưởng của character n-gram khỏi word n-gram.
EXPERIMENTS = [
    Experiment("A", "basic", "basic", "routing", "Mốc v1: ba đặc trưng cây, TF-IDF 5000 từ"),
    Experiment("B_word", "rich", "word", "routing", "Thêm đặc trưng cây và word n-gram"),
    Experiment("B", "rich", "word_char", "routing", "Thêm cả character n-gram"),
    Experiment("C", "basic", "basic", "quality", "Chỉ đổi sang dự đoán độ đúng từng LLM"),
    Experiment("D_word", "rich", "word", "quality", "Mục tiêu mới + đặc trưng, chưa thêm char"),
    Experiment("D", "rich", "word_char", "quality", "Kết hợp cả hai cải tiến"),
]


def protocol(target_saving, max_quality_drop):
    return {
        "seed": SEED,
        "experiments": [asdict(experiment) for experiment in EXPERIMENTS],
        "primary_objective": "Max validation answer_score subject to cost_saving >= target_saving",
        "target_saving": target_saving,
        "secondary_objective": "Min validation mean_cost subject to answer_score >= always_large - max_quality_drop",
        "max_quality_drop": max_quality_drop,
        "model_selection": "Minimum validation log_loss per family, feature profile and target",
        "calibration": {"method": "sigmoid", "folds": N_FOLDS, "ensemble": False,
                        "groups": "prompt_text", "stratify": "original routing label; same folds for all targets"},
        "tree_grid": {"max_depth": TREE_DEPTHS, "min_samples_leaf": TREE_LEAF_SIZES},
        "svm_grid": {"C": SVM_C_VALUES, "max_iter": 10000},
        "text_profiles": {
            "basic": {"word_max_features": 5000, "word_ngram": [1, 1], "min_df": 1},
            "word": {"word_max_features": 15000, "word_ngram": [1, 2], "min_df": 3},
            "word_char": {"word_max_features": 15000, "word_ngram": [1, 2], "min_df": 3,
                          "char_max_features": 20000, "char_analyzer": "char_wb", "char_ngram": [3, 5]},
        },
        "tree_weights": TREE_WEIGHTS,
        "thresholds": {"routing": "0..1 step 0.01, plus 1.01 for always Small",
                       "quality": "-1..1 step 0.01, plus 1.01 for always Small"},
        "quality_score": "P(large correct) - P(small correct); NOT a probability",
        "tie_break": "Primary: higher answer_score, lower cost, lower weight, lower threshold; final experiment order as declared",
        "cost_at_inference": "No per-prompt cost input. Fixed threshold selected using validation costs only.",
        "test_status": "Previously inspected test; follow-up descriptive evaluation, NOT a new independent holdout",
        "uncertainty": "Paired prompt-group bootstrap, 1000 repetitions, fixed fitted models/policies; descriptive intervals",
    }
