"""So sánh 3 đầu vào của cây; SVM đóng băng; chỉ đọc train và validation."""

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from time import perf_counter
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import select_policy, selected_outcomes
from src.embedding_experiment.common import (
    ENCODER_DIR, MAX_TOKENS, MODES, ROOT, SEED, VARIANTS, check_disjoint,
    load_split, run_dirs, sha256, split_path, write_json,
)
from src.embedding_experiment.encoder import FrozenEncoder, cached_embeddings
from src.embedding_experiment.inference import feature_matrix, positive_probability


def sweep(data, tree, svm, weights, name):
    rows = []
    for weight in weights:
        combined = weight * tree + (1 - weight) * svm
        for threshold in np.r_[np.linspace(0, 1, 101), 1.01]:
            routes = (combined >= threshold).astype(int)
            scores, costs = selected_outcomes(data, routes)
            rows.append({"name": name, "tree_weight": float(weight), "threshold": float(threshold),
                         "answer_score": float(scores.mean()), "mean_cost": float(costs.mean()),
                         "large_rate": float(routes.mean()),
                         "cost_saving": float(1 - costs.sum() / data.large_cost.sum())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="minilm_2026-11-25")
    args = parser.parse_args()
    reports, models_dir, cache_dir = run_dirs(args.run_name)
    if reports.exists() or models_dir.exists():
        parser.error("Run đã tồn tại; chọn tên mới để giữ nguyên kết quả cũ")
    baseline_path = ROOT / "models/router.joblib"
    baseline = joblib.load(baseline_path)
    for split in ["train", "validation"]:
        if sha256(split_path(split)) != baseline[f"{split}_sha256"]:
            raise ValueError(f"{split} khác dữ liệu baseline: không tái sử dụng SVM")
    if version("scikit-learn") != baseline["versions"]["sklearn"]:
        raise ValueError("Cần cùng phiên bản sklearn với baseline")
    train, validation = load_split("train"), load_split("validation")
    check_disjoint(train, validation)
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(
        train.prompt_text, train.label, groups=train.prompt_text))
    fold_ids = np.full(len(train), -1)
    for number, (fit, held) in enumerate(folds):
        check_disjoint(train.iloc[fit], train.iloc[held])
        if train.iloc[fit].label.nunique() != 2 or train.iloc[held].label.nunique() != 2:
            raise ValueError("Fold thiếu lớp")
        fold_ids[held] = number
    if (fold_ids < 0).any():
        raise ValueError("Có dòng chưa phân fold")
    reports.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    encoder_manifest = json.loads((ENCODER_DIR / "manifest.json").read_text())
    design = {
        "run_name": args.run_name, "started_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "variants": {"structure": 3, "embedding": 384, "combined": 387},
        "encoder": encoder_manifest, "max_tokens_per_chunk": MAX_TOKENS,
        "chunking": "nonoverlapping 254 content tokens, add CLS/SEP, no truncation",
        "pooling": "masked_mean_per_chunk_content_weighted_mean_then_l2_v1",
        "onnx_precision": "float32", "cpu_threads": 4, "batch_size": 32,
        "target": "original routing label: small wrong AND large correct",
        "baseline_sha256": sha256(baseline_path), "svm": "Frozen fitted v1 TF-IDF + LinearSVC, unchanged",
        "seed": SEED, "calibration": "sigmoid, 3 StratifiedGroupKFold on train, ensemble=False",
        "tree_grid": {"max_depth": [3, 5, 8, 12], "min_samples_leaf": [20, 50]},
        "model_selection": "minimum validation log_loss for each tree representation",
        "policy_modes": MODES, "thresholds": "0..1 step .01, plus 1.01 always Small",
        "primary_objective": "minimum validation cost with answer_score >= always_large - 0.02",
        "max_quality_drop": 0.02,
        "variant_selection": "lowest validation cost under primary constraint in tuned_router mode; ties prefer higher answer_score then declared variant order",
        "diagnostic_groups": ["hellaswag", "word_count >= 200", "all"],
        "diagnostics_status": "exploratory: inspired by previously inspected test errors",
        "test_status": "Previously inspected test. Follow-up analysis, not a new independent holdout.",
        "uncertainty": "paired prompt-group bootstrap, fixed models/policies, 1000 repetitions, seed 42; exploratory without multiple-comparison correction",
        "split_sha256": {name: sha256(split_path(name)) for name in ["train", "validation"]},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "src").rglob("*.py"))},
        "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "onnxruntime", "tokenizers", "huggingface-hub"]},
    }
    write_json(reports / "protocol.json", design)
    train[["sample_id"]].assign(calibration_fold=fold_ids).to_csv(reports / "train_folds.csv", index=False)
    started = perf_counter()
    encoder = FrozenEncoder()
    print("Đã ghi protocol. Tạo embedding train/validation; chưa đọc test.", flush=True)
    train_vectors, train_record = cached_embeddings(encoder, train, cache_dir, "train")
    validation_vectors, validation_record = cached_embeddings(encoder, validation, cache_dir, "validation")
    write_json(reports / "encoding_train_validation.json", {"train": train_record, "validation": validation_record})
    svm_probability = positive_probability(baseline["svm"], validation.prompt_text)
    minimum_score = float(validation.large_score.mean() - .02)
    trees, configs, policies, candidate_rows, sweep_rows, probability_rows = {}, {}, {}, [], [], []
    for variant in VARIANTS:
        x_train = feature_matrix(train.prompt_text, train_vectors, variant)
        x_validation = feature_matrix(validation.prompt_text, validation_vectors, variant)
        best_loss = float("inf")
        for depth in design["tree_grid"]["max_depth"]:
            for leaf in design["tree_grid"]["min_samples_leaf"]:
                fit_started = perf_counter()
                config = {"max_depth": depth, "min_samples_leaf": leaf}
                model = CalibratedClassifierCV(DecisionTreeClassifier(**config, random_state=SEED),
                                               method="sigmoid", cv=folds, ensemble=False, n_jobs=1)
                model.fit(x_train, train.label)
                probability = positive_probability(model, x_validation)
                loss = float(log_loss(validation.label, probability, labels=[0, 1]))
                candidate_rows.append({"variant": variant, **config, "validation_log_loss": loss,
                                       "validation_brier": float(brier_score_loss(validation.label, probability)),
                                       "fit_seconds": perf_counter() - fit_started})
                print(f"{variant}: depth={depth}, leaf={leaf}, log_loss={loss:.5f}", flush=True)
                if loss < best_loss:
                    best_loss, trees[variant], configs[variant] = loss, model, config
        tree_probability = positive_probability(trees[variant], x_validation)
        if variant == "structure":
            old = positive_probability(baseline["tree"], validation.prompt_text)
            np.testing.assert_allclose(tree_probability, old, atol=1e-12, rtol=1e-12)
        probability_rows.append(pd.DataFrame({"sample_id": validation.sample_id, "variant": variant,
                                              "tree_probability": tree_probability, "svm_probability": svm_probability}))
        policies[variant] = {}
        for mode, weights in MODES.items():
            rows = sweep(validation, tree_probability, svm_probability, weights, mode)
            selected = select_policy(rows, minimum_score)
            policies[variant][mode] = selected
            sweep_rows.extend({"variant": variant, **row} for row in rows)
            print(f"  {mode}: đúng {selected['answer_score']:.2%}, tiết kiệm {selected['cost_saving']:.2%}, "
                  f"alpha {selected['tree_weight']:.1f}, threshold {selected['threshold']:.2f}", flush=True)
    winner = min(VARIANTS, key=lambda name: (policies[name]["tuned_router"]["mean_cost"],
                                           -policies[name]["tuned_router"]["answer_score"]))
    bundle = {"trees": trees, "svm": baseline["svm"], "encoder": encoder_manifest,
              "policies": policies, "best_configs": configs, "selected_variant": winner,
              "protocol_sha256": sha256(reports / "protocol.json")}
    artifact = models_dir / "router.joblib"
    joblib.dump(bundle, artifact)
    pd.DataFrame(candidate_rows).to_csv(reports / "validation_candidates.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(reports / "validation_sweeps.csv", index=False)
    pd.concat(probability_rows).to_csv(reports / "validation_predictions.csv", index=False)
    write_json(reports / "selection.json", {"protocol_sha256": bundle["protocol_sha256"],
               "selected_variant": winner, "best_configs": configs, "policies": policies,
               "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": sha256(artifact),
               "training_and_encoding_seconds": perf_counter() - started})
    print(f"Đã khóa lựa chọn: {winner}. Chưa đọc test. Chạy evaluate để đánh giá.", flush=True)


if __name__ == "__main__":
    main()
