"""Chỉ đọc train/validation, thử 9 cấu hình SVM rồi khóa lựa chọn."""
import argparse
import json
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.embedding_experiment.common import (
    ROOT, SEED, load_split, split_path, sha256, write_json, check_disjoint, run_dirs,
)
from src.embedding_experiment.encoder import FrozenEncoder, cached_embeddings
from src.embedding_experiment.inference import positive_probability
from src.embedding_experiment.train import sweep
from src.evaluation import select_policy
from src.svm_embedding_experiment.common import DEFAULT_RUN, CACHE_RUN, VARIANTS, MODES, paths, source_hashes
from src.svm_embedding_experiment.features import SVMFeatures, input_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    args = parser.parse_args()
    reports, models = paths(args.run_name)
    if reports.exists() or models.exists():
        parser.error("Run đã tồn tại; chọn tên mới")
    baseline_path = ROOT / "models/router.joblib"
    baseline = joblib.load(baseline_path)
    for split in ["train", "validation"]:
        if sha256(split_path(split)) != baseline[f"{split}_sha256"]:
            raise ValueError("Dữ liệu khác baseline")
    if version("scikit-learn") != baseline["versions"]["sklearn"]:
        raise ValueError("Phiên bản sklearn khác baseline")
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
    encoder = FrozenEncoder()
    protocol = {
        "run_name": args.run_name, "started_at": datetime.now(timezone.utc).isoformat(),
        "variants": VARIANTS, "C_grid": [0.1, 1.0, 10.0], "max_iter": 10000,
        "tfidf": {"max_features": 5000}, "combined": "equal block weights, concatenate then L2 normalize",
        "encoder": encoder.manifest, "cache_run": CACHE_RUN,
        "pooling": "unchanged from tree experiment: all content tokens, 254-token chunks, masked mean, content-weighted mean, L2",
        "baseline_sha256": sha256(baseline_path), "tree": "frozen v1 structural tree, unchanged",
        "target": "small_score == 0 AND large_score == 1", "seed": SEED,
        "calibration": "sigmoid, 3 StratifiedGroupKFold, ensemble=False; TF-IDF inside pipeline",
        "model_selection": "minimum validation log_loss per variant; ties prefer smaller C",
        "policy_modes": MODES, "thresholds": "0..1 step .01 plus 1.01",
        "primary_objective": "minimum validation cost with answer_score >= always_large - .02",
        "variant_selection": "minimum tuned_router validation cost, then higher answer score, then declared order",
        "diagnostic_groups": ["all", "hellaswag", "word_count_ge_200"],
        "test_status": "Previously inspected test; exploratory follow-up, not an independent holdout",
        "bootstrap": "1000 paired prompt-group resamples, seed42, fixed models/policies, no multiplicity correction",
        "split_sha256": {s: sha256(split_path(s)) for s in ["train", "validation"]},
        "source_sha256": source_hashes(),
        "versions": {n: version(n) for n in ["numpy", "pandas", "scikit-learn", "scipy", "onnxruntime", "tokenizers"]},
    }
    reports.mkdir(parents=True)
    models.mkdir(parents=True)
    write_json(reports / "protocol.json", protocol)
    train[["sample_id"]].assign(calibration_fold=fold_ids).to_csv(reports / "train_folds.csv", index=False)
    print("Đã ghi protocol. Đọc cache train/validation; chưa đọc test.", flush=True)
    started = perf_counter()
    _, _, cache = run_dirs(CACHE_RUN)
    train_vectors, train_record = cached_embeddings(encoder, train, cache, "train")
    val_vectors, val_record = cached_embeddings(encoder, validation, cache, "validation")
    write_json(reports / "encoding_train_validation.json", {"train": train_record, "validation": val_record,
               "note": "Cache reused; stats.seconds records original encoding time, not this run."})
    X_train, X_val = input_frame(train.prompt_text, train_vectors), input_frame(validation.prompt_text, val_vectors)
    tree_probability = positive_probability(baseline["tree"], validation.prompt_text)
    svms, configs, policies, candidates, sweeps, predictions = {}, {}, {}, [], [], []
    warnings.filterwarnings("error", category=ConvergenceWarning)
    for variant in VARIANTS:
        best = float("inf")
        for c in protocol["C_grid"]:
            fit_started = perf_counter()
            estimator = Pipeline([("features", SVMFeatures(variant)),
                                  ("classifier", LinearSVC(C=c, max_iter=10000, random_state=SEED))])
            model = CalibratedClassifierCV(estimator, method="sigmoid", cv=folds, ensemble=False, n_jobs=1)
            model.fit(X_train, train.label)
            probability = positive_probability(model, X_val)
            loss = float(log_loss(validation.label, probability, labels=[0, 1]))
            candidates.append({"variant": variant, "C": c, "log_loss": loss,
                               "brier": float(brier_score_loss(validation.label, probability)),
                               "fit_seconds": perf_counter() - fit_started})
            print(f"{variant}: C={c:g}, log_loss={loss:.6f}", flush=True)
            if loss < best:
                best, svms[variant], configs[variant] = loss, model, {"C": c}
        probability = positive_probability(svms[variant], X_val)
        if variant == "tfidf":
            np.testing.assert_allclose(probability, positive_probability(baseline["svm"], validation.prompt_text),
                                       atol=1e-12, rtol=1e-12)
        predictions.append(pd.DataFrame({"sample_id": validation.sample_id, "variant": variant,
                                          "tree_probability": tree_probability, "svm_probability": probability}))
        policies[variant] = {}
        for mode, weights in MODES.items():
            rows = sweep(validation, tree_probability, probability, weights, mode)
            selected = select_policy(rows, float(validation.large_score.mean() - .02))
            policies[variant][mode] = selected
            sweeps.extend({"variant": variant, **r} for r in rows)
            print(f"  {mode}: đúng {selected['answer_score']:.2%}, tiết kiệm {selected['cost_saving']:.2%}, "
                  f"tree_weight={selected['tree_weight']:.1f}, threshold={selected['threshold']:.2f}", flush=True)
    winner = min(VARIANTS, key=lambda v: (policies[v]["tuned_router"]["mean_cost"], -policies[v]["tuned_router"]["answer_score"]))
    bundle = {"svms": svms, "tree": baseline["tree"], "encoder": encoder.manifest,
              "policies": policies, "best_configs": configs, "selected_variant": winner,
              "protocol_sha256": sha256(reports / "protocol.json")}
    artifact = models / "router.joblib"
    joblib.dump(bundle, artifact)
    pd.DataFrame(candidates).to_csv(reports / "validation_candidates.csv", index=False)
    pd.DataFrame(sweeps).to_csv(reports / "validation_sweeps.csv", index=False)
    pd.concat(predictions).to_csv(reports / "validation_predictions.csv", index=False)
    write_json(reports / "selection.json", {"selected_variant": winner, "policies": policies,
               "best_configs": configs, "protocol_sha256": bundle["protocol_sha256"],
               "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": sha256(artifact),
               "training_seconds": perf_counter() - started})
    print(f"Đã khóa lựa chọn: {winner}. Chưa đọc test.", flush=True)


if __name__ == "__main__":
    main()
