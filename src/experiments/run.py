"""Chạy A/B/C/D trên train + validation, lưu khóa lựa chọn; KHÔNG đọc test.

Ví dụ: python -m src.experiments.run --run-name v2_2026-11-25
"""

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import platform
from time import perf_counter
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import sklearn
import numpy as np

from src.experiments.config import EXPERIMENTS, protocol
from src.experiments.data import ROOT, check_disjoint, load_split, sha256, split_path, write_json
from src.experiments.inference import predict_components
from src.experiments.models import fit_best, make_folds
from src.experiments.policy import select_policy, sweep_policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("v2_%Y%m%d_%H%M%S"))
    parser.add_argument("--target-saving", type=float, default=0.20)
    parser.add_argument("--max-quality-drop", type=float, default=0.02)
    args = parser.parse_args()
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name phải là một tên thư mục đơn")
    if not 0 <= args.target_saving <= 1 or not 0 <= args.max_quality_drop <= 1:
        parser.error("Các mục tiêu phải nằm trong [0, 1]")
    reports = ROOT / "reports" / "experiments" / args.run_name
    models_dir = ROOT / "models" / "experiments" / args.run_name
    if reports.exists() or models_dir.exists():
        parser.error("Run đã tồn tại; chọn tên mới để không ghi đè kết quả cũ")

    started = perf_counter()
    train, validation = load_split("train"), load_split("validation")
    check_disjoint(train, validation)
    if args.target_saving > 1 - validation.small_cost.sum() / validation.large_cost.sum():
        parser.error("Ngân sách quá thấp: ngay cả always Small cũng không đạt mục tiêu này")
    folds, assignment = make_folds(train)
    reports.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    design = protocol(args.target_saving, args.max_quality_drop)
    design.update({
        "run_name": args.run_name,
        "started_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "split_sha256": {name: sha256(split_path(name)) for name in ["train", "validation"]},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                          for path in sorted((ROOT / "src").rglob("*.py"))},
    })
    write_json(reports / "protocol.json", design)
    train[["sample_id"]].assign(calibration_fold=assignment).to_csv(reports / "train_folds.csv", index=False)
    print(f"Protocol đã ghi: {reports / 'protocol.json'}. Chỉ dùng train/validation.", flush=True)

    # Một bộ (family, profile, target) chỉ fit một lần, dùng lại giữa các ablation.
    cache, all_candidates, all_sweeps, policies, selections = {}, [], [], [], {}
    minimum_score = float(validation.large_score.mean() - args.max_quality_drop)
    for experiment in EXPERIMENTS:
        print(f"\n=== {experiment.name}: {experiment.description} ===", flush=True)
        models, configs = {"tree": {}, "svm": {}}, {}
        for family, profile in [("tree", experiment.tree_features), ("svm", experiment.text_features)]:
            for target in experiment.targets:
                key = (family, profile, target)
                if key not in cache:
                    model, config, rows = fit_best(train, validation, folds, family, profile, target)
                    cache[key] = (model, config)
                    all_candidates.extend(rows)
                    pd.DataFrame(all_candidates).to_csv(reports / "validation_candidates.csv", index=False)
                models[family][target], configs[f"{family}_{target}"] = cache[key]
        bundle = {"experiment": asdict(experiment), "models": models, "best_configs": configs}
        components = predict_components(bundle, validation.prompt_text)
        table = sweep_policies(validation, components, experiment.target)
        all_sweeps.append(table.assign(experiment=experiment.name))
        experiment_policies = {}
        for objective in ["fixed_budget", "quality_constraint"]:
            chosen = select_policy(table, objective, args.target_saving, minimum_score)
            experiment_policies[objective] = chosen
            # Báo cáo cả từng thuật toán lẫn ensemble, không giấu thành phần yếu.
            for family in ["Decision Tree", "SVM", "Ensemble"]:
                family_policy = select_policy(table.loc[table.name == family], objective,
                                              args.target_saving, minimum_score)
                policies.append({"experiment": experiment.name, "objective": objective,
                                 **family_policy, "selected_in_experiment": family_policy == chosen})
        bundle.update({"selected_policy": experiment_policies["fixed_budget"],
                       "policies": experiment_policies, "protocol_sha256": sha256(reports / "protocol.json")})
        artifact = models_dir / f"{experiment.name}.joblib"
        joblib.dump(bundle, artifact)
        selections[experiment.name] = {"experiment": asdict(experiment), "best_configs": configs,
                                      "policies": experiment_policies,
                                      "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": sha256(artifact)}
        chosen = bundle["selected_policy"]
        print(f"  Khóa {experiment.name}: đúng={chosen['answer_score']:.2%}, "
              f"tiết kiệm={chosen['cost_saving']:.2%}, alpha={chosen['tree_weight']:.1f}, "
              f"ngưỡng={chosen['threshold']:.2f}", flush=True)

    # Python min giữ thứ tự EXPERIMENTS nếu tất cả khóa so sánh bằng nhau.
    winner = min(selections, key=lambda name: (
        -selections[name]["policies"]["fixed_budget"]["answer_score"],
        selections[name]["policies"]["fixed_budget"]["mean_cost"],
    ))
    pd.concat(all_sweeps, ignore_index=True).to_csv(reports / "validation_sweeps.csv", index=False)
    pd.DataFrame(policies).to_csv(reports / "validation_policies.csv", index=False)
    write_json(reports / "selection.json", {
        "protocol_sha256": sha256(reports / "protocol.json"), "experiments": selections,
        "selected_experiment": winner, "selection_split": "validation",
        "training_seconds": perf_counter() - started,
    })
    print(f"\nĐã khóa toàn bộ thí nghiệm. Chọn {winner} bằng validation. Chưa đọc test.", flush=True)
    print(f"Đánh giá: .venv/bin/python -m src.experiments.evaluate --run-name {args.run_name}", flush=True)


if __name__ == "__main__":
    main()
