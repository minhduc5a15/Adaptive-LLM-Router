"""Đánh giá các quyết định đã khóa. Test cũ chỉ là phân tích tiếp theo."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.evaluation import selected_outcomes
from src.experiments.config import SEED
from src.experiments.data import ROOT, check_disjoint, load_split, sha256, split_path, write_json
from src.experiments.inference import combine_scores, predict_components
from src.experiments.policy import summarize_routes


def paired_comparison(data, routes, reference_routes, repetitions=1000):
    """CI ghép cặp: cùng mẫu bootstrap cho hai router, cùng nhóm prompt.

    Đây là độ bất định của tập đánh giá với mô hình đã cố định, không bao gồm
    độ bất định từ huấn luyện/chọn cấu hình. CI không sửa được việc đã xem test.
    """
    score, cost = selected_outcomes(data, routes)
    reference_score, reference_cost = selected_outcomes(data, reference_routes)
    _, inverse = np.unique(data.prompt_text, return_inverse=True)
    sizes = np.bincount(inverse)
    score_diff = np.bincount(inverse, weights=score - reference_score)
    cost_diff = np.bincount(inverse, weights=reference_cost - cost)
    large_cost = np.bincount(inverse, weights=data.large_cost)
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(repetitions):
        sample = rng.integers(0, len(sizes), len(sizes))
        estimates.append([score_diff[sample].sum() / sizes[sample].sum(),
                          cost_diff[sample].sum() / large_cost[sample].sum()])
    intervals = np.quantile(estimates, [.025, .975], axis=0)
    return {
        "answer_delta": float((score - reference_score).mean()),
        "answer_delta_95ci": intervals[:, 0].tolist(),
        "saving_delta": float((reference_cost - cost).sum() / data.large_cost.sum()),
        "saving_delta_95ci": intervals[:, 1].tolist(),
        "gained_correct": int(((score == 1) & (reference_score == 0)).sum()),
        "lost_correct": int(((score == 0) & (reference_score == 1)).sum()),
        "repetitions": repetitions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name phải là một tên thư mục đơn")
    reports = ROOT / "reports" / "experiments" / args.run_name
    design = json.loads((reports / "protocol.json").read_text())
    selection = json.loads((reports / "selection.json").read_text())
    if sha256(reports / "protocol.json") != selection["protocol_sha256"]:
        raise ValueError("Protocol đã thay đổi sau khi khóa lựa chọn")
    for name, expected in design["split_sha256"].items():
        if sha256(split_path(name)) != expected:
            raise ValueError(f"Dữ liệu {name} đã thay đổi; không đánh giá artifact cũ")
    # Kiểm tra và nạp tất cả artifact trước khi đọc test; không fit/select ở đây.
    bundles = {}
    for name, info in selection["experiments"].items():
        path = ROOT / info["artifact"]
        if sha256(path) != info["artifact_sha256"]:
            raise ValueError(f"Artifact {name} đã thay đổi")
        bundles[name] = joblib.load(path)
        if bundles[name]["policies"] != info["policies"]:
            raise ValueError(f"Chính sách {name} không khớp selection.json")

    train, validation, test = [load_split(name) for name in ["train", "validation", "test"]]
    check_disjoint(train, test)
    check_disjoint(validation, test)
    rows, prediction_frames, calibration_rows, benchmark_rows = [], [], [], []
    test_routes = {}
    for split_name, data in [("validation", validation), ("test", test)]:
        for name, routes in [("Always Small", np.zeros(len(data), dtype=int)),
                             ("Always Large", np.ones(len(data), dtype=int)),
                             ("Oracle", data.label.to_numpy())]:
            rows.append({"split": split_name, "experiment": name, "objective": "baseline",
                         **summarize_routes(data, routes)})
        for name, bundle in bundles.items():
            components = predict_components(bundle, data.prompt_text)
            # Xác suất được chấm theo đúng target; tuyệt đối không tính log_loss cho gain.
            for component, probability in components.items():
                target = component.split("_", 1)[1]
                calibration_rows.append({"split": split_name, "experiment": name,
                                         "component": component, "target": target,
                                         "log_loss": log_loss(data[target], probability, labels=[0, 1]),
                                         "brier": brier_score_loss(data[target], probability)})
            for objective, policy in bundle["policies"].items():
                score, mixed = combine_scores(components, bundle["experiment"]["target"], policy["tree_weight"])
                routes = (score >= policy["threshold"]).astype(int)
                metrics = summarize_routes(data, routes)
                rows.append({"split": split_name, "experiment": name, "objective": objective,
                             "family": policy["name"], "tree_weight": policy["tree_weight"],
                             "threshold": policy["threshold"], **metrics})
                if split_name == "validation":
                    for key in ["answer_score", "mean_cost", "large_rate"]:
                        if not np.isclose(metrics[key], policy[key], rtol=1e-12, atol=1e-12):
                            raise ValueError(f"{name}: artifact không tái tạo kết quả validation")
                if objective != "fixed_budget":
                    continue
                for target, probability in mixed.items():
                    calibration_rows.append({"split": split_name, "experiment": name,
                                             "component": f"combined_{target}", "target": target,
                                             "log_loss": log_loss(data[target], probability, labels=[0, 1]),
                                             "brier": brier_score_loss(data[target], probability)})
                if split_name == "test":
                    test_routes[name] = routes
                    selected_score, selected_cost = selected_outcomes(data, routes)
                    predictions = data[["sample_id", "eval_name", "small_score", "large_score"]].copy()
                    predictions = predictions.assign(experiment=name, routing_score=score, route=routes,
                                                     selected_score=selected_score, selected_cost=selected_cost)
                    for component, probability in components.items():
                        predictions[component] = probability
                    prediction_frames.append(predictions)
                    for benchmark, indices in data.groupby("eval_name").groups.items():
                        benchmark_rows.append({"experiment": name, "eval_name": benchmark,
                                               **summarize_routes(data.loc[indices], routes[indices])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(reports / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(reports / "test_predictions.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(reports / "probability_metrics.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(reports / "test_by_benchmark.csv", index=False)
    winner = selection["selected_experiment"]
    summary = {
        "selected_experiment": winner, "selection_split": "validation",
        "test_status": design["test_status"], "test_sha256": sha256(split_path("test")),
        "selection_sha256": sha256(reports / "selection.json"),
        "evaluation_source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                                     for path in sorted((ROOT / "src").rglob("*.py"))},
        "exploratory_comparisons_vs_A": {
            name: paired_comparison(test, routes, test_routes["A"])
            for name, routes in test_routes.items() if name != "A"
        },
        "selected_vs_A": paired_comparison(test, test_routes[winner], test_routes["A"]),
        "selected_vs_always_large": paired_comparison(test, test_routes[winner], np.ones(len(test), dtype=int)),
    }
    write_json(reports / "summary.json", summary)
    from src.experiments.reporting import write_report
    write_report(reports, design, selection, metrics, summary)
    print(metrics.loc[(metrics.split == "test") & (metrics.objective == "fixed_budget"),
                      ["experiment", "correct_count", "answer_score", "cost_saving", "large_rate"]].to_string(index=False))
    print(f"\nChọn từ validation: {winner}. Báo cáo: {reports / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
