"""Đánh giá artifact đã khóa; phân tích lỗi theo nhóm, không chọn lại bằng test."""

import argparse
import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from src.evaluation import routing_metrics, selected_outcomes
from src.features import extract_structure_features
from src.embedding_experiment.common import (
    MODES, ROOT, SEED, VARIANTS, check_disjoint, load_split, run_dirs,
    sha256, split_path, write_json,
)
from src.embedding_experiment.encoder import FrozenEncoder, cached_embeddings
from src.embedding_experiment.inference import feature_matrix, positive_probability, route_scores


def error_counts(data, routes):
    label = data.label.to_numpy()
    fn = (label == 1) & (routes == 0)
    fp = (label == 0) & (routes == 1)
    return {"n_samples": len(data), "label1_count": int(label.sum()),
            "fn": int(fn.sum()), "fp": int(fp.sum()),
            "fn_rate": float(fn.sum() / label.sum()) if label.sum() else None,
            "fp_rate": float(fp.sum() / (label == 0).sum()) if (label == 0).any() else None,
            "fp_both_wrong": int((fp & (data.small_score == 0) & (data.large_score == 0)).sum()),
            "fp_both_correct": int((fp & (data.small_score == 1) & (data.large_score == 1)).sum()),
            "fp_harmful": int((fp & (data.small_score == 1) & (data.large_score == 0)).sum())}


def compare_paired(data, routes, reference, repetitions=1000):
    scores, costs = selected_outcomes(data, routes)
    ref_scores, ref_costs = selected_outcomes(data, reference)
    _, groups = np.unique(data.prompt_text, return_inverse=True)
    sizes = np.bincount(groups)
    differences = np.bincount(groups, weights=scores - ref_scores)
    saved_costs = np.bincount(groups, weights=ref_costs - costs)
    large_costs = np.bincount(groups, weights=data.large_cost)
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(repetitions):
        sample = rng.integers(0, len(sizes), len(sizes))
        estimates.append([differences[sample].sum() / sizes[sample].sum(),
                          saved_costs[sample].sum() / large_costs[sample].sum()])
    ci = np.quantile(estimates, [.025, .975], axis=0)
    return {"answer_delta": float((scores - ref_scores).mean()), "answer_delta_95ci": ci[:, 0].tolist(),
            "saving_delta": float((ref_costs - costs).sum() / data.large_cost.sum()),
            "saving_delta_95ci": ci[:, 1].tolist(),
            "gained_correct": int(((scores == 1) & (ref_scores == 0)).sum()),
            "lost_correct": int(((scores == 0) & (ref_scores == 1)).sum()), "repetitions": repetitions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="minilm_2026-11-25")
    args = parser.parse_args()
    reports, _, cache_dir = run_dirs(args.run_name)
    design = json.loads((reports / "protocol.json").read_text())
    selection = json.loads((reports / "selection.json").read_text())
    if sha256(reports / "protocol.json") != selection["protocol_sha256"]:
        raise ValueError("Protocol đã thay đổi")
    for split, expected in design["split_sha256"].items():
        if sha256(split_path(split)) != expected:
            raise ValueError(f"{split} đã thay đổi")
    artifact = ROOT / selection["artifact"]
    if sha256(artifact) != selection["artifact_sha256"]:
        raise ValueError("Artifact đã thay đổi")
    bundle = joblib.load(artifact)
    if bundle["policies"] != selection["policies"] or bundle["protocol_sha256"] != selection["protocol_sha256"]:
        raise ValueError("Artifact và selection không khớp")
    # Nếu đánh giá lại chỉ để sửa báo cáo, không cho âm thầm thay bộ test.
    if (reports / "summary.json").exists():
        previous = json.loads((reports / "summary.json").read_text())
        if previous["test_sha256"] != sha256(split_path("test")):
            raise ValueError("Test đã đổi; cần run đánh giá mới")
    train, validation, test = [load_split(name) for name in ["train", "validation", "test"]]
    check_disjoint(train, test)
    check_disjoint(validation, test)
    encoder = FrozenEncoder()
    if encoder.manifest != bundle["encoder"]:
        raise ValueError("Encoder không khớp artifact")
    # Chỉ đến đây mới encode test: chính sách đã khóa trên đĩa.
    encoding_records, metric_rows, group_rows, predictions, timings = {}, [], [], [], []
    routes_by_split = {}
    for split, data in [("validation", validation), ("test", test)]:
        vectors, record = cached_embeddings(encoder, data, cache_dir, split)
        encoding_records[split] = record
        started = perf_counter()
        svm_probability = positive_probability(bundle["svm"], data.prompt_text)
        timings.append({"split": split, "component": "frozen_svm", "seconds": perf_counter() - started,
                        "n_samples": len(data)})
        structure = extract_structure_features(data.prompt_text)
        group_masks = {"all": np.ones(len(data), dtype=bool),
                       "hellaswag": data.eval_name.eq("hellaswag").to_numpy(),
                       "word_count_ge_200": structure.word_count.ge(200).to_numpy()}
        for benchmark in data.eval_name.unique():
            group_masks[f"benchmark:{benchmark}"] = data.eval_name.eq(benchmark).to_numpy()
        routes_by_split[split] = {}
        for name, routes in [("Always Small", np.zeros(len(data), dtype=int)),
                             ("Always Large", np.ones(len(data), dtype=int)), ("Oracle", data.label.to_numpy())]:
            metric_rows.append({"split": split, "variant": name, "mode": "baseline",
                                "correct_count": int(selected_outcomes(data, routes)[0].sum()),
                                **routing_metrics(data, routes)})
        for variant in VARIANTS:
            matrix = feature_matrix(data.prompt_text, vectors, variant)
            started = perf_counter()
            tree_probability = positive_probability(bundle["trees"][variant], matrix)
            timings.append({"split": split, "component": f"tree_{variant}",
                            "seconds": perf_counter() - started, "n_samples": len(data)})
            for mode in MODES:
                policy = bundle["policies"][variant][mode]
                probability, routes = route_scores(tree_probability, svm_probability, policy)
                metrics = routing_metrics(data, routes, probability)
                if split == "validation":
                    for key in ["answer_score", "mean_cost", "large_rate"]:
                        if not np.isclose(metrics[key], policy[key], rtol=1e-12, atol=1e-12):
                            raise ValueError("Nạp artifact không tái tạo kết quả validation")
                selected_score, selected_cost = selected_outcomes(data, routes)
                metric_rows.append({"split": split, "variant": variant, "mode": mode,
                                    "tree_weight": policy["tree_weight"], "threshold": policy["threshold"],
                                    "correct_count": int(selected_score.sum()), **metrics})
                routes_by_split[split][(variant, mode)] = routes
                for group, mask in group_masks.items():
                    group_rows.append({"split": split, "variant": variant, "mode": mode, "group": group,
                                       **error_counts(data.loc[mask], routes[mask]),
                                       "answer_score": float(selected_score[mask].mean()) if mask.any() else None})
                frame = data[["sample_id", "eval_name", "label", "small_score", "large_score"]].copy()
                frame = frame.assign(split=split, variant=variant, mode=mode,
                                     tree_probability=tree_probability, svm_probability=svm_probability,
                                     probability=probability, route=routes,
                                     selected_score=selected_score, selected_cost=selected_cost)
                predictions.append(frame)
    comparisons = {variant: {mode: compare_paired(test, routes_by_split["test"][(variant, mode)],
                                                 routes_by_split["test"][("structure", mode)])
                             for mode in MODES} for variant in ["embedding", "combined"]}
    metrics = pd.DataFrame(metric_rows)
    groups = pd.DataFrame(group_rows)
    metrics.to_csv(reports / "metrics.csv", index=False)
    groups.to_csv(reports / "group_errors.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(reports / "predictions.csv", index=False)
    pd.DataFrame(timings).to_csv(reports / "inference_timings.csv", index=False)
    write_json(reports / "encoding_evaluation.json", encoding_records)
    summary = {"selected_variant": selection["selected_variant"], "test_status": design["test_status"],
               "test_sha256": sha256(split_path("test")), "selection_sha256": sha256(reports / "selection.json"),
               "exploratory_paired_comparisons": comparisons,
               "evaluation_source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))}}
    write_json(reports / "summary.json", summary)
    # Ví dụ lỗi trên validation: mỗi bản/mode lấy tối đa 5 câu cải thiện và 5 câu mất đúng.
    examples = []
    data = validation
    for variant in ["embedding", "combined"]:
        for mode in ["fixed_ensemble", "tuned_router"]:
            score = selected_outcomes(data, routes_by_split["validation"][(variant, mode)])[0]
            reference = selected_outcomes(data, routes_by_split["validation"][("structure", mode)])[0]
            for change, mask in [("gained_correct", score > reference), ("lost_correct", score < reference)]:
                for index in np.flatnonzero(mask)[:5]:
                    row = data.iloc[index]
                    examples.append({"variant": variant, "mode": mode, "change": change,
                                     "sample_id": row.sample_id, "eval_name": row.eval_name,
                                     "prompt_text": row.prompt_text, "small_score": row.small_score,
                                     "large_score": row.large_score})
    pd.DataFrame(examples).to_csv(reports / "validation_error_examples.csv", index=False)
    from src.embedding_experiment.reporting import write_report
    write_report(reports, design, selection, metrics, groups, summary)
    print(metrics[(metrics.split == "test") & (metrics["mode"] != "baseline")][
        ["variant", "mode", "answer_score", "cost_saving", "fn", "fp"]].to_string(index=False))
    print(f"Được chọn từ validation: {selection['selected_variant']}. Báo cáo: {reports / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
