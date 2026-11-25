"""Đánh giá các chính sách đã khóa; không chọn lại tham số bằng test."""
import argparse
import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from src.embedding_experiment.common import ROOT, load_split, split_path, sha256, write_json, check_disjoint, run_dirs
from src.embedding_experiment.encoder import FrozenEncoder, cached_embeddings
from src.embedding_experiment.inference import positive_probability, route_scores
from src.embedding_experiment.evaluate import error_counts, compare_paired
from src.evaluation import routing_metrics, selected_outcomes
from src.features import extract_structure_features
from src.svm_embedding_experiment.common import DEFAULT_RUN, VARIANTS, MODES, paths, source_hashes
from src.svm_embedding_experiment.features import input_frame


def load_locked(name):
    reports, _ = paths(name)
    selection = json.loads((reports / "selection.json").read_text())
    protocol = json.loads((reports / "protocol.json").read_text())
    if sha256(reports / "protocol.json") != selection["protocol_sha256"]:
        raise ValueError("Protocol đã đổi")
    artifact = ROOT / selection["artifact"]
    if sha256(artifact) != selection["artifact_sha256"]:
        raise ValueError("Artifact đã đổi")
    bundle = joblib.load(artifact)
    if bundle["policies"] != selection["policies"] or bundle["protocol_sha256"] != selection["protocol_sha256"]:
        raise ValueError("Artifact không khớp selection")
    return reports, protocol, selection, bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    args = parser.parse_args()
    reports, protocol, selection, bundle = load_locked(args.run_name)
    for split, expected in protocol["split_sha256"].items():
        if sha256(split_path(split)) != expected:
            raise ValueError("Dữ liệu khác lúc training")
    if sha256(ROOT / "models/router.joblib") != protocol["baseline_sha256"]:
        raise ValueError("Baseline đã đổi")
    if (reports / "summary.json").exists():
        old_summary = json.loads((reports / "summary.json").read_text())
        if old_summary["test_sha256"] != sha256(split_path("test")):
            raise ValueError("Test khác lần đánh giá trước")
    baseline = joblib.load(ROOT / "models/router.joblib")
    train, validation, test = [load_split(s) for s in ["train", "validation", "test"]]
    check_disjoint(train, test)
    check_disjoint(validation, test)
    encoder = FrozenEncoder()
    if encoder.manifest != bundle["encoder"]:
        raise ValueError("Encoder đã đổi")
    _, _, cache = run_dirs(protocol["cache_run"])
    metrics, predictions, group_rows, timings, encoding, all_routes = [], [], [], [], {}, {}
    for split, data in [("validation", validation), ("test", test)]:
        vectors, encoding[split] = cached_embeddings(encoder, data, cache, split)
        inputs = input_frame(data.prompt_text, vectors)
        tree = positive_probability(bundle["tree"], data.prompt_text)
        structure = extract_structure_features(data.prompt_text)
        masks = {"all": np.ones(len(data), dtype=bool), "hellaswag": data.eval_name.eq("hellaswag").to_numpy(),
                 "word_count_ge_200": structure.word_count.ge(200).to_numpy()}
        masks.update({f"benchmark:{b}": data.eval_name.eq(b).to_numpy() for b in data.eval_name.unique()})
        all_routes[split] = {}
        for name, routes in [("Always Small", np.zeros(len(data), dtype=int)),
                             ("Always Large", np.ones(len(data), dtype=int)), ("Oracle", data.label.to_numpy())]:
            metrics.append({"split": split, "variant": name, "mode": "baseline",
                            "correct_count": int(selected_outcomes(data, routes)[0].sum()), **routing_metrics(data, routes)})
        for variant in VARIANTS:
            started = perf_counter()
            svm = positive_probability(bundle["svms"][variant], inputs)
            timings.append({"split": split, "variant": variant, "seconds": perf_counter() - started,
                            "n_samples": len(data), "includes_encoding": False})
            if variant == "tfidf":
                np.testing.assert_allclose(svm, positive_probability(baseline["svm"], data.prompt_text), atol=1e-12, rtol=1e-12)
            for mode in MODES:
                policy = bundle["policies"][variant][mode]
                probability, routes = route_scores(tree, svm, policy)
                result = routing_metrics(data, routes, probability)
                if split == "validation":
                    for key in ["answer_score", "mean_cost", "large_rate"]:
                        np.testing.assert_allclose(result[key], policy[key], atol=1e-12, rtol=1e-12)
                scores, costs = selected_outcomes(data, routes)
                all_routes[split][(variant, mode)] = routes
                metrics.append({"split": split, "variant": variant, "mode": mode,
                                "tree_weight": policy["tree_weight"], "threshold": policy["threshold"],
                                "correct_count": int(scores.sum()), **result})
                predictions.append(data[["sample_id", "eval_name", "label", "small_score", "large_score"]].assign(
                    split=split, variant=variant, mode=mode, tree_probability=tree, svm_probability=svm,
                    probability=probability, route=routes, selected_score=scores, selected_cost=costs))
                for group, mask in masks.items():
                    group_rows.append({"split": split, "variant": variant, "mode": mode, "group": group,
                                       **error_counts(data.loc[mask], routes[mask])})
    comparisons = {v: {m: compare_paired(test, all_routes["test"][(v, m)], all_routes["test"][("tfidf", m)])
                       for m in MODES} for v in ["embedding", "combined"]}
    metrics = pd.DataFrame(metrics)
    groups = pd.DataFrame(group_rows)
    metrics.to_csv(reports / "metrics.csv", index=False)
    groups.to_csv(reports / "group_errors.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(reports / "predictions.csv", index=False)
    pd.DataFrame(timings).to_csv(reports / "inference_timings.csv", index=False)
    write_json(reports / "encoding_evaluation.json", encoding)
    summary = {"selected_variant": selection["selected_variant"], "test_sha256": sha256(split_path("test")),
               "selection_sha256": sha256(reports / "selection.json"), "paired_comparisons": comparisons,
               "test_status": protocol["test_status"], "evaluation_source_sha256": source_hashes()}
    write_json(reports / "summary.json", summary)
    from src.svm_embedding_experiment.reporting import write_report
    write_report(reports, selection, metrics, groups, summary)
    print(metrics[(metrics.split == "test") & (metrics['mode'] != "baseline")][
        ["variant", "mode", "correct_count", "answer_score", "cost_saving", "fn", "fp"]].to_string(index=False))
    print(f"Validation chọn: {selection['selected_variant']}. Báo cáo: {reports / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
