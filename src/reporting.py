"""Đánh giá cấu hình đã khóa và xuất báo cáo có thể đọc, kiểm tra lại."""

import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "adaptive_llm_router_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree

from src.evaluation import bootstrap_vs_large, routing_metrics, selected_outcomes, sweep_thresholds
from src.router import combine_probabilities, predict_probabilities


def markdown_metrics(metrics):
    header = "| Phương án | Đúng nhãn Router | F1 nhãn 1 | Câu được chấm đúng | Tổng chi phí (USD benchmark) | Tiết kiệm | Gọi Mixtral |"
    lines = [header, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in metrics.to_dict("records"):
        lines.append(f'| {row["name"]} | {row["accuracy"]:.2%} | {row["f1"]:.4f} | '
                     f'{row["answer_score"]:.2%} | {row["total_cost"]:.6f} | '
                     f'{row["cost_saving"]:.2%} | {row["large_rate"]:.2%} |')
    return "\n".join(lines)


def evaluate_split(bundle, data, probabilities):
    rng = np.random.default_rng(bundle["seed"])
    n = len(data)
    matched_rate = bundle["selected_policy"]["large_rate"]  # chỉ từ validation
    routes = {
        "Always Small": np.zeros(n, dtype=int),
        "Always Large": np.ones(n, dtype=int),
        "Random 50%": (rng.random(n) < 0.5).astype(int),
        "Random matched": (rng.random(n) < matched_rate).astype(int),
    }
    probability_by_name = {}
    for policy in bundle["policies"]:
        name = policy["name"]
        p = combine_probabilities(probabilities, policy["tree_weight"])
        routes[name] = (p >= policy["threshold"]).astype(int)
        probability_by_name[name] = p
    routes["Oracle (labels)"] = data["label"].to_numpy()
    rows = [{"name": name, **routing_metrics(data, route, probability_by_name.get(name))}
            for name, route in routes.items()]
    return pd.DataFrame(rows), routes


def make_figures(bundle, validation, test, val_probabilities, test_probabilities, report_dir):
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    selected_name = bundle["selected_policy"]["name"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    all_curves = []
    for ax, split, data, probabilities in [
        (axes[0], "validation", validation, val_probabilities),
        (axes[1], "test", test, test_probabilities),
    ]:
        for policy in bundle["policies"]:
            p = combine_probabilities(probabilities, policy["tree_weight"])
            curve = pd.DataFrame(sweep_thresholds(data, p, policy["tree_weight"], policy["name"]))
            curve["split"] = split
            all_curves.append(curve)
            ax.plot(curve["mean_cost"] * 1000, curve["answer_score"] * 100, label=policy["name"])
            if policy["name"] == selected_name:
                scores, costs = selected_outcomes(data, p >= policy["threshold"])
                ax.scatter(costs.mean() * 1000, scores.mean() * 100, marker="*", s=180,
                           color="black", zorder=10, label="Locked policy")
        ax.plot([data["small_cost"].mean() * 1000, data["large_cost"].mean() * 1000],
                [data["small_score"].mean() * 100, data["large_score"].mean() * 100],
                "k--", alpha=0.5, label="Expected random mix")
        ax.set(title=split.capitalize(), xlabel="Benchmark USD / 1,000 prompts", ylabel="Scored correct (%)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "quality_cost.png", dpi=160)
    plt.close(fig)
    pd.concat(all_curves, ignore_index=True).to_csv(report_dir / "quality_cost_curves.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, policy in zip(axes, bundle["policies"]):
        p = combine_probabilities(test_probabilities, policy["tree_weight"])
        ConfusionMatrixDisplay.from_predictions(test["label"], p >= policy["threshold"],
            display_labels=["Mistral", "Mixtral"], colorbar=False, ax=ax, cmap="Blues")
        ax.set_title(f'{policy["name"]}, threshold={policy["threshold"]:.2f}')
    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    for policy in bundle["policies"]:
        p = combine_probabilities(test_probabilities, policy["tree_weight"])
        fraction, mean = calibration_curve(test["label"], p, n_bins=10, strategy="quantile")
        ax.plot(mean, fraction, "o-", label=policy["name"])
    ax.set(xlabel="Mean predicted P(label=1)", ylabel="Observed fraction label=1", title="Test calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "calibration.png", dpi=160)
    plt.close(fig)

    tree = bundle["tree"].calibrated_classifiers_[0].estimator.named_steps["classifier"]
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(tree, max_depth=2, feature_names=["char_count", "word_count", "line_count"],
              class_names=["Mistral", "Mixtral"], filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Decision Tree: first levels (before probability calibration)")
    fig.tight_layout()
    fig.savefig(figure_dir / "decision_tree.png", dpi=160)
    plt.close(fig)
    return tree


def evaluate_and_report(bundle, train, validation, test, report_dir, test_hash):
    val_probabilities = predict_probabilities(bundle, validation["prompt_text"])
    started = perf_counter()
    test_probabilities = predict_probabilities(bundle, test["prompt_text"])
    batch_seconds = perf_counter() - started
    val_metrics, _ = evaluate_split(bundle, validation, val_probabilities)
    test_metrics, test_routes = evaluate_split(bundle, test, test_probabilities)
    val_metrics.to_csv(report_dir / "validation_metrics.csv", index=False)
    test_metrics.to_csv(report_dir / "test_metrics.csv", index=False)
    # Đo thêm ở ngưỡng 0.5 để phân biệt chất lượng phân loại và chính sách tiết kiệm.
    classification_rows = []
    train_probabilities = predict_probabilities(bundle, train["prompt_text"])
    for split, data, probabilities in [("train", train, train_probabilities),
                                       ("validation", validation, val_probabilities),
                                       ("test", test, test_probabilities)]:
        for policy in bundle["policies"]:
            p = combine_probabilities(probabilities, policy["tree_weight"])
            classification_rows.append({"split": split, "name": policy["name"],
                                       **routing_metrics(data, p >= 0.5, p)})
    pd.DataFrame(classification_rows).to_csv(report_dir / "classification_at_050.csv", index=False)

    selected_policy = bundle["selected_policy"]
    selected_name = selected_policy["name"]
    route = test_routes[selected_name]
    scores, costs = selected_outcomes(test, route)
    predictions = test.drop(columns="prompt_text").copy()
    predictions["tree_probability"] = test_probabilities["tree"]
    predictions["svm_probability"] = test_probabilities["svm"]
    predictions["selected_probability"] = combine_probabilities(test_probabilities, selected_policy["tree_weight"])
    predictions["route"] = route
    predictions["selected_score"] = scores
    predictions["selected_cost"] = costs
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    by_benchmark = predictions.groupby("eval_name").agg(
        n=("sample_id", "size"), answer_score=("selected_score", "mean"),
        always_large_score=("large_score", "mean"), large_rate=("route", "mean"),
        total_cost=("selected_cost", "sum"), always_large_cost=("large_cost", "sum"),
    )
    by_benchmark["answer_delta"] = by_benchmark["answer_score"] - by_benchmark["always_large_score"]
    by_benchmark["cost_saving"] = 1 - by_benchmark["total_cost"] / by_benchmark["always_large_cost"]
    by_benchmark.sort_values("n", ascending=False).to_csv(report_dir / "test_by_benchmark.csv")
    errors = predictions.loc[predictions["route"] != predictions["label"]].copy()
    errors["prompt_excerpt"] = test.loc[errors.index, "prompt_text"].str.slice(0, 600)
    errors["error_type"] = np.where(errors["label"] == 1, "missed_large_gain", "unnecessary_large")
    examples = errors.groupby("error_type", sort=True).head(10)
    examples.to_csv(report_dir / "error_examples.csv", index=False)
    confidence = bootstrap_vs_large(test, route)
    selected_metrics = test_metrics.set_index("name").loc[selected_name].to_dict()
    for field in ["n_samples", "tn", "fp", "fn", "tp"]:
        selected_metrics[field] = int(selected_metrics[field])
    summary = {
        "selected_policy": selected_policy, "test_metrics": selected_metrics,
        "bootstrap": confidence, "test_sha256": test_hash,
        "test_prediction_batch_seconds": batch_seconds,
        "batch_average_ms_per_prompt": batch_seconds * 1000 / len(test),
        "passes_quality_constraint_on_test": bool(
            selected_metrics["answer_delta_vs_large"] >= -bundle["max_quality_drop"]),
        "test_used_for_selection": False,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tree = make_figures(bundle, validation, test, val_probabilities, test_probabilities, report_dir)
    write_report(bundle, train, validation, test, val_metrics, test_metrics, summary, tree, report_dir)
    print("\nKẾT QUẢ TEST (cấu hình chọn từ validation):")
    print(test_metrics[["name", "accuracy", "answer_score", "total_cost", "cost_saving", "large_rate"]].to_string(index=False))
    print(f'\nPhương án đã chọn: {selected_name}. Báo cáo: {report_dir / "RESULTS.md"}', flush=True)


def write_report(bundle, train, validation, test, val_metrics, test_metrics, summary, tree, report_dir):
    policy = bundle["selected_policy"]
    metrics = summary["test_metrics"]
    quality_ci = summary["bootstrap"]["answer_delta_95ci"]
    savings_ci = summary["bootstrap"]["cost_saving_95ci"]
    test_by_name = test_metrics.set_index("name")
    svm = test_by_name.loc["SVM"]
    ensemble = test_by_name.loc["Ensemble"]
    classifier_metrics = pd.read_csv(report_dir / "classification_at_050.csv")
    default_selected = classifier_metrics.loc[
        (classifier_metrics["split"] == "test") & (classifier_metrics["name"] == policy["name"])
    ].iloc[0]
    passed = summary["passes_quality_constraint_on_test"] and metrics["cost_saving"] > 0
    verdict = ("Khả quan trong phạm vi thí nghiệm: tiết kiệm chi phí và đạt mức giảm điểm cho phép theo ước lượng điểm trên test."
               if passed else "Chưa đạt đồng thời mục tiêu điểm trả lời và tiết kiệm chi phí trên test.")
    lines = [
        "# Kết quả Adaptive LLM Router", "", "## Kết luận", "", verdict, "",
        f'- Phương án chọn từ validation: **{policy["name"]}**, trọng số cây **{policy["tree_weight"]:.2f}**, ngưỡng **{policy["threshold"]:.2f}**.',
        f'- Test: **{metrics["answer_score"]:.2%}** câu được chấm đúng; tiết kiệm **{metrics["cost_saving"]:.2%}** so với luôn chọn Mixtral.',
        f'- Thay đổi tỷ lệ đúng so với Mixtral: **{metrics["answer_delta_vs_large"] * 100:+.2f} điểm phần trăm**.',
        f'- Khoảng bootstrap 95% cho thay đổi điểm: [{quality_ci[0]*100:+.2f}, {quality_ci[1]*100:+.2f}] điểm phần trăm.',
        f'- Khoảng bootstrap 95% cho tiết kiệm chi phí: [{savings_ci[0]:.2%}, {savings_ci[1]:.2%}].',
        "", "Các khoảng trên lấy mẫu lại các nhóm prompt test (1.000 lần), giữ cố định mô hình. Chúng không bao gồm bất định do huấn luyện lại hay thay đổi phân bố tác vụ.",
        "",
        f'Khoảng bất định của thay đổi điểm {"vượt" if quality_ci[0] < -bundle["max_quality_drop"] else "không vượt"} mức giảm {bundle["max_quality_drop"]*100:.1f} điểm phần trăm ở phía bất lợi; việc đạt tiêu chí theo ước lượng điểm không phải cam kết thống kê cho dữ liệu mới.',
        "", "## Dữ liệu và quy trình", "",
        f'- Train / validation / test: **{len(train):,} / {len(validation):,} / {len(test):,}** mẫu.',
        '- RouterBench 0-shot, Mistral 7B Chat → Mixtral 8×7B Chat; lọc điểm nhị phân và prompt chứa chữ Hán theo DATA_PREPARATION.md.',
        '- Nhãn 1: Small sai, Large đúng. Nhãn 0 gồm cả hai mô hình cùng sai; dự đoán đúng nhãn không đồng nghĩa câu trả lời đúng.',
        '- Các tập không có prompt chung. Hiệu chỉnh sigmoid bằng dự đoán ngoài fold trên train, 3 fold StratifiedGroupKFold theo prompt.',
        '- Pipeline SVM fit TF-IDF riêng từng fold; mô hình cơ sở cuối cùng fit toàn train. Validation không dùng để fit tham số nội bộ của cây/SVM hoặc hàm hiệu chỉnh; dùng để chọn cấu hình, trọng số kết hợp và ngưỡng.',
        '- Decision Tree dùng đúng ba cột: char_count, word_count (tách theo khoảng trắng), line_count.',
        f'- Cấu hình cây chọn bằng log loss validation: `{json.dumps(bundle["best_configs"]["tree"])}`. Cây có độ sâu {tree.get_depth()}, {tree.get_n_leaves()} lá.',
        f'- SVM tuyến tính: `{json.dumps(bundle["best_configs"]["svm"])}`; TF-IDF tối đa 5.000 mục.',
        '- Thử 8 cấu hình cây và 3 giá trị C cho SVM. Mỗi họ chọn log loss validation nhỏ nhất.',
        '- Ensemble thử 9 trọng số cây từ 0,1 đến 0,9. Mỗi phương án thử 101 ngưỡng từ 0 đến 1.',
        f'- Quy tắc chọn chính sách đặt trước khi xem test: chi phí validation thấp nhất, tỷ lệ đúng giảm tối đa **{bundle["max_quality_drop"]*100:.1f} điểm phần trăm** so với Always Large.',
        '- Lưu selection.json và router.joblib trước khi nạp test. Các bảng test không được dùng để đổi mô hình, trọng số hoặc ngưỡng.',
        f'- Huấn luyện và lựa chọn: {bundle["training_and_selection_seconds"]:.1f} giây. Dự đoán cả hai mô hình trên batch test: {summary["test_prediction_batch_seconds"]:.3f} giây (không phải độ trễ một yêu cầu tương tác).',
        "", "## Validation", "", markdown_metrics(val_metrics),
        "", "## Test", "", markdown_metrics(test_metrics),
        "", "Random matched dùng xác suất gọi Large bằng tỷ lệ của chính sách được chọn trên validation; các lần rút ngẫu nhiên có seed 42. Oracle dùng nhãn thật nên chỉ là mốc tham chiếu không triển khai được.",
        "", "Mọi giá chi phí là ước tính được lưu trong RouterBench, không phải hóa đơn đã trả hoặc giá API hiện tại. Precision, recall và F1 dùng nhãn 1 làm lớp dương.",
        "", "## Diễn giải kết quả", "",
        f'- Ensemble so với SVM: tiết kiệm thêm {(ensemble["cost_saving"]-svm["cost_saving"])*100:+.2f} điểm phần trăm chi phí tham chiếu, thay đổi tỷ lệ đúng {(ensemble["answer_score"]-svm["answer_score"])*100:+.2f} điểm phần trăm. Chênh lệch nhỏ này chưa chứng minh kết hợp tốt hơn SVM nói chung.',
        f'- AUC của phương án chọn: {metrics["roc_auc"]:.3f}. Khả năng phân biệt nhãn ở mức vừa phải.',
        f'- Tại ngưỡng 0,5, accuracy của phương án chọn là {default_selected["accuracy"]:.2%}, so với baseline luôn Small {test_by_name.loc["Always Small", "accuracy"]:.2%}.',
        f'- Ngưỡng {policy["threshold"]:.2f} được chọn để ưu tiên giữ điểm câu trả lời, làm recall nhãn 1 đạt {metrics["recall"]:.2%} nhưng accuracy Router còn {metrics["accuracy"]:.2%}. Đây là các mục tiêu khác nhau; không nên dùng accuracy riêng để kết luận về chi phí và điểm trả lời.',
        "", "## Phân tích sai số của phương án được chọn", "",
        f'- TN={int(metrics["tn"])}, FP={int(metrics["fp"])}, FN={int(metrics["fn"])}, TP={int(metrics["tp"])}.',
        '- FN: bỏ lỡ trường hợp Mixtral có điểm 1 còn Mistral có điểm 0.',
        '- FP: gọi Mixtral khi nhãn là 0; có thể tăng chi phí mà không tăng điểm, hoặc làm giảm điểm khi Small đúng và Large sai.',
        '- [Kết quả theo benchmark](test_by_benchmark.csv); các nhóm ít mẫu không đủ để kết luận chắc chắn.',
        '- [Ví dụ sai](error_examples.csv): tối đa 10 trường hợp mỗi loại, chọn theo thứ tự dataset, không phải mẫu đại diện ngẫu nhiên.',
        '- [Chi tiết từng mẫu test](test_predictions.csv).',
        '- [Độ đúng phân loại tại ngưỡng 0,5](classification_at_050.csv) cho train/validation/test giúp quan sát chênh lệch khả năng khái quát.',
        "", "## Biểu đồ", "",
        "![Điểm trả lời và chi phí](figures/quality_cost.png)", "",
        "Ngôi sao là cấu hình đã khóa từ validation. Đường test ở các ngưỡng khác chỉ để mô tả đánh đổi, không dùng để lựa chọn lại.", "",
        "![Ma trận nhầm lẫn](figures/confusion_matrices.png)", "",
        "![Hiệu chỉnh xác suất](figures/calibration.png)", "",
        "![Các tầng đầu của cây](figures/decision_tree.png)", "",
        "## Giới hạn", "",
        '- Đây là kiểm thử trên các prompt chưa thấy trong cùng hỗn hợp benchmark. Chưa thực hiện thí nghiệm giữ riêng toàn bộ nhóm benchmark chưa thấy khi train.',
        '- Ba đặc trưng cấu trúc chỉ là phiên bản ban đầu. TF-IDF chủ yếu ghi nhận từ xuất hiện, chưa mô hình hóa đầy đủ ngữ nghĩa.',
        '- Lọc chữ Hán không đảm bảo dữ liệu chỉ có tiếng Anh; word_count vẫn là phép đếm theo khoảng trắng.',
        '- Ngưỡng đạt ràng buộc trên validation không đảm bảo giữ được ràng buộc trên mọi tập dữ liệu mới. Đọc cả khoảng bất định, không chỉ ước lượng điểm.',
        '- Các xác suất đã qua hiệu chỉnh vẫn là ước lượng; biểu đồ calibration thể hiện sai lệch còn lại.',
        '- Đổi cặp LLM, ngôn ngữ hoặc tác vụ cần đánh giá lại. Demo prompt mới chỉ định tuyến, không gọi LLM tạo câu trả lời.',
        "", "## Tái lập và đọc mã", "",
        '```bash', '.venv/bin/python src/train_models.py', '.venv/bin/python -m src.predict --prompt "Explain binary search."', '.venv/bin/streamlit run app.py', '```',
        "", '- Mô hình: `models/router.joblib`; seed, phiên bản thư viện và checksum dữ liệu: [selection.json](selection.json), [summary.json](summary.json).',
        '- [Các cấu hình đã thử](validation_candidates.csv); [toàn bộ lựa chọn ngưỡng validation](validation_thresholds.csv).',
        '- Thứ tự đọc mã: features.py → train_models.py → router.py → evaluation.py → reporting.py → app.py.',
        "", "## Tài liệu phương pháp", "",
        '- [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html).',
        '- [scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).',
        "",
    ]
    (report_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
