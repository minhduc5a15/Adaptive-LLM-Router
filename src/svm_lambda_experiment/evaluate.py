"""Áp dụng toàn bộ ngưỡng khóa trên validation lên test; không chọn lambda lại."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd

from src.embedding_experiment.common import ROOT, load_split, split_path, sha256, write_json, check_disjoint, run_dirs
from src.embedding_experiment.encoder import FrozenEncoder, cached_embeddings
from src.embedding_experiment.inference import positive_probability
from src.embedding_experiment.evaluate import compare_paired
from src.svm_embedding_experiment.features import input_frame
from src.svm_embedding_experiment.common import source_hashes
from src.svm_lambda_experiment.common import DEFAULT_RUN, VARIANTS, MODES, paths
from src.svm_lambda_experiment.curves import threshold_curve, normalized_area, frontier


def load_locked(name):
    reports, _ = paths(name)
    selection = json.loads((reports / 'selection.json').read_text())
    protocol = json.loads((reports / 'protocol.json').read_text())
    for filename, key in [('protocol.json', 'protocol_sha256'), ('locked_budget_policies.csv', 'budget_policies_sha256')]:
        if sha256(reports / filename) != selection[key]:
            raise ValueError(f'{filename} đã thay đổi')
    if sha256(ROOT / selection['artifact']) != selection['artifact_sha256']:
        raise ValueError('Artifact đã thay đổi')
    bundle = joblib.load(ROOT / selection['artifact'])
    if bundle['protocol_sha256'] != selection['protocol_sha256'] or bundle['selected_lambda'] != selection['selected_lambda']:
        raise ValueError('Artifact và selection không khớp')
    return reports, protocol, selection, bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default=DEFAULT_RUN)
    args = parser.parse_args()
    reports, protocol, selection, bundle = load_locked(args.run_name)
    for split, expected in protocol['split_sha256'].items():
        if sha256(split_path(split)) != expected:
            raise ValueError('Train/validation đã thay đổi')
    if sha256(ROOT / 'models/router.joblib') != protocol['baseline_sha256']:
        raise ValueError('Baseline đã thay đổi')
    if (reports / 'summary.json').exists():
        if json.loads((reports / 'summary.json').read_text())['test_sha256'] != sha256(split_path('test')):
            raise ValueError('Test đã đổi so với lần đánh giá trước')
    train, val, test = [load_split(s) for s in ['train', 'validation', 'test']]
    check_disjoint(train, test)
    check_disjoint(val, test)
    encoder = FrozenEncoder()
    if encoder.manifest != bundle['encoder']:
        raise ValueError('Encoder đã đổi')
    _, _, cache = run_dirs(protocol['cache_run'])
    policies = pd.DataFrame(bundle['budget_policies'])
    rows, curves, frontiers, budgets, predictions, meta, route20 = [], [], [], [], [], {}, {}
    for split, data in [('validation', val), ('test', test)]:
        vectors, meta[split] = cached_embeddings(encoder, data, cache, split)
        X = input_frame(data.prompt_text, vectors)
        tree = positive_probability(bundle['tree'], data.prompt_text)
        upper = float(1 - data.small_cost.sum() / data.large_cost.sum())
        for variant in VARIANTS:
            svm = positive_probability(bundle['svms'][variant], X)
            predictions.append(pd.DataFrame({'split': split, 'variant': variant, 'sample_id': data.sample_id,
                                             'tree_probability': tree, 'svm_probability': svm}))
            for mode, alpha in MODES.items():
                scores = alpha * tree + (1 - alpha) * svm
                # Test chỉ dùng tập ngưỡng đã lưu; không tạo/chọn ngưỡng mới từ nhãn test.
                curve = threshold_curve(data, scores, bundle['thresholds'][variant][mode])
                area = normalized_area(curve, upper)
                if split == 'validation' and mode == 'fixed_ensemble':
                    np.testing.assert_allclose(area, selection['configs'][variant]['area'], atol=1e-12, rtol=1e-12)
                rows.append({'split': split, 'variant': variant, 'mode': mode,
                             'normalized_area': area, 'area_upper_saving': upper,
                             'C': selection['configs'][variant]['C'], 'n_thresholds': len(curve)})
                curves.append(curve.assign(split=split, variant=variant, mode=mode))
                frontiers.append(frontier(curve).assign(split=split, variant=variant, mode=mode))
                locked = policies[(policies.variant == variant) & (policies['mode'] == mode)].reset_index(drop=True)
                actual = threshold_curve(data, scores, locked.threshold.to_numpy())
                if split == 'validation':
                    np.testing.assert_allclose(actual.answer_score, locked.answer_score, atol=1e-12)
                    np.testing.assert_allclose(actual.cost_saving, locked.cost_saving, atol=1e-12)
                actual['target_saving_validation'] = locked.target_saving.to_numpy()
                actual['validation_quality'] = locked.answer_score.to_numpy()
                actual['validation_saving'] = locked.cost_saving.to_numpy()
                budgets.append(actual.assign(split=split, variant=variant, mode=mode))
                if split == 'test' and mode == 'fixed_ensemble':
                    matching = locked[np.isclose(locked.target_saving, .2)]
                    if len(matching) != 1:
                        raise ValueError('Cần một policy validation cho budget 20%')
                    route20[variant] = (scores >= matching.iloc[0].threshold).astype(int)
    areas = pd.DataFrame(rows)
    all_curves, all_budgets = pd.concat(curves, ignore_index=True), pd.concat(budgets, ignore_index=True)
    areas.to_csv(reports / 'curve_areas.csv', index=False)
    all_curves.to_csv(reports / 'threshold_curves.csv', index=False)
    pd.concat(frontiers, ignore_index=True).to_csv(reports / 'pareto_frontiers.csv', index=False)
    all_budgets.to_csv(reports / 'budget_evaluation.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(reports / 'predictions.csv', index=False)
    winner = selection['selected_lambda']
    comparisons = {reference: compare_paired(test, route20[winner], route20[reference]) for reference in ['tfidf', 'lambda_1']}
    summary = {'selected_lambda': winner, 'selected_overall': selection['selected_overall'],
               'test_sha256': sha256(split_path('test')), 'selection_sha256': sha256(reports / 'selection.json'),
               'test_status': protocol['test_status'], 'paired_at_locked_20pct': comparisons,
               'evaluation_source_sha256': source_hashes()}
    write_json(reports / 'summary.json', summary)
    write_json(reports / 'encoding_evaluation.json', meta)
    from src.svm_lambda_experiment.reporting import write_report
    write_report(reports, protocol, selection, summary, areas, all_curves, all_budgets)
    print(areas[areas['mode'] == 'fixed_ensemble'].to_string(index=False))
    print('\nTest tại policy chọn trên validation cho tiết kiệm 20%:')
    mask = (all_budgets.split == 'test') & (all_budgets['mode'] == 'fixed_ensemble') & np.isclose(all_budgets.target_saving_validation, .2)
    print(all_budgets.loc[mask, ['variant', 'threshold', 'correct_count', 'answer_score', 'cost_saving']].to_string(index=False))
    print(f'Đã đánh giá. Lambda khóa từ validation: {winner}. Báo cáo: {reports / "RESULTS.md"}')


if __name__ == '__main__':
    main()
