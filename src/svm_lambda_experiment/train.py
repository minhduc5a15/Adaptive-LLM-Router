"""Chọn C và lambda trên diện tích đường validation; chưa mở test."""
import argparse
import json
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo
from importlib.metadata import version
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.embedding_experiment.common import ROOT, SEED, load_split, split_path, sha256, write_json, check_disjoint, run_dirs
from src.embedding_experiment.encoder import FrozenEncoder, cached_embeddings
from src.embedding_experiment.inference import positive_probability
from src.svm_embedding_experiment.features import SVMFeatures, input_frame
from src.svm_embedding_experiment.common import source_hashes
from src.svm_embedding_experiment.evaluate import load_locked as load_previous
from src.svm_lambda_experiment.common import DEFAULT_RUN, CACHE_RUN, PREVIOUS_RUN, LAMBDAS, VARIANTS, MODES, paths
from src.svm_lambda_experiment.features import WeightedSVMFeatures
from src.svm_lambda_experiment.curves import threshold_curve, normalized_area, best_for_budgets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default=DEFAULT_RUN)
    args = parser.parse_args()
    reports, models_dir = paths(args.run_name)
    if reports.exists() or models_dir.exists():
        parser.error('Run đã tồn tại; dùng tên mới')
    baseline = joblib.load(ROOT / 'models/router.joblib')
    old_reports, _, old_selection, previous = load_previous(PREVIOUS_RUN)
    for split in ['train', 'validation']:
        if sha256(split_path(split)) != baseline[f'{split}_sha256']:
            raise ValueError('Dữ liệu khác baseline')
    if version('scikit-learn') != baseline['versions']['sklearn']:
        raise ValueError('Sklearn khác baseline')
    train, val = load_split('train'), load_split('validation')
    check_disjoint(train, val)
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(
        train.prompt_text, train.label, groups=train.prompt_text))
    fold_ids = np.full(len(train), -1)
    for number, (fit, held) in enumerate(folds):
        check_disjoint(train.iloc[fit], train.iloc[held])
        if train.iloc[fit].label.nunique() != 2 or train.iloc[held].label.nunique() != 2:
            raise ValueError('Fold thiếu lớp')
        fold_ids[held] = number
    encoder = FrozenEncoder()
    if encoder.manifest != previous['encoder']:
        raise ValueError('Encoder khác thí nghiệm trước')
    upper = float(1 - val.small_cost.sum() / val.large_cost.sum())
    if upper <= 0:
        raise ValueError('Mistral cần rẻ hơn Mixtral trên validation')
    protocol = {
        'run_name': args.run_name, 'started_at': datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),
        'lambda_grid': LAMBDAS, 'C_grid': [0.1, 1., 10.], 'variants': VARIANTS, 'seed': SEED,
        'formula': 'L2 normalize([TF-IDF, lambda * MiniLM]); no per-block normalization after weighting',
        'tree': 'frozen v1 structural tree', 'modes': MODES, 'primary_mode': 'fixed_ensemble',
        'selection': 'maximum normalized area of validation budget envelope for C and lambda; ties: lower log loss then smaller C, declared lambda order',
        'area_definition': 'integral Q*(b) db / upper, Q*(b)=max quality among threshold policies with saving>=b',
        'validation_area_domain': [0., upper], 'domain_rule': '0 to always-Small saving, shared across variants within split',
        'thresholds': 'all distinct validation scores plus 0 and 1.01; scores>=threshold route Large',
        'budget_grid': '101 fractions of validation upper saving plus .10/.20/.30/.40/.50/.60 when feasible',
        'C_selection': 'same full-curve area objective for TF-IDF, embedding and all lambdas; 18 configurations',
        'calibration': 'sigmoid, grouped3fold, ensemble=False; TF-IDF fitted inside pipeline each fold',
        'svm': {'max_iter': 10000, 'random_state': SEED, 'tfidf_max_features': 5000},
        'encoder': encoder.manifest, 'cache_run': CACHE_RUN,
        'split_sha256': {s: sha256(split_path(s)) for s in ['train', 'validation']},
        'baseline_sha256': sha256(ROOT / 'models/router.joblib'),
        'previous_selection_sha256': sha256(old_reports / 'selection.json'),
        'previous_artifact_sha256': old_selection['artifact_sha256'],
        'test_protocol': 'Only evaluate locked validation thresholds; test envelope/area descriptive, never choose deployment thresholds or lambda from test',
        'test_status': 'Previously inspected test; exploratory follow-up, not independent holdout',
        'bootstrap': '1000 paired prompt-group samples at locked validation 20% saving policy; selected lambda vs TF-IDF and lambda1; no multiplicity correction',
        'source_sha256': source_hashes(),
        'versions': {n: version(n) for n in ['scikit-learn', 'numpy', 'pandas', 'scipy', 'onnxruntime', 'tokenizers']},
    }
    reports.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    write_json(reports / 'protocol.json', protocol)
    train[['sample_id']].assign(fold=fold_ids).to_csv(reports / 'train_folds.csv', index=False)
    print('Protocol đã lưu. Chỉ đọc cache train/validation.', flush=True)
    started = perf_counter()
    _, _, cache = run_dirs(CACHE_RUN)
    train_vectors, train_meta = cached_embeddings(encoder, train, cache, 'train')
    val_vectors, val_meta = cached_embeddings(encoder, val, cache, 'validation')
    write_json(reports / 'encoding.json', {'train': train_meta, 'validation': val_meta, 'cache_reused': True})
    X_train, X_val = input_frame(train.prompt_text, train_vectors), input_frame(val.prompt_text, val_vectors)
    tree = positive_probability(baseline['tree'], val.prompt_text)
    candidates, models, configs, scores_by_variant, curves_all = [], {}, {}, {}, []
    warnings.filterwarnings('error', category=ConvergenceWarning)
    for variant in VARIANTS:
        best_key = None
        for c in protocol['C_grid']:
            features = (WeightedSVMFeatures(float(variant.split('_')[1])) if variant.startswith('lambda_')
                        else SVMFeatures(variant))
            estimator = Pipeline([('features', features), ('classifier', LinearSVC(C=c, max_iter=10000, random_state=SEED))])
            model = CalibratedClassifierCV(estimator, method='sigmoid', cv=folds, ensemble=False, n_jobs=1)
            fit_started = perf_counter()
            model.fit(X_train, train.label)
            svm = positive_probability(model, X_val)
            loss = float(log_loss(val.label, svm, labels=[0, 1]))
            areas = {}
            for mode, alpha in MODES.items():
                curve = threshold_curve(val, alpha * tree + (1 - alpha) * svm)
                areas[mode] = normalized_area(curve, upper)
                curves_all.append(curve.assign(variant=variant, C=c, mode=mode))
            candidates.append({'variant': variant, 'C': c, 'log_loss': loss,
                               'area_fixed_ensemble': areas['fixed_ensemble'], 'area_svm_only': areas['svm_only'],
                               'fit_seconds': perf_counter() - fit_started})
            # λ=1 và hai mốc đơn phải tái tạo mô hình trước ở C đã chọn cũ.
            old_key = 'combined' if variant == 'lambda_1' else variant
            if old_key in previous['svms'] and c == old_selection['best_configs'][old_key]['C']:
                np.testing.assert_allclose(svm, positive_probability(previous['svms'][old_key], X_val), atol=1e-12, rtol=1e-12)
            key = (-areas['fixed_ensemble'], loss, c)
            if best_key is None or key < best_key:
                best_key, models[variant], scores_by_variant[variant] = key, model, svm
                configs[variant] = {'C': c, 'area': areas['fixed_ensemble'], 'log_loss': loss}
            print(f"{variant}: C={c:g}, area={areas['fixed_ensemble']:.6f}, log_loss={loss:.6f}", flush=True)
        # Ghi tiến độ để giữ kết quả chẩn đoán nếu một cấu hình sau lỗi.
        pd.DataFrame(candidates).to_csv(reports / 'validation_candidates.csv', index=False)
    lambda_names = [v for v in VARIANTS if v.startswith('lambda_')]
    selected_lambda = min(lambda_names, key=lambda v: (-configs[v]['area'], configs[v]['log_loss'], configs[v]['C']))
    selected_overall = min(VARIANTS, key=lambda v: (-configs[v]['area'], configs[v]['log_loss'], configs[v]['C']))
    budgets = np.unique(np.r_[np.linspace(0., upper, 101), np.array([.1, .2, .3, .4, .5, .6])[np.array([.1, .2, .3, .4, .5, .6]) <= upper]])
    locked, thresholds, predictions = [], {}, []
    for variant, svm in scores_by_variant.items():
        thresholds[variant] = {}
        predictions.append(pd.DataFrame({'sample_id': val.sample_id, 'variant': variant, 'tree_probability': tree, 'svm_probability': svm}))
        for mode, alpha in MODES.items():
            curve = threshold_curve(val, alpha * tree + (1 - alpha) * svm)
            thresholds[variant][mode] = curve.threshold.tolist()
            locked.append(best_for_budgets(curve, budgets).assign(variant=variant, mode=mode))
    policy_frame = pd.concat(locked, ignore_index=True)
    policy_frame.to_csv(reports / 'locked_budget_policies.csv', index=False)
    pd.concat(curves_all, ignore_index=True).to_csv(reports / 'validation_candidate_curves.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(reports / 'validation_predictions.csv', index=False)
    bundle = {'svms': models, 'tree': baseline['tree'], 'encoder': encoder.manifest,
              'configs': configs, 'selected_lambda': selected_lambda, 'selected_overall': selected_overall,
              'thresholds': thresholds, 'budget_policies': policy_frame.to_dict('records'),
              'protocol_sha256': sha256(reports / 'protocol.json')}
    artifact = models_dir / 'router.joblib'
    joblib.dump(bundle, artifact)
    write_json(reports / 'selection.json', {'configs': configs, 'selected_lambda': selected_lambda,
               'selected_overall': selected_overall, 'protocol_sha256': bundle['protocol_sha256'],
               'artifact': str(artifact.relative_to(ROOT)), 'artifact_sha256': sha256(artifact),
               'budget_policies_sha256': sha256(reports / 'locked_budget_policies.csv'),
               'training_seconds': perf_counter() - started})
    print(f'Đã khóa lambda: {selected_lambda}; overall: {selected_overall}. Chưa đọc test.', flush=True)


if __name__ == '__main__':
    main()
