"""Định tuyến bằng lambda và ngưỡng budget đã chọn trên validation."""
import argparse
import json
import numpy as np
import pandas as pd
from src.embedding_experiment.encoder import FrozenEncoder
from src.embedding_experiment.inference import positive_probability
from src.svm_embedding_experiment.features import input_frame
from src.svm_lambda_experiment.common import DEFAULT_RUN, VARIANTS, MODES
from src.svm_lambda_experiment.evaluate import load_locked
from src.router import MODEL_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default=DEFAULT_RUN)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--budget', type=float, default=.2, help='Mức tiết kiệm tối thiểu yêu cầu trên validation, ví dụ 0.2')
    parser.add_argument('--prompt', required=True)
    args = parser.parse_args()
    _, _, selection, bundle = load_locked(args.run_name)
    variant = args.variant or selection['selected_lambda']
    policies = pd.DataFrame(bundle['budget_policies'])
    policy = policies[(policies.variant == variant) & (policies['mode'] == 'fixed_ensemble') &
                      np.isclose(policies.target_saving, args.budget, atol=1e-12, rtol=0)]
    if len(policy) != 1:
        parser.error('Budget chưa được khóa; dùng 0.1/0.2/0.3/0.4/0.5/0.6 hoặc xem locked_budget_policies.csv')
    vectors = None
    if variant != 'tfidf':
        encoder = FrozenEncoder()
        if encoder.manifest != bundle['encoder']:
            raise ValueError('Encoder đã đổi')
        vectors, _ = encoder.encode([args.prompt], progress=False)
    inputs = input_frame([args.prompt], vectors)
    svm = positive_probability(bundle['svms'][variant], inputs)[0]
    tree = positive_probability(bundle['tree'], [args.prompt])[0]
    alpha = MODES['fixed_ensemble']
    score = float(alpha * tree + (1 - alpha) * svm)
    threshold = float(policy.iloc[0].threshold)
    print(json.dumps({'variant': variant, 'score': score, 'threshold': threshold,
                      'model': MODEL_NAMES[int(score >= threshold)],
                      'budget_validation': args.budget,
                      'note': 'Chỉ định tuyến. Budget là mức tổng trên validation, không đảm bảo cho từng prompt hay dữ liệu mới.'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
