"""Thiết kế và đường dẫn; giữ nguyên nguồn và artifact các thí nghiệm cũ."""
from src.embedding_experiment.common import ROOT, run_dirs

DEFAULT_RUN = 'lambda_2026-11-26'
CACHE_RUN = 'minilm_2026-11-25'
PREVIOUS_RUN = 'minilm_svm_2026-11-25'
LAMBDAS = [0.25, 0.5, 1., 2.]
VARIANTS = ['tfidf', 'embedding'] + [f'lambda_{value:g}' for value in LAMBDAS]
MODES = {'fixed_ensemble': 0.1, 'svm_only': 0.0}


def paths(name):
    run_dirs(name)  # Validate tên thư mục.
    return ROOT / 'reports/svm_lambda' / name, ROOT / 'models/svm_lambda' / name
