"""Thiết kế cố định và đường dẫn riêng; tái sử dụng encoder của thí nghiệm cây."""
import numpy as np
from src.embedding_experiment.common import ROOT, run_dirs

DEFAULT_RUN = "minilm_svm_2026-11-25"
CACHE_RUN = "minilm_2026-11-25"
VARIANTS = ["tfidf", "embedding", "combined"]
MODES = {"svm_only": [0.0], "fixed_ensemble": [0.1],
         "tuned_router": np.linspace(0, 1, 11).tolist()}


def paths(name):
    run_dirs(name)  # Kiểm tra tên thư mục đơn.
    return ROOT / "reports/svm_embedding" / name, ROOT / "models/svm_embedding" / name


def source_hashes():
    from src.embedding_experiment.common import sha256
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))}
