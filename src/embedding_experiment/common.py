"""Cấu hình cố định, dấu vết file và kiểm tra dữ liệu."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENCODER_DIR = ROOT / "models" / "embedding_encoder"
MAX_TOKENS = 256
DIMENSION = 384
SEED = 42
VARIANTS = ["structure", "embedding", "combined"]
MODES = {"tree_only": [1.0], "fixed_ensemble": [0.1],
         "tuned_router": np.linspace(0, 1, 11).tolist()}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def split_path(name):
    return ROOT / "data" / "processed" / f"routerbench_0shot_{name}.csv"


def load_split(name):
    data = pd.read_csv(split_path(name))
    if data.empty or data.isna().any().any() or data.sample_id.duplicated().any():
        raise ValueError(f"Dữ liệu {name} thiếu hoặc trùng ID")
    if not data.prompt_text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Prompt không hợp lệ")
    if not data[["label", "small_score", "large_score"]].isin([0, 1]).all().all():
        raise ValueError("Chỉ dùng điểm nhị phân")
    if not np.array_equal(data.label, ((data.small_score == 0) & (data.large_score == 1)).astype(int)):
        raise ValueError("Nhãn không khớp điểm LLM")
    costs = data[["small_cost", "large_cost"]].to_numpy()
    if not np.isfinite(costs).all() or (costs < 0).any() or data.large_cost.sum() <= 0:
        raise ValueError("Chi phí không hợp lệ")
    return data


def check_disjoint(left, right):
    for column in ["sample_id", "prompt_text"]:
        if not set(left[column]).isdisjoint(right[column]):
            raise ValueError(f"Rò rỉ {column} giữa hai tập")


def run_dirs(name):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Tên run phải là tên thư mục đơn")
    return (ROOT / "reports" / "embedding" / name,
            ROOT / "models" / "embedding" / name,
            ROOT / "data" / "processed" / "embeddings" / name)
