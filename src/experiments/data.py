"""Đọc, kiểm tra và ghi dấu vết dữ liệu; không chia lại dữ liệu đã dùng ở v1."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_path(name):
    return DATA_DIR / f"routerbench_0shot_{name}.csv"


def load_split(name):
    data = pd.read_csv(split_path(name))
    columns = ["sample_id", "eval_name", "prompt_text", "label", "small_score", "large_score", "small_cost", "large_cost"]
    if data.empty or not set(columns).issubset(data.columns) or data[columns].isna().any().any():
        raise ValueError(f"{name}: thiếu dữ liệu")
    if not data.prompt_text.map(lambda p: isinstance(p, str) and bool(p.strip())).all():
        raise ValueError(f"{name}: prompt không hợp lệ")
    if data.sample_id.duplicated().any():
        raise ValueError(f"{name}: sample_id trùng")
    if not data[["label", "small_score", "large_score"]].isin([0, 1]).all().all():
        raise ValueError(f"{name}: điểm không phải nhị phân")
    expected = ((data.small_score == 0) & (data.large_score == 1)).astype(int)
    if not expected.equals(data.label):
        raise ValueError(f"{name}: nhãn không khớp điểm LLM")
    costs = data[["small_cost", "large_cost"]].to_numpy()
    if not np.isfinite(costs).all() or (costs < 0).any() or data.large_cost.sum() <= 0:
        raise ValueError(f"{name}: chi phí không hợp lệ")
    return data


def check_disjoint(left, right):
    for column in ["sample_id", "prompt_text"]:
        if not set(left[column]).isdisjoint(right[column]):
            raise ValueError(f"Rò rỉ giữa hai tập: {column}")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
