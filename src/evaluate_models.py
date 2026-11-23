"""Tạo lại báo cáo bằng mô hình đã khóa, không chọn lại cấu hình."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting import evaluate_and_report
from src.router import load_router
from src.train_models import REPORT_DIR, load_split, split_hash


def main():
    bundle = load_router()
    for split in ["train", "validation"]:
        if split_hash(split) != bundle[f"{split}_sha256"]:
            raise ValueError(f"Dữ liệu {split} đã đổi; cần huấn luyện lại trước khi đánh giá")
    train, validation, test = (load_split(name) for name in ["train", "validation", "test"])
    assert set(test["prompt_text"]).isdisjoint(train["prompt_text"])
    assert set(test["prompt_text"]).isdisjoint(validation["prompt_text"])
    evaluate_and_report(bundle, train, validation, test, REPORT_DIR, split_hash("test"))


if __name__ == "__main__":
    main()
