"""Dự đoán LLM nên chọn cho một prompt mới từ mô hình đã lưu."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.router import load_router, route_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    try:
        bundle = load_router()
        result = route_prompts(bundle, [args.prompt], threshold=args.threshold)
    except FileNotFoundError:
        parser.error("Chưa có mô hình. Hãy chạy src/train_models.py trước.")
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({
        "policy": bundle["selected_policy"]["name"],
        "threshold": bundle["selected_policy"]["threshold"] if args.threshold is None else args.threshold,
        "tree_probability": float(result["tree_probability"][0]),
        "svm_probability": float(result["svm_probability"][0]),
        "combined_probability": float(result["probability"][0]),
        "route": int(result["route"][0]), "model": result["model"][0],
        "note": "Chỉ định tuyến; chưa gọi LLM để tạo câu trả lời.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
