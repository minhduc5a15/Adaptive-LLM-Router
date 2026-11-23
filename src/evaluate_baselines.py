"""Đo hai cách định tuyến cố định trên tập validation."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
VALIDATION_PATH = PROJECT_DIR / "data" / "processed" / "routerbench_0shot_validation.csv"


def main() -> None:
    data = pd.read_csv(VALIDATION_PATH)

    # 0 = chọn Mistral 7B (Small), 1 = chọn Mixtral 8×7B (Large).
    routes = {
        "Luôn chọn Mistral": np.zeros(len(data), dtype=int),
        "Luôn chọn Mixtral": np.ones(len(data), dtype=int),
    }

    for name, route in routes.items():
        # Mỗi dòng chỉ lấy điểm và chi phí của mô hình đã chọn.
        scores = np.where(route == 1, data["large_score"], data["small_score"])
        costs = np.where(route == 1, data["large_cost"], data["small_cost"])

        print(name)
        print(f"  Tỷ lệ câu được chấm đúng: {scores.mean():.2%}")
        print(f"  Chi phí trung bình / câu: {costs.mean():.8f}")
        print(f"  Tổng chi phí: {costs.sum():.6f}")
        print(f"  Tỷ lệ gọi Mixtral: {route.mean():.2%}")
        print(f"  Accuracy so với nhãn Router: {(route == data['label'].to_numpy()).mean():.2%}")


if __name__ == "__main__":
    main()
