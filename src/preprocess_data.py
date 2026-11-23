"""Chuẩn bị dữ liệu 0-shot cho bài toán chọn Mistral hoặc Mixtral."""

import ast
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "data" / "raw" / "routerbench_0shot.pkl"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

SMALL_MODEL = "mistralai/mistral-7b-chat"
LARGE_MODEL = "mistralai/mixtral-8x7b-chat"
RANDOM_STATE = 42
HAN_PATTERN = r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"


def parse_prompt(value: str) -> str:
    """Đổi chuỗi biểu diễn danh sách thành văn bản, giữ ranh giới các đoạn."""
    parts = ast.literal_eval(value)
    if not isinstance(parts, list) or not all(isinstance(part, str) for part in parts):
        raise ValueError("Prompt không phải danh sách các chuỗi")
    return "\n\n".join(part.strip() for part in parts)


def main() -> None:
    # Pickle chỉ nên được nạp từ nguồn đã kiểm tra checksum (xem DATA_PREPARATION.md).
    raw = pd.read_pickle(INPUT_PATH)

    # Điểm lẻ chưa có quy tắc nhãn rõ ràng, nên chỉ giữ cặp điểm 0/1.
    binary_scores = raw[[SMALL_MODEL, LARGE_MODEL]].isin([0, 1]).all(axis=1)
    data = raw.loc[binary_scores].copy()

    data["prompt_text"] = data["prompt"].map(parse_prompt)
    if data["prompt_text"].eq("").any():
        raise ValueError("Có prompt rỗng sau khi chuyển thành văn bản")

    # Lọc prompt chứa chữ Hán trước khi chia tập, kể cả prompt trộn ngôn ngữ.
    contains_han = data["prompt_text"].str.contains(HAN_PATTERN, regex=True)
    data = data.loc[~contains_han].copy()

    # 1 = Small sai, Large đúng; 0 = chọn Small trong ba trường hợp còn lại.
    data["label"] = ((data[SMALL_MODEL] == 0) & (data[LARGE_MODEL] == 1)).astype(int)

    # Chỉ prompt_text là đặc trưng đầu vào. Các cột khác dùng để kiểm tra
    # dữ liệu và tính chất lượng/chi phí sau khi Router dự đoán.
    data = data.rename(
        columns={
            SMALL_MODEL: "small_score",
            LARGE_MODEL: "large_score",
            f"{SMALL_MODEL}|total_cost": "small_cost",
            f"{LARGE_MODEL}|total_cost": "large_cost",
        }
    )[
        [
            "sample_id",
            "eval_name",
            "prompt_text",
            "label",
            "small_score",
            "large_score",
            "small_cost",
            "large_cost",
        ]
    ]

    # Chia theo prompt_text để các bản ghi có cùng prompt không lọt vào
    # hai tập khác nhau. Tỷ lệ xấp xỉ 80% / 10% / 10%.
    first_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, holdout_idx = next(first_split.split(data, groups=data["prompt_text"]))
    train = data.iloc[train_idx]
    holdout = data.iloc[holdout_idx]

    second_split = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE + 1)
    validation_idx, test_idx = next(
        second_split.split(holdout, groups=holdout["prompt_text"])
    )
    validation = holdout.iloc[validation_idx]
    test = holdout.iloc[test_idx]

    assert len(train) + len(validation) + len(test) == len(data)
    assert set(train["prompt_text"]).isdisjoint(validation["prompt_text"])
    assert set(train["prompt_text"]).isdisjoint(test["prompt_text"])
    assert set(validation["prompt_text"]).isdisjoint(test["prompt_text"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("validation", validation), ("test", test)]:
        output_path = OUTPUT_DIR / f"routerbench_0shot_{name}.csv"
        split.to_csv(output_path, index=False, encoding="utf-8")
        print(f"{name}: {len(split):,} dòng; nhãn 1: {split['label'].sum():,} -> {output_path}")

    print(f"Tổng: {len(raw):,} dòng; giữ: {len(data):,}; loại điểm lẻ: {(~binary_scores).sum():,}")
    print(f"Loại prompt chứa chữ Hán sau khi lọc điểm: {contains_han.sum():,}")
    print(f"Prompt trùng sau chuẩn hóa: {data['prompt_text'].duplicated().sum():,}")


if __name__ == "__main__":
    main()
