from pathlib import Path

import pandas as pd

dataset_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "routerbench_0shot.pkl"
data = pd.read_pickle(dataset_path)

print(f"Kích thước dữ liệu (dòng, cột): {data.shape}")

print("\nTên tất cả các cột:")
for column in data.columns:
    print(f"- {column}")

small_model = "mistralai/mistral-7b-chat"
large_model = "mistralai/mixtral-8x7b-chat"
selected_columns = [
    "sample_id",
    "prompt",
    "eval_name",
    small_model,
    large_model,
    f"{small_model}|total_cost",
    f"{large_model}|total_cost",
]

print("\nBa dòng đầu ở các cột cần dùng:")
print(data[selected_columns].head(3).to_string(index=False, max_colwidth=80))

print("\nSố giá trị thiếu theo cột:")
print(data[selected_columns].isna().sum().to_string())

first_prompt = data.loc[0, "prompt"]
print(f"\nKiểu dữ liệu của prompt đầu tiên: {type(first_prompt).__name__}")
print(f"Prompt đầu tiên (tối đa 200 ký tự): {first_prompt[:200]}")
