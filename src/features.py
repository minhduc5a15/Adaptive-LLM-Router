"""Đặc trưng chỉ dùng nội dung prompt, dùng chung khi train và dự đoán."""

import pandas as pd


def extract_structure_features(text) -> pd.DataFrame:
    text = pd.Series(text)
    return pd.DataFrame({
        "char_count": text.str.len(),
        "word_count": text.str.split().str.len(),
        "line_count": text.str.count("\n") + 1,
    })
