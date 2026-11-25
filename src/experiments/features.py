"""Đặc trưng mở rộng chỉ đọc prompt; giữ nguyên ba đặc trưng v1."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.features import extract_structure_features


def extract_rich_features(text):
    """Các số đếm là dấu hiệu hình thức, không phải phép đo độ khó thực sự.

    sentence_end_count đếm cụm dấu kết câu, không gọi là số câu ngữ nghĩa.
    code_fence_count đếm dấu ``` (một khối hoàn chỉnh thường có hai dấu).
    """
    text = pd.Series(text)
    result = extract_structure_features(text)
    result["code_fence_count"] = text.str.count(r"```")
    result["math_symbol_count"] = text.str.count(r"[+\-*/=<>^%√∑∫≤≥≠]")
    result["question_mark_count"] = text.str.count(r"\?")
    result["digit_ratio"] = text.str.count(r"\d") / result["char_count"].clip(lower=1)
    result["avg_word_length"] = text.map(
        lambda value: sum(map(len, value.split())) / max(len(value.split()), 1))
    result["sentence_end_count"] = text.str.count(r"[.!?]+")
    return result


def make_text_features(profile):
    if profile == "basic":
        return TfidfVectorizer(max_features=5000)
    word = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=3)
    if profile == "word":
        return word
    if profile == "word_char":
        # Mỗi nhánh chuẩn hóa L2 riêng, FeatureUnion ghép sparse matrices.
        # Không chuyển sang ma trận dense; C được chọn lại cho mỗi profile.
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               max_features=20000, min_df=3)
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(f"Text profile không hợp lệ: {profile}")
