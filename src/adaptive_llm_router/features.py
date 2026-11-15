"""Prompt-only features for the interpretable Decision Tree baseline."""

from __future__ import annotations

import re

import pandas as pd


def structural_features(prompts: list[str]) -> pd.DataFrame:
    """Extract features available before either LLM is called."""
    rows = []
    for prompt in prompts:
        words = re.findall(r"\b\w+\b", prompt)
        rows.append(
            {
                "character_count": len(prompt),
                "word_count": len(words),
                "line_count": prompt.count("\n") + 1,
                "digit_count": sum(character.isdigit() for character in prompt),
                "question_mark_count": prompt.count("?"),
                "has_code_fence": int("```" in prompt),
                "math_symbol_count": sum(
                    prompt.count(symbol) for symbol in ("=", "+", "*", "^", "∑")
                ),
            }
        )
    return pd.DataFrame(rows)
