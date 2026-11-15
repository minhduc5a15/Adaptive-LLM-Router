"""Offline quality and cost evaluation using benchmark results."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def routing_summary(
    choose_large: Sequence[bool],
    small_quality: Sequence[float],
    large_quality: Sequence[float],
    small_cost: Sequence[float],
    large_cost: Sequence[float],
) -> dict[str, float]:
    """Summarize the outcome of a routing policy on matched prompts."""
    route = np.asarray(choose_large, dtype=bool)
    quality_small = np.asarray(small_quality, dtype=float)
    quality_large = np.asarray(large_quality, dtype=float)
    cost_small = np.asarray(small_cost, dtype=float)
    cost_large = np.asarray(large_cost, dtype=float)
    if not route.size or any(
        len(values) != len(route)
        for values in (quality_small, quality_large, cost_small, cost_large)
    ):
        raise ValueError("All inputs must have the same nonzero length")
    selected_quality = np.where(route, quality_large, quality_small)
    selected_cost = np.where(route, cost_large, cost_small)
    return {
        "mean_quality": float(np.mean(selected_quality)),
        "total_cost": float(np.sum(selected_cost)),
        "large_call_rate": float(np.mean(route)),
    }
