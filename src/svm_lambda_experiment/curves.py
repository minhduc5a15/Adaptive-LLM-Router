"""Quét mọi quyết định ngưỡng bằng tổng tích lũy; không giả định cost đơn điệu."""
import numpy as np
import pandas as pd


def threshold_curve(data, scores, thresholds=None):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(data),) or not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError('Điểm định tuyến không hợp lệ')
    if not len(data) or data.large_cost.sum() <= 0:
        raise ValueError('Cần dữ liệu và tổng chi phí Mixtral > 0')
    # ≥ threshold chọn Large. Các score bằng nhau luôn được chuyển cùng lúc.
    if thresholds is None:
        thresholds = np.unique(np.r_[0., scores, 1.01])
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not np.isfinite(thresholds).all():
        raise ValueError('Ngưỡng không hợp lệ')
    order = np.argsort(scores, kind='stable')
    n_small = np.searchsorted(scores[order], thresholds, side='left')
    score_delta = (data.small_score - data.large_score).to_numpy()[order]
    cost_delta = (data.small_cost - data.large_cost).to_numpy()[order]
    correct = data.large_score.sum() + np.r_[0, np.cumsum(score_delta)][n_small]
    costs = data.large_cost.sum() + np.r_[0., np.cumsum(cost_delta)][n_small]
    # Cố định chính xác hai đầu mút, tránh sai số cộng float tại budget tối đa.
    costs[n_small == 0] = data.large_cost.sum()
    costs[n_small == len(data)] = data.small_cost.sum()
    return pd.DataFrame({'threshold': thresholds, 'answer_score': correct / len(data),
                         'correct_count': correct.astype(int), 'total_cost': costs,
                         'cost_saving': 1 - costs / data.large_cost.sum(),
                         'large_rate': 1 - n_small / len(data)})


def best_for_budgets(curve, budgets):
    """Max quality với saving ≥ budget; hòa ưu tiên tiết kiệm rồi ngưỡng lớn."""
    ordered = curve.sort_values(['cost_saving', 'answer_score', 'threshold']).reset_index(drop=True)
    quality = ordered.answer_score.to_numpy()
    best = np.empty(len(ordered), dtype=int)
    winner = len(ordered) - 1
    for index in range(len(ordered) - 1, -1, -1):
        if quality[index] > quality[winner]:
            winner = index
        best[index] = winner
    starts = np.searchsorted(ordered.cost_saving.to_numpy(), budgets, side='left')
    if np.any(starts == len(ordered)):
        raise ValueError('Có budget không đạt được bằng các ngưỡng đã cho')
    selected = ordered.iloc[best[starts]].copy().reset_index(drop=True)
    selected.insert(0, 'target_saving', budgets)
    return selected


def frontier(curve):
    """Các điểm không bị điểm khác vượt cả quality và saving; không nội suy trộn policy."""
    ordered = curve.sort_values(['cost_saving', 'answer_score', 'threshold'], ascending=False)
    best_quality = -np.inf
    keep = []
    for index, row in ordered.iterrows():
        if row.answer_score > best_quality:
            keep.append(index)
            best_quality = row.answer_score
    return curve.loc[keep].sort_values('cost_saving').reset_index(drop=True)


def normalized_area(curve, upper_saving):
    """Tích phân chính xác đường bậc thang Q*(b), chia độ rộng miền budget."""
    if not 0 < upper_saving <= curve.cost_saving.max():
        raise ValueError('Miền diện tích không khả thi')
    inside = curve.cost_saving.to_numpy()
    edges = np.unique(np.r_[0., inside[(inside > 0) & (inside < upper_saving)], upper_saving])
    widths = np.diff(edges)
    midpoints = (edges[:-1] + edges[1:]) / 2
    values = best_for_budgets(curve, midpoints).answer_score.to_numpy()
    return float(np.dot(widths, values) / upper_saving)
