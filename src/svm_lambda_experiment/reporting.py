"""Báo cáo diện tích, toàn bộ đường và các chính sách được chọn trước test."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/adaptive_llm_router_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.svm_lambda_experiment.curves import best_for_budgets
from src.svm_lambda_experiment.common import VARIANTS, MODES


def draw_curves(reports, selection, curves, budgets):
    colors = dict(zip(VARIANTS, ['#222222', '#888888', '#0072B2', '#009E73', '#E69F00', '#CC79A7']))
    for mode in MODES:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, split in zip(axes, ['validation', 'test']):
            for variant in VARIANTS:
                curve = curves[(curves.split == split) & (curves['mode'] == mode) & (curves.variant == variant)]
                upper = curve.loc[curve.threshold == 1.01, 'cost_saving'].iloc[0]
                inside = curve.cost_saving.to_numpy()
                x = np.unique(np.r_[0., inside[(inside > 0) & (inside < upper)], upper])
                y = best_for_budgets(curve, x).answer_score.to_numpy()
                label = variant.replace('lambda_', 'lambda=')
                selected = variant == selection['selected_lambda']
                if selected:
                    label += ' (selected)'
                # Đường mảnh là mọi policy, đường đậm là budget envelope bậc thang.
                ax.plot(curve.cost_saving * 100, curve.answer_score * 100, color=colors[variant], alpha=.12, linewidth=.5)
                ax.step(x * 100, y * 100, where='pre', label=label, color=colors[variant], linewidth=2.3 if selected else 1.3)
            ax.set_title('Validation: selection' if split == 'validation' else 'Test: locked threshold sweep (descriptive)')
            ax.set_xlabel('Minimum LLM cost saving (%)')
            ax.grid(alpha=.2)
            ax.set_xlim(0, upper * 100)
        axes[0].set_ylabel('Best reachable answer quality (%)')
        axes[1].legend(fontsize=8, loc='lower left')
        fig.suptitle(f'{mode}: full trade-off curves; CPU embedding cost excluded')
        fig.tight_layout()
        fig.savefig(reports / f'{mode}_curves.png', dpi=160)
        fig.savefig(reports / f'{mode}_curves.svg')
        plt.close(fig)
    # Chênh lệch nhỏ khó thấy khi sáu đường gần nhau: hiển thị thêm Δ quality.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, split in zip(axes, ['validation', 'test']):
        base = curves[(curves.split == split) & (curves['mode'] == 'fixed_ensemble') & (curves.variant == 'tfidf')]
        upper = base.loc[base.threshold == 1.01, 'cost_saving'].iloc[0]
        for variant in VARIANTS[1:]:
            curve = curves[(curves.split == split) & (curves['mode'] == 'fixed_ensemble') & (curves.variant == variant)]
            edges = np.r_[base.cost_saving.to_numpy(), curve.cost_saving.to_numpy()]
            x = np.unique(np.r_[0., edges[(edges > 0) & (edges < upper)], upper])
            delta = best_for_budgets(curve, x).answer_score.to_numpy() - best_for_budgets(base, x).answer_score.to_numpy()
            ax.step(x * 100, delta * 100, where='pre', color=colors[variant],
                    linewidth=2 if variant == selection['selected_lambda'] else 1,
                    label=variant.replace('lambda_', 'lambda='))
        ax.axhline(0, color='#222222', linewidth=.8)
        ax.set(title=split, xlabel='Minimum LLM cost saving (%)', xlim=(0, upper * 100))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Envelope quality minus TF-IDF (percentage points)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Fixed ensemble: difference from TF-IDF; test envelopes are descriptive')
    fig.tight_layout()
    fig.savefig(reports / 'quality_delta_curves.png', dpi=160)
    fig.savefig(reports / 'quality_delta_curves.svg')
    plt.close(fig)
    # Chính sách theo budget chọn trên validation, test giữ nguyên ngưỡng.
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in VARIANTS:
        points = budgets[(budgets.split == 'test') & (budgets['mode'] == 'fixed_ensemble') & (budgets.variant == variant)]
        ax.plot(points.cost_saving * 100, points.answer_score * 100, color=colors[variant],
                linewidth=2 if variant == selection['selected_lambda'] else 1, label=variant.replace('lambda_', 'lambda='))
    ax.set(xlabel='Actual test LLM cost saving (%)', ylabel='Test answer quality (%)',
           title='Validation budget policies transferred unchanged to test')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(reports / 'locked_policies_test.png', dpi=160)
    plt.close(fig)


def write_report(reports, protocol, selection, summary, areas, curves, budgets):
    winner = selection['selected_lambda']
    main = areas[areas['mode'] == 'fixed_ensemble']
    test_area = main[main.split == 'test'].set_index('variant').normalized_area
    area_lines = ['| Đầu vào SVM | C chọn | Diện tích chuẩn hóa validation | Diện tích test (mô tả) |',
                  '| --- | ---: | ---: | ---: |']
    for variant in VARIANTS:
        rows = main[main.variant == variant].set_index('split')
        area_lines.append(f"| {variant} | {selection['configs'][variant]['C']:g} | {rows.loc['validation','normalized_area']:.4%} | {rows.loc['test','normalized_area']:.4%} |")
    budget_lines = ['| Budget yêu cầu trên validation | Đầu vào SVM | Ngưỡng khóa | Đúng validation | Tiết kiệm test thực tế | Đúng test |',
                    '| ---: | --- | ---: | ---: | ---: | ---: |']
    for budget in [.1, .2, .3, .4, .5, .6]:
        subset = budgets[(budgets.split == 'test') & (budgets['mode'] == 'fixed_ensemble') & np.isclose(budgets.target_saving_validation, budget)]
        for r in subset.itertuples():
            budget_lines.append(f'| {budget:.0%} | {r.variant} | {r.threshold:.6f} | {r.validation_quality:.2%} | {r.cost_saving:.2%} | {r.answer_score:.2%} |')
    paired = []
    for reference, r in summary['paired_at_locked_20pct'].items():
        lo, hi = r['answer_delta_95ci']
        paired.append(f"- {winner} so với {reference}: Δ đúng {r['answer_delta']*100:+.2f} điểm %, CI95% [{lo*100:+.2f}, {hi*100:+.2f}]; "
                      f"Δ tiết kiệm {r['saving_delta']*100:+.2f} điểm %, cứu {r['gained_correct']}, mất {r['lost_correct']} câu đúng.")
    text = f"""# Trọng số MiniLM λ và toàn bộ đường Answer Quality–Cost Saving

## Lựa chọn đã khóa trên validation

- **λ được chọn: {winner.removeprefix('lambda_')}**, C = {selection['configs'][winner]['C']:g}.
- Trong cả sáu biểu diễn, diện tích validation cao nhất: **{selection['selected_overall']}**.
- Cây v1 giữ nguyên; chính: **10% cây + 90% SVM**. SVM riêng chỉ là phân tích bổ sung.
- Thử λ = 0.25 / 0.5 / 1 / 2 và C = 0.1 / 1 / 10. Hai mốc TF-IDF, embedding riêng được thử cùng C: **18 cấu hình**.
- Thời gian huấn luyện/chọn: {selection['training_seconds']:.1f} giây; dùng embedding cache cũ.

Trên test, diện tích của λ được chọn cao hơn TF-IDF {(test_area[winner]-test_area['tfidf'])*100:+.3f} điểm phần trăm
và cao hơn λ=1 {(test_area[winner]-test_area['lambda_1'])*100:+.3f} điểm phần trăm chất lượng trung bình theo budget.
Đây là mức tăng nhỏ trên toàn miền, không có nghĩa tốt hơn tại mọi mức tiết kiệm.
Tiêu chí này khác thí nghiệm trước (tối thiểu chi phí với giới hạn giảm đúng 2 điểm), nên lựa chọn khác không mâu thuẫn với kết quả trước.

## Công thức và tiêu chí diện tích

`X = L2_normalize([TF-IDF, λ × MiniLM])`.
TF-IDF và MiniLM vốn có norm 1; phần MiniLM được nhân λ **trước** chuẩn hóa chung.
Không chuẩn hóa từng block lần nữa sau phép nhân, vì làm vậy sẽ triệt tiêu λ.
λ=1 phải tái tạo xác suất của thí nghiệm kết hợp trước ở cùng C.

Với mỗi ngân sách tiết kiệm tối thiểu b, lấy chất lượng tốt nhất trong các ngưỡng thỏa saving ≥ b:

`Q*(b) = max_threshold AnswerQuality(threshold), subject to CostSaving(threshold) >= b`.

Điểm chọn là `integral Q*(b) db / S_small`, với b từ 0 đến mức tiết kiệm always Mistral.
Miền validation: 0–{protocol['validation_area_domain'][1]:.2%}. Tích phân chính xác đường bậc thang,
không nội suy thành chính sách trộn chưa triển khai. Diện tích chuẩn hóa là **chất lượng trung bình
trên các yêu cầu tiết kiệm**, không phải accuracy của một ngưỡng hay ROC-AUC.
Mọi mức tiết kiệm có trọng số đều theo độ rộng miền; tiêu chí này có thể ưu tiên λ khác khi chỉ quan tâm vùng 20%.
Miền test dùng mức tiết kiệm always Mistral của test, chung cho mọi variant; không so trực tiếp chênh lệch diện tích validation–test như cùng miền dữ liệu.

{chr(10).join(area_lines)}

## Toàn bộ đường đánh đổi

Đường mảnh: mọi ngưỡng. Đường đậm: đường biên chất lượng tốt nhất tại mỗi yêu cầu tiết kiệm.
Mọi score khác nhau đều được xét, gồm hai đầu luôn Mixtral/Mistral; không chỉ quét bước 0.01.
Score bằng nhau được xử lý cùng lúc. Nhánh test **chỉ dùng các ngưỡng đã lưu từ validation**.
Đường biên/diện tích test dùng nhãn test để mô tả kết quả đạt được; không dùng chúng chọn ngưỡng triển khai hoặc chọn lại λ.

![Đường fixed ensemble](fixed_ensemble_curves.png)

![SVM riêng, cùng C đã chọn theo chế độ chính](svm_only_curves.png)

### Nhìn rõ chênh lệch khi các đường gần nhau

![Chênh lệch đường biên so với TF-IDF](quality_delta_curves.png)

Đường trên 0 là chất lượng đường biên cao hơn TF-IDF tại cùng yêu cầu tiết kiệm;
đây là so sánh mô tả, không phải khoảng tin cậy hay chính sách được chọn từ test.

## Chính sách chọn trên validation, áp dụng nguyên trạng lên test

Chọn ngưỡng cho 101 mức tiết kiệm cách đều miền validation và thêm các mốc 10/20/30/40/50/60%.
Bảng dưới giữ nguyên ngưỡng đó trên test. Vì phân phối thay đổi, saving test có thể thấp hơn budget validation.
Các hàng cùng budget yêu cầu **không bảo đảm cùng chi phí test thực tế**.

{chr(10).join(budget_lines)}

![Các chính sách khóa trên test](locked_policies_test.png)

### So sánh ghép cặp tại budget validation 20%

{chr(10).join(paired)}

Bootstrap 1000 lần theo nhóm prompt, mô hình và ngưỡng giữ cố định; không điều chỉnh nhiều phép so sánh.
Không diễn giải khác biệt quality thành cải thiện tại cùng chi phí nếu saving test khác nhau.
Các khoảng ở mục này áp dụng cho policy khóa tại budget validation 20%, không phải khoảng tin cậy của toàn đường hay diện tích.

## Giới hạn và tái lập

- Test đã xem trong các thí nghiệm trước: phân tích tiếp nối, không phải holdout độc lập mới.
- Tối ưu đường biên trên validation cũng có thể overfit validation; không chọn lại cấu hình từ test.
- Chi phí LLM là giá trị lưu trong RouterBench, chưa gồm CPU tạo embedding. Không gọi API trả lời.
- TF-IDF fit riêng trong từng fold; encoder đóng băng và không cắt token, cách gộp đoạn giữ nguyên.
- λ điều chỉnh tương quan hai block; C điều chỉnh regularization, vì vậy cả hai được thử cùng lưới.

Xem `docs/SVM_LAMBDA_EXPERIMENT.md` để chạy lại. `protocol.json` ghi thiết kế trước fitting;
`selection.json` khóa λ/C/artifact trước test. `validation_candidates.csv` và `validation_candidate_curves.csv`
ghi tất cả 18 cấu hình. `threshold_curves.csv` chứa mọi ngưỡng của các cấu hình đã chọn;
`pareto_frontiers.csv` và `curve_areas.csv` là tổng hợp. `locked_budget_policies.csv` khóa ngưỡng;
`budget_evaluation.csv` ghi kết quả chuyển ngưỡng sang test; `predictions.csv` đủ để kiểm tra lại.
Ảnh PNG và SVG được lưu cùng báo cáo.
"""
    draw_curves(reports, selection, curves, budgets)
    (reports / 'RESULTS.md').write_text(text, encoding='utf-8')
