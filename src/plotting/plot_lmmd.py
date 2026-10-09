"""Compare the one new LMMD run with the five frozen CORAL controls."""

import json
import math
import os
import shutil
from datetime import date

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.paths import ROOT
from src.experiments.lmmd.protocol import REPORTS_DIR, RUN_DIR
from src.experiments.lmmd.verify import mathematical_checks, read_lmmd_run, verify_warmup
from src.experiments.subdomain.protocol import RUNS_DIR, SETTINGS as CONTROL_SETTINGS
from src.experiments.subdomain.verify import read_csv, verify_suite
from src.plotting.plot_subdomain import COLORS as CONTROL_COLORS, LABELS as CONTROL_LABELS, STYLES as CONTROL_STYLES
from src.plotting.plot_subdomain import finish_figure, frozen_diagnostics, temporal_axis
from src.training.evaluation import write_csv
from src.training.train import write_json

SETTINGS = (*CONTROL_SETTINGS, 'lmmd')
LABELS = dict(CONTROL_LABELS, lmmd='LMMD 0.5')
COLORS = dict(CONTROL_COLORS, lmmd='#b7791f')
STYLES = dict(CONTROL_STYLES, lmmd=(0, (3, 1, 1, 1, 1, 1)))


def curves(axis, runs, field, factor=1):
    for setting in SETTINGS:
        rows = runs[setting][1]
        axis.plot([row['epoch'] for row in rows], [row[field] * factor for row in rows], label=LABELS[setting],
                  color=COLORS[setting], linestyle=STYLES[setting], linewidth=1.7,
                  marker='o' if setting == 'lmmd' else None, markevery=5, markersize=3)


def figures(runs, summaries, diagnostics, per_class, confusion, classes):
    assets = REPORTS_DIR / 'assets'
    subtitle = 'AlexNet fc8 | Amazon to Webcam | seed 42 | adaptation runs: global warm-up through epoch 10'
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), layout='constrained')
    for axis, field, title in zip(axes, ['target_accuracy', 'target_macro_accuracy'], ['Target accuracy', 'Target macro class accuracy']):
        curves(axis, runs, field, 100)
        temporal_axis(axis, title, 'Accuracy (%)')
        axis.set_ylim(0, 100)
    axes[0].legend(ncol=2, fontsize=9)
    fig.suptitle(subtitle, fontsize=11)
    finish_figure(fig, assets / 'accuracy.png')

    fig, axis = plt.subplots(figsize=(11, 5.5), layout='constrained')
    bars = axis.bar(range(6), [row['target_accuracy'] * 100 for row in summaries],
                    color=[COLORS[setting] for setting in SETTINGS], edgecolor='#334155', linewidth=.6)
    axis.bar_label(bars, labels=[f'{row["target_accuracy"]:.2%}\n{row["target_correct"]}/795' for row in summaries], padding=5)
    axis.set_xticks(range(6), [LABELS[setting] for setting in SETTINGS])
    axis.set(ylim=(0, 100), ylabel='Target accuracy (%)', title='Epoch 50 | one new LMMD run + five frozen controls | seed 42')
    axis.grid(axis='y', alpha=.18)
    axis.set_axisbelow(True)
    finish_figure(fig, assets / 'final_comparison.png')

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2), layout='constrained')
    for axis, field, title in [(axes[0, 0], 'global_loss', 'Monitored global CORAL (raw)'),
                              (axes[0, 1], 'local_loss', 'Monitored class-conditional CORAL (raw)'),
                              (axes[1, 0], 'adaptation_loss', 'Optimized adaptation term')]:
        curves(axis, runs, field)
        temporal_axis(axis, title, 'Loss')
        if field != 'adaptation_loss':
            axis.set_yscale('symlog', linthresh=.001)
            axis.set_ylabel('Distance (symlog; linear < 0.001)')
        axis.set_ylim(bottom=0)
    lmmd_rows = runs['lmmd'][1]
    for field, label, style in [('lmmd_loss', 'Raw LMMD', '-'), ('weighted_lmmd_loss', 'Weighted LMMD', '--')]:
        axes[1, 1].plot([row['epoch'] for row in lmmd_rows], [row[field] for row in lmmd_rows],
                        label=label, color=COLORS['lmmd'], linestyle=style)
    temporal_axis(axes[1, 1], 'LMMD control only | unavailable for frozen runs', 'Loss')
    axes[1, 1].set_ylim(bottom=0)
    axes[0, 0].legend(ncol=2, fontsize=9)
    axes[1, 1].legend(fontsize=9)
    fig.suptitle(subtitle, fontsize=11)
    finish_figure(fig, assets / 'losses.png')

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2), layout='constrained')
    for axis, field, title, ylabel, upper in [
            (axes[0, 0], 'valid_classes', 'Monitored CORAL-valid classes', 'Classes', 31),
            (axes[0, 1], 'target_confidence', 'Target training confidence', 'Mean max probability', 1),
            (axes[1, 1], 'target_entropy', 'Target training entropy', 'Entropy (nats)', math.log(31))]:
        curves(axis, runs, field)
        temporal_axis(axis, title, ylabel)
        axis.set_ylim(0, upper)
    for field, label, color, style in [('lmmd_valid_classes', 'LMMD shared (>=1)', COLORS['lmmd'], '-'),
                                      ('valid_classes', 'CORAL valid (>=2 + ESS)', '#64748b', '--')]:
        axes[1, 0].plot([row['epoch'] for row in lmmd_rows], [row[field] for row in lmmd_rows],
                        label=label, color=color, linestyle=style)
    temporal_axis(axes[1, 0], 'Two support rules in the LMMD run', 'Classes')
    axes[1, 0].set_ylim(0, 31)
    axes[0, 0].legend(ncol=2, fontsize=9)
    axes[1, 0].legend(fontsize=9)
    fig.suptitle(subtitle, fontsize=11)
    finish_figure(fig, assets / 'coverage.png')

    focus = ('global', 'local', 'mixed', 'lmmd')
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
    for axis, field, title in zip(axes.flat,
            ['global_covariance_distance', 'true_local_covariance_distance', 'centroid_mse',
             'target_mean_logit_norm', 'target_covariance_trace', 'target_effective_rank'],
            ['Global covariance distance', 'True-class covariance distance', 'True-class centroid MSE',
             'Target mean logit norm', 'Target covariance trace', 'Target covariance effective rank']):
        values = [diagnostics[setting][field] for setting in focus]
        bars = axis.barh(range(4), values, color=[COLORS[setting] for setting in focus])
        axis.bar_label(bars, labels=[f'{value:.4g}' for value in values], padding=3, fontsize=9)
        axis.set_yticks(range(4), [LABELS[setting] for setting in focus])
        axis.invert_yaxis()
        axis.set(title=title, xlim=(0, max(values) * 1.3 if max(values) else 1))
        axis.grid(axis='x', alpha=.18)
        axis.set_axisbelow(True)
    fig.suptitle('Frozen epoch-50 outputs | four focal settings | all six settings retained in diagnostics.csv', fontsize=11)
    finish_figure(fig, assets / 'alignment_diagnostics.png')

    recalls = np.array([[row['target_accuracy'] * 100 for row in per_class[setting]] for setting in SETTINGS]).T
    baselines = ('local', 'global', 'mixed')
    deltas = np.column_stack([recalls[:, -1] - recalls[:, SETTINGS.index(setting)] for setting in baselines])
    fig, axes = plt.subplots(1, 2, figsize=(16, 12), layout='constrained')
    absolute = axes[0].imshow(recalls, vmin=0, vmax=100, cmap='Blues', aspect='auto')
    limit = max(1, float(np.abs(deltas).max()))
    relative = axes[1].imshow(deltas, vmin=-limit, vmax=limit, cmap='PuOr', aspect='auto')
    for axis in axes:
        axis.set_yticks(range(31), [name.replace('_', ' ') for name in classes], fontsize=9)
    axes[0].set_xticks(range(6), [LABELS[setting].replace(' ', '\n', 1) for setting in SETTINGS])
    axes[1].set_xticks(range(3), [f'LMMD vs\n{LABELS[setting]}' for setting in baselines])
    axes[0].set_title('Epoch-50 target class accuracy (%)')
    axes[1].set_title('LMMD change (percentage points)')
    fig.colorbar(absolute, ax=axes[0], shrink=.7)
    fig.colorbar(relative, ax=axes[1], shrink=.7)
    finish_figure(fig, assets / 'class_accuracy.png')

    normalized = 100 * confusion / confusion.sum(1, keepdims=True)
    fig, axis = plt.subplots(figsize=(12, 10.5), layout='constrained')
    heatmap = axis.imshow(normalized, cmap='Blues', vmin=0, vmax=100, interpolation='nearest')
    axis.set_xticks(range(31), [name.replace('_', ' ') for name in classes], rotation=60, ha='right', fontsize=8)
    axis.set_yticks(range(31), [name.replace('_', ' ') for name in classes], fontsize=8)
    axis.set(title='LMMD 0.5 | epoch 50 target confusion | 795 images', xlabel='Predicted class', ylabel='True class')
    fig.colorbar(heatmap, ax=axis, label='Share within true class (%)', shrink=.8)
    finish_figure(fig, assets / 'confusion_lmmd.png')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    epochs = [row['epoch'] for row in lmmd_rows]
    axes[0, 0].plot(epochs, [row['source_accuracy'] * 100 for row in lmmd_rows], color='#64748b', linestyle='--', label='Source (training images)')
    axes[0, 0].plot(epochs, [row['target_accuracy'] * 100 for row in lmmd_rows], color=COLORS['lmmd'], label='Target')
    axes[0, 0].set_ylim(0, 100)
    for field, label, color, style in [('classification_loss', 'CE', '#64748b', '-'),
                                      ('weighted_global_loss', 'Weighted global', '#2563eb', '--'),
                                      ('weighted_lmmd_loss', 'Weighted LMMD', COLORS['lmmd'], '-.')]:
        axes[0, 1].plot(epochs, [row[field] for row in lmmd_rows], color=color, linestyle=style, label=label)
    for field, label, color, style in [('lmmd_valid_classes', 'LMMD shared', COLORS['lmmd'], '-'),
                                      ('valid_classes', 'CORAL valid (monitor)', '#64748b', '--')]:
        axes[1, 0].plot(epochs, [row[field] for row in lmmd_rows], color=color, linestyle=style, label=label)
    axes[1, 0].set_ylim(0, 31)
    axes[1, 1].plot(epochs, [row['target_confidence'] for row in lmmd_rows], label='Confidence', color=COLORS['lmmd'])
    axes[1, 1].plot(epochs, [row['target_entropy'] / math.log(31) for row in lmmd_rows], label='Entropy / log(31)',
                    color='#64748b', linestyle='--')
    axes[1, 1].set_ylim(0, 1)
    for axis, title, ylabel in zip(axes.flat, ['Accuracy', 'Optimized losses', 'Class support', 'Target predictions'],
                                  ['Accuracy (%)', 'Loss', 'Classes', 'Probability / normalized entropy']):
        temporal_axis(axis, title, ylabel)
        axis.legend(fontsize=9)
    fig.suptitle('LMMD control | fc8 | seed 42 | dashed line: global warm-up ends')
    finish_figure(fig, RUN_DIR / 'plots' / 'training.png')
    shutil.copy2(assets / 'confusion_lmmd.png', RUN_DIR / 'plots' / 'confusion.png')


def write_union_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(path, [{key: row.get(key) for key in fields} for row in rows])


def report_text(runs, summaries, diagnostics, pairs, verification):
    lmmd_rows = runs['lmmd'][1]
    final = lmmd_rows[-1]
    table = '\n'.join(f'| {row["setting"]} | {row["target_accuracy"]:.2%} ({row["target_correct"]}/795) | '
                      f'{row["target_macro_accuracy"]:.2%} | {row["gain_vs_global_pp"]:+.2f} | '
                      f'{row["gain_vs_local_pp"]:+.2f} | {"新增训练" if row["new_training"] else "复用"} |' for row in summaries)
    pair_table = '\n'.join(f'| LMMD / {row["baseline"]} | {row["delta_pp"]:+.2f} | {row["corrected"]} | '
                           f'{row["regressed"]} | {row["net_correct"]:+d} |' for row in pairs)
    diagnostic_table = '\n'.join(f'| {setting} | {diagnostics[setting]["global_covariance_distance"]:.6g} | '
                                f'{diagnostics[setting]["true_local_covariance_distance"]:.6g} | '
                                f'{diagnostics[setting]["centroid_mse"]:.6g} | '
                                f'{diagnostics[setting]["target_mean_logit_norm"]:.4f} | '
                                f'{diagnostics[setting]["target_covariance_trace"]:.4f} | '
                                f'{diagnostics[setting]["target_effective_rank"]:.3f} |' for setting in SETTINGS)
    loss_rows = []
    for setting in SETTINGS:
        row = runs[setting][1][-1]
        m = f'{row["lmmd_loss"]:.5f}' if setting == 'lmmd' else '未记录'
        loss_rows.append(f'| {setting} | {row["classification_loss"]:.5f} | {row["global_loss"]:.5f} | '
                         f'{row["local_loss"]:.5f} | {m} | {row["adaptation_loss"]:.5f} |')
    loss_table = '\n'.join(loss_rows)
    phase_rows = []
    for name, rows in [('预热 1–10', lmmd_rows[:10]), ('LMMD 11–50', lmmd_rows[10:])]:
        means = {key: float(np.mean([row[key] for row in rows])) for key in
                 ['lmmd_loss', 'weighted_lmmd_loss', 'lmmd_valid_classes', 'valid_classes', 'lmmd_bandwidth_raw']}
        phase_rows.append(f'| {name} | {means["lmmd_loss"]:.5f} | {means["weighted_lmmd_loss"]:.5f} | '
                          f'{means["lmmd_valid_classes"]:.2f}/31 | {means["valid_classes"]:.2f}/31 | '
                          f'{means["lmmd_bandwidth_raw"]:.4f} |')
    phase_table = '\n'.join(phase_rows)
    local_pair = next(row for row in pairs if row['baseline'] == 'local')
    global_pair = next(row for row in pairs if row['baseline'] == 'global')
    support = sum(row['lmmd_no_shared_classes'] * 6 for row in lmmd_rows)
    clamps = sum(row['lmmd_bandwidth_clamped'] * 6 for row in lmmd_rows)
    author = verification.get('author_implementation')
    author_note = (f'另直接调用作者原始权重和核函数进行 FP32 对照，损失绝对差 '
                   f'{author["absolute_loss_error"]:.3g}，最大特征梯度差 '
                   f'{author["maximum_feature_gradient_error"]:.3g}；见 [author_validation.json](author_validation.json)。'
                   if author else '本次重建未包含在线作者实现核对记录，独立公式检查已通过。')
    outcome = ('本次 LMMD 在两个主要比较中均提高了准确率。' if local_pair['delta_pp'] > 0 and global_pair['delta_pp'] > 0
               else '本次 LMMD 的分类效果由两个主要比较分别评价，不能仅由其损失下降判断有效性。')
    structure_note = ''
    if (local_pair['delta_pp'] > 0 and diagnostics['lmmd']['true_local_covariance_distance']
            > diagnostics['local']['true_local_covariance_distance']):
        structure_note = (f'LMMD 的真实类协方差距离为 {diagnostics["lmmd"]["true_local_covariance_distance"]:.6g}，'
                          f'大于 local 的 {diagnostics["local"]["true_local_covariance_distance"]:.6g}，'
                          '但分类准确率更高。这进一步说明较小的类内协方差距离不足以评价迁移效果。'
                          '中心距离的改善与分类提升同时出现，尚未通过额外消融确定因果关系。')
    return f'''# AlexNet fc8 LMMD 追加对照

日期：{date.today().isoformat()}。新增一组 50 轮/300 步 LMMD 训练，复用[原五组已完成运行](../report.md)，不改写原结果。

固定第 50 轮 LMMD 目标准确率为 **{final['target_accuracy']:.2%}（{int(final['target_correct'])}/795）**，相对类条件 CORAL 为 **{local_pair['delta_pp']:+.2f} 个百分点**，相对全局 CORAL 为 **{global_pair['delta_pp']:+.2f} 个百分点**。

## 固定协议与结果

[追加协议](protocol.md)：沿用 seed=42、AlexNet ImageNet-1K V1、fc8 31 维原始 logits、Amazon→Webcam、batch=128、每轮六步、FP32/严格确定性/禁用 TF32。前十轮 `CE + 0.5G`，第十一轮起 `CE + 0.5 LMMD`，SGD 和预处理保持一致。前十轮共同的逐轮/逐步/逐类指标已与原全局组核对一致。

LMMD 采用 [DSAN 作者实现](https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/lmmd.py) 的五核 RBF 和停止梯度的源/目标类别权重。它对齐类条件核均值嵌入；类条件 CORAL 只约束类内协方差。这是匹配本轮训练配置的损失对照，并非完整 DSAN 复现。

| 设置 | 第 50 轮目标准确率 | 宏平均类别准确率 | 相对 global（百分点） | 相对 local（百分点） | 本次执行 |
|---|---|---|---|---|---|
{table}

![最终比较](assets/final_comparison.png)

![准确率曲线](assets/accuracy.png)

固定最终轮不由目标标签选择。LMMD 最佳准确率 {runs['lmmd'][2]['best_target_accuracy']:.2%}，出现在第 {runs['lmmd'][2]['best_target_epoch']} 轮，仅作补充描述。源域准确率使用源训练图片，不代表独立测试性能。

新增训练循环及逐轮评估耗时 {runs['lmmd'][2]['elapsed_seconds']/60:.2f} 分钟，峰值显存 {runs['lmmd'][2]['peak_gpu_memory_bytes']/2**30:.2f} GiB；不包含进程启动和模型初始化时间。

## 同图像配对比较

| 变体 / 对照 | 准确率变化（百分点） | 对照错误→LMMD 正确 | 对照正确→LMMD 错误 | 净新增正确 |
|---|---|---|---|---|
{pair_table}

全部比较对应相同顺序的 795 张目标图像，不作跨种子显著性推断。原五组配置、环境和完整性重新核验通过；本次只训练 LMMD。

![逐类准确率与变化](assets/class_accuracy.png)

## 损失尺度和类别支持

![损失曲线](assets/losses.png)

下表为第 50 轮六个训练批次均值。原五组未记录同批 LMMD，保留为空/未记录，不以零代替。G 和类条件 CORAL 在 LMMD 组仅作监测；预热后只优化 LMMD。

| 设置 | CE | G 原始值 | 类条件 CORAL 原始值 | LMMD 原始值 | 实际加权适应项 |
|---|---|---|---|---|---|
{loss_table}

LMMD 目标软权重来自同一次训练前向，温度 1，并停止梯度。共享类为源真实标签与目标 top-1 的交集，至少各出现一次；所有目标图像仍按软权重参与。CORAL 的 >=2/ESS>=2 条件只用于监测，不用于 LMMD。下表为 LMMD 运行的批均值。

| 阶段 | 原始 LMMD | 加权 LMMD | LMMD 共享类别 | CORAL 有效类别（监测） | 原始带宽 b |
|---|---|---|---|---|---|
{phase_table}

五个 RBF 分母为 `b/4, b/2, b, 2b, 4b`，核相加，包含 V-statistic 对角线。带宽为拼接批次的非对角平均平方距离，停止梯度。原始值和加权值不与 CORAL 解释为相同约束强度。

本次零共享类别批次为 {round(support)}，带宽下限触发批次为 {round(clamps)}。权重质量、ESS、支持和无效类空距离见 [lmmd_class_metrics.csv](lmmd_class_metrics.csv)；同批 CORAL 监测见 [coral_class_metrics.csv](coral_class_metrics.csv)。

![类别支持与置信度](assets/coverage.png)

## 冻结模型事后诊断

目标真实标签只在评估和本节事后诊断使用，不进入训练损失、权重或支持筛选。全域 logits 关闭 Dropout，与训练批次损失分开解释。下表距离均采用相同 fc8/31 维定义；没有将全域 LMMD 与训练批 LMMD 混用。

| 设置 | 全局协方差距离 | 真实类协方差距离 | 真实类中心 MSE | 目标 logits 范数 | 目标协方差迹 | 目标有效秩 |
|---|---|---|---|---|---|---|
{diagnostic_table}

![四个主要设置的特征诊断](assets/alignment_diagnostics.png)

完整两域范数、迹、有效秩及逐类结构见 [diagnostics.csv](diagnostics.csv)、[class_diagnostics.csv](class_diagnostics.csv)。有效秩沿用协方差特征值归一化后的熵指数。目标混淆矩阵如下，原始计数见 [confusion_counts.csv](confusion_counts.csv)。

![LMMD 混淆矩阵](assets/confusion_lmmd.png)

## 解释与验收

{outcome} LMMD 的真实类中心 MSE 为 {diagnostics['lmmd']['centroid_mse']:.6g}，类条件 CORAL 为 {diagnostics['local']['centroid_mse']:.6g}，全局组为 {diagnostics['global']['centroid_mse']:.6g}。协方差距离、类别中心及幅度/谱变化应与准确率和支持覆盖共同评价。

{structure_note}

本次仍为单种子、单方向探索。LMMD 与自定义类条件 CORAL 在统计对象、核函数、支持筛选和数值尺度上均有差异，因此此对照不能单独确定其中某一因素的因果贡献，也不能代表原论文完整方法的结果。

- 六项数学边界检查通过，特征梯度有限，目标概率不接收梯度。
- {author_note}
- LMMD 完成 50 轮、300 步，保存 9,300 条 LMMD 类记录和 9,300 条 CORAL 监测类记录；第十一轮调度正确，前十轮与原全局组共同指标在 1e-6 内一致。
- CSV、训练日志、最终 NPZ、逐图预测及 2,817/795 张全域准确率互相一致；环境与旧对照一致，逐图路径顺序一致。原五组只复用，未新增训练。

```powershell
uv run --no-sync python -m src.experiments.lmmd.run
uv run --no-sync python -m src.experiments.lmmd.run --resume
uv run --no-sync python -m src.experiments.lmmd.verify
uv run --no-sync python -m src.plotting.plot_lmmd
```

[summary.csv](summary.csv)、[epoch_metrics.csv](epoch_metrics.csv)、[step_metrics.csv](step_metrics.csv)、[prediction_changes.csv](prediction_changes.csv)、[verification.json](verification.json)、[provenance.json](provenance.json) 保留可核对数字。新运行的原始配置、环境、日志、CSV 和最终 logits/预测快照位于 `data/lmmd/`；原五组快照保持在 `../data/<setting>/`，完整新运行位于 `results/subdomain_coral/lmmd/`。
'''


def plot_experiments():
    torch.set_num_threads(1)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titleweight': 'bold'})
    controls_verification, runs = verify_suite(return_runs=True)
    runs['lmmd'] = read_lmmd_run()
    verification = dict(mathematics=mathematical_checks(), controls=controls_verification,
                        new_run=dict(passed=True, epochs=50, steps=300, lmmd_class_rows=9300, coral_monitor_class_rows=9300,
                                     warmup=verify_warmup(), target_training_labels_used=False))
    author_file = REPORTS_DIR / 'author_validation.json'
    if author_file.exists():
        verification['author_implementation'] = json.loads(author_file.read_text(encoding='utf-8'))
        if verification['author_implementation']['passed'] is not True:
            raise ValueError('Author implementation validation failed')
    diagnostics, per_class, confusions, predictions = {}, {}, {}, {}
    summaries, epoch_rows, step_rows, confusion_rows = [], [], [], []
    for setting in SETTINGS:
        config, rows, summary = runs[setting]
        diagnostics[setting], per_class[setting], confusions[setting], predictions[setting] = frozen_diagnostics(setting)
        final = rows[-1]
        summaries.append(dict(setting=setting, epoch=50, target_accuracy=final['target_accuracy'],
                              target_correct=int(final['target_correct']), target_samples=795,
                              target_macro_accuracy=final['target_macro_accuracy'], source_accuracy=final['source_accuracy'],
                              gain_vs_global_pp=100 * (final['target_accuracy'] - runs['global'][1][-1]['target_accuracy']),
                              gain_vs_local_pp=100 * (final['target_accuracy'] - runs['local'][1][-1]['target_accuracy']),
                              new_training=int(setting == 'lmmd'), best_target_accuracy=summary['best_target_accuracy'],
                              best_target_epoch=summary['best_target_epoch'], elapsed_seconds=summary['elapsed_seconds'],
                              peak_gpu_memory_bytes=summary['peak_gpu_memory_bytes']))
        epoch_rows.extend(dict(setting=setting, **row) for row in rows)
        step_rows.extend(dict(setting=setting, **row) for row in read_csv(RUNS_DIR / setting / 'step_metrics.csv'))
        confusion_rows.extend(dict(setting=setting, true_class=truth, predicted_class=predicted,
                                   count=int(confusions[setting][truth, predicted])) for truth in range(31) for predicted in range(31))
    pairs = []
    labels = predictions['lmmd']['target_labels']
    lmmd_correct = predictions['lmmd']['target_logits'].argmax(1) == labels
    for baseline in ('local', 'global', 'mixed', 'global_quarter', 'no_adaptation'):
        if not np.array_equal(labels, predictions[baseline]['target_labels']):
            raise ValueError('LMMD paired labels differ')
        baseline_correct = predictions[baseline]['target_logits'].argmax(1) == labels
        corrected = int(np.sum(lmmd_correct & ~baseline_correct))
        regressed = int(np.sum(~lmmd_correct & baseline_correct))
        pairs.append(dict(variant='lmmd', baseline=baseline, corrected=corrected, regressed=regressed,
                          both_correct=int(np.sum(lmmd_correct & baseline_correct)), both_wrong=int(np.sum(~lmmd_correct & ~baseline_correct)),
                          net_correct=corrected - regressed, delta_pp=100 * (corrected - regressed) / 795))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [('summary', summaries), ('epoch_metrics', epoch_rows), ('step_metrics', step_rows),
                       ('diagnostics', list(diagnostics.values())), ('class_diagnostics', [row for setting in SETTINGS for row in per_class[setting]]),
                       ('confusion_counts', confusion_rows), ('prediction_changes', pairs)]:
        write_union_csv(REPORTS_DIR / f'{name}.csv', data)
    shutil.copy2(RUN_DIR / 'lmmd_class_metrics.csv', REPORTS_DIR / 'lmmd_class_metrics.csv')
    shutil.copy2(RUN_DIR / 'class_metrics.csv', REPORTS_DIR / 'coral_class_metrics.csv')
    snapshot = REPORTS_DIR / 'data' / 'lmmd'
    snapshot.mkdir(parents=True, exist_ok=True)
    for filename in ('config.json', 'environment.json', 'summary.json', 'metrics.csv', 'step_metrics.csv',
                     'class_metrics.csv', 'lmmd_class_metrics.csv', 'train.log', 'final_epoch.json'):
        shutil.copy2(RUN_DIR / filename, snapshot / filename)
    (snapshot / 'final_epoch').mkdir(exist_ok=True)
    for filename in ('evaluation.npz', 'metrics.csv', 'source_predictions.csv', 'target_predictions.csv'):
        shutil.copy2(RUN_DIR / 'final_epoch' / filename, snapshot / 'final_epoch' / filename)
    write_json(REPORTS_DIR / 'verification.json', verification)
    write_json(REPORTS_DIR / 'provenance.json', dict(primary_metric='epoch_50_target_accuracy', new_training_runs=['lmmd'],
               reused_training_runs=list(CONTROL_SETTINGS), target_training_labels_used=False,
               source_runs=[dict(setting=setting, directory=(RUNS_DIR / setting).relative_to(ROOT).as_posix(),
                                 config=runs[setting][0], summary=runs[setting][2], new_training=setting == 'lmmd') for setting in SETTINGS]))
    figures(runs, summaries, diagnostics, per_class, confusions['lmmd'], predictions['lmmd']['classes'].tolist())
    (REPORTS_DIR / 'report.md').write_text(report_text(runs, summaries, diagnostics, pairs, verification), encoding='utf-8')
    return REPORTS_DIR / 'report.md'


def main():
    print(f'Validated LMMD report: {plot_experiments()}', flush=True)


if __name__ == '__main__':
    main()
