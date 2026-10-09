"""Validate the five runs and produce offline fc8 diagnostics and PNG reports."""

import json
import math
import os
import shutil

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.paths import ROOT
from src.core.coral import compute_covariance
from src.core.subdomain_coral import class_conditional_coral
from src.experiments.subdomain.protocol import CLASS_COUNT, REPORTS_DIR, RUNS_DIR, SETTINGS
from src.experiments.subdomain.verify import mathematical_checks, read_csv, verify_suite
from src.training.evaluation import write_csv
from src.training.train import write_json

LABELS = {'no_adaptation': 'CE only', 'global': 'Global 0.5', 'global_quarter': 'Global 0.25',
          'local': 'Local 0.5', 'mixed': 'Global + local'}
COLORS = {'no_adaptation': '#64748b', 'global': '#2563eb', 'global_quarter': '#0891b2',
          'local': '#c2410c', 'mixed': '#7c3aed'}
STYLES = {'no_adaptation': '--', 'global': '-', 'global_quarter': ':', 'local': '-.', 'mixed': (0, (5, 2, 1, 2))}


def effective_rank(covariance):
    values = torch.linalg.eigvalsh(covariance).clamp_min(0)
    if values.sum() == 0:
        return 0.0
    probabilities = values / values.sum()
    return (-(probabilities * probabilities.clamp_min(1e-30).log()).sum()).exp().item()


@torch.no_grad()
def frozen_diagnostics(setting):
    """Target truth is used only here, on stored final evaluation outputs."""
    with np.load(RUNS_DIR / setting / 'final_epoch' / 'evaluation.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    source = torch.from_numpy(arrays['source_logits']).double()
    target = torch.from_numpy(arrays['target_logits']).double()
    source_labels = torch.from_numpy(arrays['source_labels'])
    target_labels = torch.from_numpy(arrays['target_labels'])
    source_cov = compute_covariance(source)
    target_cov = compute_covariance(target)
    probabilities = target.softmax(1)
    pseudo_loss, pseudo_records = class_conditional_coral(source, target, source_labels, probabilities)
    predictions = target.argmax(1)
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(confusion, (arrays['target_labels'], predictions.numpy()), 1)
    per_class = []
    for index, name in enumerate(arrays['classes'].tolist()):
        source_class = source[source_labels == index]
        target_class = target[target_labels == index]
        source_class_cov = compute_covariance(source_class)
        target_class_cov = compute_covariance(target_class)
        pseudo = pseudo_records[index]
        per_class.append(dict(setting=setting, class_index=index, class_name=name,
                              source_samples=len(source_class), target_samples=len(target_class),
                              source_accuracy=(source_class.argmax(1) == index).double().mean().item(),
                              target_correct=int(confusion[index, index]),
                              target_accuracy=float(confusion[index, index] / confusion[index].sum()),
                              true_covariance_distance=(source_class_cov - target_class_cov).square().sum().item()
                              / (4 * CLASS_COUNT ** 2),
                              centroid_mse=(source_class.mean(0) - target_class.mean(0)).square().mean().item(),
                              source_covariance_trace=source_class_cov.trace().item(),
                              target_covariance_trace=target_class_cov.trace().item(),
                              source_effective_rank=effective_rank(source_class_cov),
                              target_effective_rank=effective_rank(target_class_cov),
                              pseudo_valid=pseudo['valid'], pseudo_covariance_distance=pseudo['local_loss'],
                              pseudo_target_top1_count=pseudo['target_top1_count'],
                              pseudo_target_mass=pseudo['target_mass'], pseudo_target_ess=pseudo['target_ess']))
    result = dict(setting=setting,
                  global_covariance_distance=(source_cov - target_cov).square().sum().item() / (4 * CLASS_COUNT ** 2),
                  true_local_covariance_distance=float(np.mean([row['true_covariance_distance'] for row in per_class])),
                  pseudo_local_covariance_distance=pseudo_loss.item(),
                  pseudo_valid_classes=sum(record['valid'] for record in pseudo_records),
                  centroid_mse=float(np.mean([row['centroid_mse'] for row in per_class])),
                  source_mean_logit_norm=source.norm(dim=1).mean().item(),
                  target_mean_logit_norm=target.norm(dim=1).mean().item(),
                  source_covariance_trace=source_cov.trace().item(), target_covariance_trace=target_cov.trace().item(),
                  source_effective_rank=effective_rank(source_cov), target_effective_rank=effective_rank(target_cov),
                  target_confidence=probabilities.max(1).values.mean().item(),
                  target_entropy=(-(probabilities * probabilities.clamp_min(1e-30).log()).sum(1)).mean().item(),
                  target_predicted_classes=int(np.count_nonzero(confusion.sum(0))))
    if not all(math.isfinite(value) for key, value in result.items() if key != 'setting'):
        raise FloatingPointError(f'Non-finite frozen diagnostic: {setting}')
    return result, per_class, confusion, arrays


def finish_figure(fig, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def temporal_axis(axis, title, ylabel):
    axis.set(title=title, xlabel='Epoch', ylabel=ylabel, xlim=(1, 50))
    axis.set_xticks([1, 10, 20, 30, 40, 50])
    axis.axvline(10.5, color='#475569', linestyle='--', linewidth=0.9, alpha=0.6)
    axis.grid(alpha=0.18)
    axis.spines[['top', 'right']].set_visible(False)


def time_series_figure(runs, fields, destination, shape=(1, 2)):
    fig, axes = plt.subplots(*shape, figsize=(7 * shape[1], 4.1 * shape[0]), layout='constrained', squeeze=False)
    for axis, (field, title, ylabel, factor) in zip(axes.flat, fields):
        for setting in SETTINGS:
            rows = runs[setting][1]
            axis.plot([row['epoch'] for row in rows], [row[field] * factor for row in rows],
                      color=COLORS[setting], linestyle=STYLES[setting], linewidth=1.7, label=LABELS[setting])
        temporal_axis(axis, title, ylabel)
        if 'accuracy' in field:
            axis.set_ylim(0, 100)
        elif field in ('valid_classes', 'target_prediction_classes'):
            axis.set_ylim(0, CLASS_COUNT)
        elif field == 'target_confidence':
            axis.set_ylim(0, 1)
        elif field == 'target_entropy':
            axis.set_ylim(0, math.log(CLASS_COUNT))
        elif field in ('global_loss', 'local_loss'):
            axis.set_yscale('symlog', linthresh=1e-3)
            axis.set_ylim(bottom=0)
            axis.set_ylabel('Distance (symlog; linear < 0.001)')
    axes.flat[0].legend(fontsize=9, ncol=2)
    fig.suptitle('AlexNet fc8 | Amazon to Webcam | seed 42 | dashed line: warm-up ends', fontsize=12)
    finish_figure(fig, destination)


def confusion_figure(setting, matrix, classes, destination):
    normalized = 100 * matrix / matrix.sum(1, keepdims=True)
    fig, axis = plt.subplots(figsize=(12, 10.5), layout='constrained')
    image = axis.imshow(normalized, cmap='Blues', vmin=0, vmax=100, interpolation='nearest')
    axis.set_xticks(range(CLASS_COUNT), [name.replace('_', ' ') for name in classes], rotation=60, ha='right', fontsize=8)
    axis.set_yticks(range(CLASS_COUNT), [name.replace('_', ' ') for name in classes], fontsize=8)
    axis.set(title=f'{LABELS[setting]} | epoch 50 target confusion | 795 images',
             xlabel='Predicted class', ylabel='True class')
    fig.colorbar(image, ax=axis, label='Share within true class (%)', shrink=0.8)
    finish_figure(fig, destination)


def per_run_figures(setting, rows, confusion, classes):
    destination = RUNS_DIR / setting / 'plots'
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    epochs = [row['epoch'] for row in rows]
    for field, label, color, style in [('source_accuracy', 'Source (training images)', '#64748b', '--'),
                                       ('target_accuracy', 'Target', COLORS[setting], '-')]:
        axes[0, 0].plot(epochs, [row[field] * 100 for row in rows], label=label, color=color, linestyle=style)
    axes[0, 0].set_ylim(0, 100)
    for field, label, color, style in [('classification_loss', 'CE', '#64748b', '-'),
                                      ('weighted_global_loss', 'Weighted global', '#2563eb', '--'),
                                      ('weighted_local_loss', 'Weighted local', '#c2410c', '-.')]:
        axes[0, 1].plot(epochs, [row[field] for row in rows], label=label, color=color, linestyle=style)
    axes[1, 0].plot(epochs, [row['valid_classes'] for row in rows], color=COLORS[setting])
    axes[1, 0].set_ylim(0, 31)
    axes[1, 1].plot(epochs, [row['target_confidence'] for row in rows], label='Confidence', color=COLORS[setting])
    axes[1, 1].plot(epochs, [row['target_entropy'] / math.log(31) for row in rows], label='Entropy / log(31)',
                    linestyle='--', color='#64748b')
    axes[1, 1].set_ylim(0, 1)
    for axis, title, ylabel in zip(axes.flat, ['Accuracy', 'Optimized losses', 'Monitored valid classes', 'Target predictions'],
                                   ['Accuracy (%)', 'Loss', 'Classes', 'Probability / normalized entropy']):
        temporal_axis(axis, title, ylabel)
    for axis in [axes[0, 0], axes[0, 1], axes[1, 1]]:
        axis.legend(fontsize=9)
    fig.suptitle(f'{LABELS[setting]} | fc8 | seed 42 | dashed line: warm-up ends')
    finish_figure(fig, destination / 'training.png')
    confusion_figure(setting, confusion, classes, destination / 'confusion.png')


def comparison_figures(runs, summaries, diagnostics, per_class, confusions, classes):
    assets = REPORTS_DIR / 'assets'
    time_series_figure(runs, [('target_accuracy', 'Target accuracy', 'Accuracy (%)', 100),
                              ('target_macro_accuracy', 'Target macro class accuracy', 'Accuracy (%)', 100)],
                       assets / 'accuracy.png')
    time_series_figure(runs, [('classification_loss', 'Source classification', 'CE loss', 1),
                              ('global_loss', 'Global fc8 CORAL (raw)', 'Covariance distance', 1),
                              ('local_loss', 'Class-conditional fc8 CORAL (raw)', 'Mean valid class distance', 1),
                              ('adaptation_loss', 'Optimized adaptation term', 'Weighted global + local', 1)],
                       assets / 'losses.png', (2, 2))
    time_series_figure(runs, [('valid_classes', 'Monitored valid classes', 'Classes (0–31)', 1),
                              ('target_confidence', 'Target training confidence', 'Mean max probability', 1),
                              ('target_entropy', 'Target training entropy', 'Entropy (nats)', 1),
                              ('target_prediction_classes', 'Predicted class diversity per batch', 'Classes (0–31)', 1)],
                       assets / 'coverage.png', (2, 2))
    fig, axis = plt.subplots(figsize=(10, 5.5), layout='constrained')
    bars = axis.bar(range(len(SETTINGS)), [row['target_accuracy'] * 100 for row in summaries],
                    color=[COLORS[setting] for setting in SETTINGS], edgecolor='#334155', linewidth=0.6)
    axis.bar_label(bars, labels=[f'{row["target_accuracy"]:.2%}\n{row["target_correct"]}/795' for row in summaries], padding=5)
    axis.set_xticks(range(len(SETTINGS)), [LABELS[setting] for setting in SETTINGS])
    axis.set(ylim=(0, 100), ylabel='Target accuracy (%)', title='Epoch 50 | five matched AlexNet runs | seed 42')
    axis.grid(axis='y', alpha=0.18)
    axis.set_axisbelow(True)
    finish_figure(fig, assets / 'final_comparison.png')

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
    fields = [('global_covariance_distance', 'Global covariance distance'),
              ('true_local_covariance_distance', 'True-class covariance distance'),
              ('centroid_mse', 'True-class centroid MSE'),
              ('target_mean_logit_norm', 'Target mean logit norm'),
              ('target_covariance_trace', 'Target covariance trace'),
              ('target_effective_rank', 'Target covariance effective rank')]
    for axis, (field, title) in zip(axes.flat, fields):
        values = [diagnostics[setting][field] for setting in SETTINGS]
        axis.barh(range(len(SETTINGS)), values, color=[COLORS[setting] for setting in SETTINGS])
        axis.set_yticks(range(len(SETTINGS)), [LABELS[setting] for setting in SETTINGS])
        axis.invert_yaxis()
        axis.set(title=title, xlim=(0, max(values) * 1.12 if max(values) else 1))
        axis.grid(axis='x', alpha=0.18)
        axis.set_axisbelow(True)
    fig.suptitle('Frozen epoch-50 model | full-domain evaluation | target truth used only after training')
    finish_figure(fig, assets / 'alignment_diagnostics.png')

    recalls = np.array([[row['target_accuracy'] * 100 for row in per_class[setting]] for setting in SETTINGS]).T
    nonreference = [setting for setting in SETTINGS if setting != 'global']
    deltas = np.column_stack([recalls[:, SETTINGS.index(setting)] - recalls[:, SETTINGS.index('global')]
                             for setting in nonreference])
    fig, axes = plt.subplots(1, 2, figsize=(14, 12), layout='constrained')
    absolute = axes[0].imshow(recalls, vmin=0, vmax=100, cmap='Blues', aspect='auto')
    limit = max(1, float(np.abs(deltas).max()))
    relative = axes[1].imshow(deltas, vmin=-limit, vmax=limit, cmap='PuOr', aspect='auto')
    for axis in axes:
        axis.set_yticks(range(CLASS_COUNT), [name.replace('_', ' ') for name in classes], fontsize=9)
    axes[0].set_xticks(range(len(SETTINGS)), [LABELS[setting].replace(' ', '\n', 1) for setting in SETTINGS])
    axes[1].set_xticks(range(len(nonreference)), [LABELS[setting].replace(' ', '\n', 1) for setting in nonreference])
    axes[0].set_title('Epoch-50 target class accuracy (%)')
    axes[1].set_title('Change vs Global 0.5 (percentage points)')
    fig.colorbar(absolute, ax=axes[0], shrink=0.7)
    fig.colorbar(relative, ax=axes[1], shrink=0.7)
    finish_figure(fig, assets / 'class_accuracy.png')
    for setting in SETTINGS:
        per_run_figures(setting, runs[setting][1], confusions[setting], classes)
        shutil.copy2(RUNS_DIR / setting / 'plots' / 'confusion.png', assets / f'confusion_{setting}.png')


def report_text(summaries, diagnostics, pairs, verification):
    table = '\n'.join(f'| {row["setting"]} | {row["target_accuracy"]:.2%} ({row["target_correct"]}/795) | '
                      f'{row["target_macro_accuracy"]:.2%} | {row["gain_vs_global_pp"]:+.2f} | '
                      f'{row["best_target_accuracy"]:.2%} / {row["best_target_epoch"]} |' for row in summaries)
    pair_table = '\n'.join(f'| {row["variant"]} / {row["baseline"]} | {row["delta_pp"]:+.2f} | '
                           f'{row["corrected"]} | {row["regressed"]} | {row["net_correct"]:+d} |' for row in pairs)
    diagnostic_table = '\n'.join(f'| {setting} | {diagnostics[setting]["global_covariance_distance"]:.6g} | '
                                f'{diagnostics[setting]["true_local_covariance_distance"]:.6g} | '
                                f'{diagnostics[setting]["centroid_mse"]:.6g} | '
                                f'{diagnostics[setting]["target_covariance_trace"]:.6g} | '
                                f'{diagnostics[setting]["target_effective_rank"]:.3f} |' for setting in SETTINGS)
    support_table = '\n'.join(f'| {row["setting"]} | {row["post_warmup_valid_classes"]:.2f}/31 | '
                             f'{row["post_warmup_zero_class_batches"]} | {row["final_training_confidence"]:.3f} | '
                             f'{row["final_training_entropy"]:.3f} | {row["elapsed_seconds"]/60:.2f} | '
                             f'{row["peak_gpu_memory_bytes"]/2**30:.2f} |' for row in summaries)
    loss_table = '\n'.join(f'| {row["setting"]} | {row["final_classification_loss"]:.5f} | '
                          f'{row["final_global_loss"]:.5f} | {row["final_local_loss"]:.5f} | '
                          f'{row["final_weighted_global_loss"]:.5f} | {row["final_weighted_local_loss"]:.5f} |'
                          for row in summaries)
    by_setting = {row['setting']: row for row in summaries}
    replacement, addition, redistribution = pairs
    diagnostic_pairs = '\n'.join(
        f'- {row["variant"]}/{row["baseline"]}：真实类协方差距离 '
        f'{diagnostics[row["baseline"]]["true_local_covariance_distance"]:.6g} → '
        f'{diagnostics[row["variant"]]["true_local_covariance_distance"]:.6g}；类别中心 MSE '
        f'{diagnostics[row["baseline"]]["centroid_mse"]:.6g} → {diagnostics[row["variant"]]["centroid_mse"]:.6g}；'
        f'目标准确率变化 {row["delta_pp"]:+.2f} 个百分点。' for row in pairs)
    coverage_pairs = '\n'.join(
        f'- {row["variant"]}/{row["baseline"]}：预热后每批平均有效类 '
        f'{by_setting[row["baseline"]]["post_warmup_valid_classes"]:.2f} → '
        f'{by_setting[row["variant"]]["post_warmup_valid_classes"]:.2f}；目标 logits 平均范数 '
        f'{diagnostics[row["baseline"]]["target_mean_logit_norm"]:.3f} → '
        f'{diagnostics[row["variant"]]["target_mean_logit_norm"]:.3f}；目标协方差有效秩 '
        f'{diagnostics[row["baseline"]]["target_effective_rank"]:.3f} → '
        f'{diagnostics[row["variant"]]["target_effective_rank"]:.3f}。' for row in pairs[:2])
    interpretation = []
    if replacement['delta_pp'] < 0 and addition['delta_pp'] < 0 and redistribution['delta_pp'] < 0:
        interpretation.append('本轮三个预先指定比较均为负结果，未观察到类条件约束改善全局 CORAL 的分类效果。'
                              '这一结论限定于本协议、seed=42 和 Amazon→Webcam。')
    if (replacement['delta_pp'] < 0 and diagnostics['local']['true_local_covariance_distance']
            < diagnostics['global']['true_local_covariance_distance']):
        interpretation.append('local 的真实类协方差距离更小，但分类准确率下降，说明距离减小没有转化为分类收益。')
    interpretation.append(
        f'local 的目标 logits 平均范数为 {diagnostics["local"]["target_mean_logit_norm"]:.3f}，'
        f'全局组为 {diagnostics["global"]["target_mean_logit_norm"]:.3f}；目标协方差迹分别为 '
        f'{diagnostics["local"]["target_covariance_trace"]:.3f} 和 '
        f'{diagnostics["global"]["target_covariance_trace"]:.3f}，有效秩分别为 '
        f'{diagnostics["local"]["target_effective_rank"]:.3f} 和 '
        f'{diagnostics["global"]["target_effective_rank"]:.3f}。解释协方差距离时需要同时考虑这些幅度与谱变化。'
        '上述数值不足以单独确定伪标签误差、正则强度或特征退化各自的因果贡献。')
    interpretation = '\n\n'.join(interpretation)
    followup = ('\n\n追加的 [LMMD 对照与六组比较](lmmd/report.md)复用本报告五组，仅新增一组 LMMD 训练。'
                if (REPORTS_DIR / 'lmmd' / 'report.md').exists() else '')
    return f'''# AlexNet fc8 类条件 CORAL 五组实验

日期：2026-10-09。Office-31 Amazon→Webcam，seed=42，全部五组重新训练 50 轮。主结果为固定第 50 轮，最佳轮仅描述。{followup}

类条件替换相对全局 CORAL 的变化为 **{replacement['delta_pp']:+.2f} 个百分点**；联合组相对较弱全局对照为 **{addition['delta_pp']:+.2f} 个百分点**，相对全局 0.5 为 **{redistribution['delta_pp']:+.2f} 个百分点**。

## 设计与主要结果

[固定协议](protocol.md)采用 ImageNet-1K V1 AlexNet、batch=128、workers=0、每轮六步；四适应组前十轮统一 0.5 全局 CORAL，第十一轮切换。`global_quarter` 和 `local` 的名称指预热后的损失。全局/类条件联合的系数各为 0.25。

方法借鉴 [DSAN](https://jd92.wang/assets/files/a24tnnls20.pdf) 的类别软权重，其原方法使用 LMMD；本轮为自行定义的类条件协方差 CORAL。目标权重来自同一次训练前向的 softmax，停止梯度。目标训练标签在训练循环边界丢弃。

| 设置 | 第 50 轮目标准确率 | 宏平均类别准确率 | 相对 global（百分点） | 最佳准确率 / epoch，仅描述 |
|---|---|---|---|---|
{table}

![第50轮比较](assets/final_comparison.png)

![准确率曲线](assets/accuracy.png)

源域评估使用源训练图片，不是独立测试集。五组使用相同原始数据、初始化、采样及评估协议；旧实验只保留为历史资料，不混入本表。

## 三个预先指定的比较

| 变体 / 对照 | 准确率变化（百分点） | 对照错误→变体正确 | 对照正确→变体错误 | 净新增正确 |
|---|---|---|---|---|
{pair_table}

local/global 比较预热后替换对齐方式；mixed/global_quarter 比较固定全局系数下增加局部约束；mixed/global 比较固定名义总系数下分配约束。原始损失尺度不同，同系数不代表同梯度强度。paired change 数来自相同 795 张目标图像的最终预测，不作跨种子显著性推断。

![逐类别准确率和增益](assets/class_accuracy.png)

## 损失与类别覆盖

![原始和加权损失](assets/losses.png)

以下为第 50 轮六个训练批次的均值，保留未优化项的原始监测值；与冻结模型的全域距离分开报告。

| 设置 | CE | G 原始值 | L 原始值 | 加权 G | 加权 L |
|---|---|---|---|---|---|
{loss_table}

所有设置均监测 G 和 L，即使该项没有参与优化。L 按有效类平均；有效类要求源计数 >=2、目标 top-1 计数 >=2、目标质量 >1e-12、有效样本量 >=2。全部目标图像仍以软概率参与有效类协方差。无效类距离在 CSV 留空；无有效类批次返回连接到特征的零值。

| 设置 | 预热后每批平均有效类 | 预热后零有效类批次 | 最后轮训练置信度 | 最后轮训练熵（nats） | 训练耗时（分钟） | 峰值显存（GiB） |
|---|---|---|---|---|---|---|
{support_table}

![类别覆盖和伪标签诊断](assets/coverage.png)

类别支持筛选会改变参与平均的类别集合，不能仅由 L 的下降推断类条件对齐改善。逐批权重质量、有效样本量、top-1 计数和跳过原因见 [class_metrics.csv](class_metrics.csv)。

## 冻结模型的事后诊断

下表在训练完成后使用第 50 轮全域 logits 计算；真实类诊断才使用目标真实标签，训练与调度不读取这些标签。距离均在相同的 fc8、31 维定义下比较。全域评估关闭 Dropout，与训练批次损失分开解释。

| 设置 | 全局协方差距离 | 真实类平均协方差距离 | 真实类中心 MSE | 目标协方差迹 | 目标有效秩 |
|---|---|---|---|---|---|
{diagnostic_table}

{diagnostic_pairs}

{coverage_pairs}

![最终特征诊断](assets/alignment_diagnostics.png)

完整的两域 logits 范数、协方差迹、有效秩、软预测类距离和置信度见 [diagnostics.csv](diagnostics.csv)；各类别中心、协方差及准确率见 [class_diagnostics.csv](class_diagnostics.csv)。有效秩定义为归一化非负协方差特征值的熵指数；全零协方差记为零。中心 MSE 不参与训练损失。

混淆矩阵：[无适应](assets/confusion_no_adaptation.png)、[全局 0.5](assets/confusion_global.png)、[较弱全局](assets/confusion_global_quarter.png)、[类条件](assets/confusion_local.png)、[联合](assets/confusion_mixed.png)。矩阵按真实类别归一化；[confusion_counts.csv](confusion_counts.csv) 保留全部原始计数。

## 结果解释

{interpretation}

## 验证与复现

- 数学检查通过：单类别退化为原 CORAL、类别缺失/单样本/无有效类别、有限特征梯度、软概率停止梯度，以及其他 top-1 类图像仍贡献软权重。
- 正式结果核验：{verification['runs']} 组、{verification['epochs']} 轮、{verification['steps']} 步、{verification['class_rows']} 条类别记录；四适应组前十轮逐批/逐轮/逐类指标在 1e-6 容差内一致，第十一轮系数正确切换。
- CSV、逐轮日志、最终 NPZ、逐图预测与汇总互相一致；源/目标每次评估分别覆盖 2,817/795 张。固定最终轮的选择不使用目标准确率。

```powershell
uv run --no-sync python -m src.experiments.subdomain.run
uv run --no-sync python -m src.experiments.subdomain.run --resume
uv run --no-sync python -m src.experiments.subdomain.verify
uv run --no-sync python -m src.plotting.plot_subdomain
```

[summary.csv](summary.csv)、[epoch_metrics.csv](epoch_metrics.csv)、[step_metrics.csv](step_metrics.csv)、[prediction_changes.csv](prediction_changes.csv)、[verification.json](verification.json)、[provenance.json](provenance.json) 提供可检查的数字及来源；`data/<setting>/` 保存本报告使用的原始配置、日志、CSV 和最终 logits/预测快照，完整训练产物位于 `results/subdomain_coral/`。

本轮为单 seed、单方向探索。协方差接近并不保证类别中心正确或分类改善，特征尺度变化也会影响距离；应结合准确率、类别覆盖、中心和谱诊断解释。每类批内样本有限，伪标签误差、类别支持筛选和损失尺度变化的贡献未分别隔离。本轮不以目标标签调节阈值或选取模型，不将单次增益作为统计显著性结论。
'''


def plot_experiments():
    torch.set_num_threads(1)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titleweight': 'bold'})
    verification, runs = verify_suite(return_runs=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics, per_class, confusions, predictions = {}, {}, {}, {}
    summaries, epoch_rows, step_rows, class_rows, confusion_rows = [], [], [], [], []
    global_accuracy = runs['global'][1][-1]['target_accuracy']
    for setting, (config, rows, summary) in runs.items():
        diagnostics[setting], per_class[setting], confusions[setting], predictions[setting] = frozen_diagnostics(setting)
        final = rows[-1]
        summaries.append(dict(setting=setting, epoch=50, target_accuracy=final['target_accuracy'],
                              target_correct=int(final['target_correct']), target_samples=795,
                              target_macro_accuracy=final['target_macro_accuracy'], source_accuracy=final['source_accuracy'],
                              gain_vs_global_pp=100 * (final['target_accuracy'] - global_accuracy),
                              best_target_accuracy=summary['best_target_accuracy'], best_target_epoch=summary['best_target_epoch'],
                              post_warmup_valid_classes=float(np.mean([row['valid_classes'] for row in rows[10:]])),
                              post_warmup_zero_class_batches=int(round(sum(row['no_valid_classes'] * 6 for row in rows[10:]))),
                              final_training_confidence=final['target_confidence'], final_training_entropy=final['target_entropy'],
                              final_classification_loss=final['classification_loss'],
                              final_global_loss=final['global_loss'], final_local_loss=final['local_loss'],
                              final_weighted_global_loss=final['weighted_global_loss'],
                              final_weighted_local_loss=final['weighted_local_loss'],
                              elapsed_seconds=summary['elapsed_seconds'], peak_gpu_memory_bytes=summary['peak_gpu_memory_bytes']))
        epoch_rows.extend(dict(setting=setting, **row) for row in rows)
        step_rows.extend(dict(setting=setting, **row) for row in read_csv(RUNS_DIR / setting / 'step_metrics.csv'))
        class_rows.extend(dict(setting=setting, **row) for row in read_csv(RUNS_DIR / setting / 'class_metrics.csv'))
        confusion_rows.extend(dict(setting=setting, true_class=truth, predicted_class=predicted,
                                   count=int(confusions[setting][truth, predicted]))
                              for truth in range(CLASS_COUNT) for predicted in range(CLASS_COUNT))
        snapshot = REPORTS_DIR / 'data' / setting
        snapshot.mkdir(parents=True, exist_ok=True)
        for filename in ('config.json', 'environment.json', 'summary.json', 'metrics.csv', 'step_metrics.csv',
                         'class_metrics.csv', 'train.log', 'final_epoch.json'):
            shutil.copy2(RUNS_DIR / setting / filename, snapshot / filename)
        final_snapshot = snapshot / 'final_epoch'
        final_snapshot.mkdir(exist_ok=True)
        for filename in ('evaluation.npz', 'metrics.csv', 'source_predictions.csv', 'target_predictions.csv'):
            shutil.copy2(RUNS_DIR / setting / 'final_epoch' / filename, final_snapshot / filename)
    pairs = []
    for variant, baseline in [('local', 'global'), ('mixed', 'global_quarter'), ('mixed', 'global')]:
        labels = predictions[variant]['target_labels']
        if not np.array_equal(labels, predictions[baseline]['target_labels']):
            raise ValueError('Paired prediction labels differ')
        variant_correct = predictions[variant]['target_logits'].argmax(1) == labels
        baseline_correct = predictions[baseline]['target_logits'].argmax(1) == labels
        corrected = int(np.sum(variant_correct & ~baseline_correct))
        regressed = int(np.sum(~variant_correct & baseline_correct))
        pairs.append(dict(variant=variant, baseline=baseline, corrected=corrected, regressed=regressed,
                          both_correct=int(np.sum(variant_correct & baseline_correct)),
                          both_wrong=int(np.sum(~variant_correct & ~baseline_correct)),
                          net_correct=corrected - regressed, delta_pp=100 * (corrected - regressed) / len(labels)))
    classes = predictions['global']['classes'].tolist()
    for name, data in [('summary', summaries), ('epoch_metrics', epoch_rows), ('step_metrics', step_rows),
                       ('class_metrics', class_rows), ('diagnostics', list(diagnostics.values())),
                       ('class_diagnostics', [row for setting in SETTINGS for row in per_class[setting]]),
                       ('confusion_counts', confusion_rows), ('prediction_changes', pairs)]:
        write_csv(REPORTS_DIR / f'{name}.csv', data)
    write_json(REPORTS_DIR / 'verification.json', dict(mathematics=mathematical_checks(), suite=verification))
    write_json(REPORTS_DIR / 'provenance.json', dict(protocol='reports/subdomain_coral/protocol.md',
               primary_metric='epoch_50_target_accuracy', training_performed=True, all_five_controls_retrained=True,
               target_training_labels_used=False, frozen_true_class_diagnostics='post_training_only',
               source_runs=[dict(setting=setting, directory=(RUNS_DIR / setting).relative_to(ROOT).as_posix(),
                                 archived_directory=f'reports/subdomain_coral/data/{setting}',
                                 config=runs[setting][0], summary=runs[setting][2]) for setting in SETTINGS]))
    comparison_figures(runs, summaries, diagnostics, per_class, confusions, classes)
    (REPORTS_DIR / 'report.md').write_text(report_text(summaries, diagnostics, pairs, verification), encoding='utf-8')
    return REPORTS_DIR / 'report.md'


def main():
    print(f'Validated report: {plot_experiments()}', flush=True)


if __name__ == '__main__':
    main()
