"""Validate completed ResNet runs and compare each with its same-epoch control."""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.common.experiment_protocol import RESNET_LAYERS, SUITES
from src.common.paths import ROOT, suite_results_dir
from src.plotting.plot_accuracies_losses import environment_differences, read_run, save_plot
from src.training.evaluation import write_csv

SETTINGS = ('no_coral', *('_'.join(layers) for layers in SUITES['resnet50']['settings']))


def read_resnet_run(directory):
    config, summary, data = read_run(directory)
    if config['schema_version'] != 6 or directory.name != config['setting']:
        raise ValueError(f'Expected a ResNet suite archive: {directory}')
    count = len(data['epoch'])
    required = dict(training_complete=True, max_epochs=config['epochs'], stop_epoch=count,
                    stopped_early=count < config['epochs'], steps_completed=24 * count,
                    selected_epoch=count, selected_target_accuracy=float(data['target_accuracy'][-1]),
                    selection_metric='training_stop', original_controls_retrained=False,
                    batchnorm_training_verified=True, batchnorm_evaluation_verified=True,
                    optimizer_coverage_verified=True, feature_gradient_verified=True)
    if any(summary.get(key) != value for key, value in required.items()):
        raise ValueError(f'Incomplete ResNet summary: {directory}')
    if not np.all(data['bn_updates_per_module'] == 48):
        raise ValueError(f'Incorrect BatchNorm updates: {directory}')
    if config['lambda_coral'] == 0 and summary['stop_reason'] != 'max_epochs':
        raise ValueError('The no-CORAL baseline must run to the maximum budget')
    for key in ('elapsed_seconds', 'peak_gpu_memory_bytes'):
        if not math.isfinite(summary[key]) or summary[key] < 0:
            raise ValueError(f'Invalid cost metric: {directory}: {key}')
    with (directory / 'train.log').open(encoding='utf-8') as stream:
        logged = [json.loads(line) for line in stream if line.strip()]
    if len(logged) != count or any(set(row) != set(data) or any(row[key] != data[key][i]
            for key in data) for i, row in enumerate(logged)):
        raise ValueError(f'train.log differs from CSV: {directory}')
    for name, epoch in (('best_epoch', summary['best_target_epoch']), ('stop_epoch', count)):
        metadata = json.loads((directory / f'{name}.json').read_text(encoding='utf-8'))
        row = {key: float(values[epoch - 1]) for key, values in data.items()}
        if metadata['epoch'] != epoch or metadata['metrics'] != row:
            raise ValueError(f'Incorrect {name} metadata: {directory}')
        with (directory / name / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
            recorded_rows = [{key: float(value) for key, value in item.items()} for item in csv.DictReader(stream)]
        if recorded_rows != [row]:
            raise ValueError(f'Incorrect {name} metric CSV: {directory}')
        with np.load(directory / name / 'evaluation.npz', allow_pickle=False) as arrays:
            classes = [label for label, _ in sorted(config['class_to_idx'].items(), key=lambda item: item[1])]
            if arrays['classes'].tolist() != classes:
                raise ValueError(f'Incorrect class mapping: {directory}')
            for domain in ('source', 'target'):
                logits, labels = arrays[f'{domain}_logits'], arrays[f'{domain}_labels']
                size = config[f'{domain}_samples']
                if logits.shape != (size, 31) or labels.shape != (size,) or not np.isfinite(logits).all():
                    raise ValueError(f'Invalid prediction arrays: {directory}: {name}/{domain}')
                predictions = logits.argmax(axis=1)
                with (directory / name / f'{domain}_predictions.csv').open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                if (len(rows) != size or int((predictions == labels).sum()) != row[f'{domain}_correct']
                        or any(int(item['sample_index']) != i or int(item['label_index']) != labels[i]
                               or int(item['prediction_index']) != predictions[i]
                               or int(item['correct']) != int(predictions[i] == labels[i])
                               for i, item in enumerate(rows))):
                    raise ValueError(f'Predictions differ from metrics/NPZ: {directory}: {name}/{domain}')
    return config, summary, data


def plot_reference_figure(directory, runs, setting):
    """Four panels matching plots.png, using a common observed epoch window."""
    _, summary, data = runs[setting]
    _, _, baseline = runs['no_coral']
    stop = summary['stop_epoch']
    control = {key: values[:stop] for key, values in baseline.items()}
    with plt.rc_context({'font.size': 12, 'axes.titlesize': 14, 'legend.fontsize': 11,
                         'lines.linewidth': 1.6, 'lines.markersize': 5}):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'ResNet-50 V2: {setting} vs no_coral', fontsize=19, y=0.98)
        ax = axes[0, 0]
        for series, key, label, style in (
            (data, 'target_accuracy', 'target acc. w/ CORAL', 'b-*'),
            (control, 'target_accuracy', 'target acc. w/o CORAL', 'g-o'),
            (data, 'source_accuracy', 'source acc. w/ CORAL', 'r-^'),
            (control, 'source_accuracy', 'source acc. w/o CORAL', 'm-+'),
        ):
            ax.plot(series['epoch'], 100 * series[key], style, label=label)
        ax.set(title='Source and target accuracy', ylabel='Accuracy (%)', ylim=(0, 105))
        ax.legend(loc='lower right')

        ax = axes[0, 1]
        ax.plot(data['epoch'], data['classification_loss'], 'r-o', label='classification loss')
        ax.plot(data['epoch'], data['weighted_CORAL_loss'], 'b-*', label='weighted CORAL loss')
        ax.set(title=f'Training losses: {setting}', ylabel='Loss (log scale)', yscale='log')
        ax.legend(loc='upper right')

        ax = axes[1, 0]
        ax.plot(control['epoch'], control['CORAL_logits'], 'b-o', label='logits distance w/o CORAL')
        ax.set(title='Logits CORAL distance in the control', ylabel='Unweighted CORAL distance')
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left')

        ax = axes[1, 1]
        ax.plot(data['epoch'], data['classification_loss'], 'r-o', label='classification loss w/ CORAL')
        ax.plot(control['epoch'], control['classification_loss'], 'b-*', label='classification loss w/o CORAL')
        ax.set(title='Classification loss comparison', ylabel='Loss (log scale)', yscale='log')
        ax.legend(loc='upper right')

        ticks = sorted({1, stop, *range(10, stop, 10)})
        for letter, ax in zip('abcd', axes.flat):
            ax.set(xlim=(0.5, stop + 0.5), xticks=ticks, xlabel=f'Epoch\n({letter})')
        fig.text(0.5, 0.025,
                 f'Amazon (2,817) -> Webcam (795) | seed=42 | shared BatchNorm | epochs 1-{stop}\n'
                 'Control truncated to the same epoch; (b) uses the actual weighted CORAL term; '
                 '(c) shows logits only. No smoothing or synthetic epoch 0.',
                 ha='center', fontsize=11)
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.115, top=0.92,
                            wspace=0.22, hspace=0.28)
        fig.savefig(directory / 'plots.png', dpi=180)
        plt.close(fig)


def plot_experiments(root):
    directories = {setting: root / setting for setting in SETTINGS}
    runs = {setting: read_resnet_run(directory) for setting, directory in directories.items()}
    base_config, _, baseline = runs['no_coral']
    reference_env = json.loads((directories['no_coral'] / 'environment.json').read_text(encoding='utf-8'))
    varying = {'setting', 'coral_layers', 'lambda_coral', 'effective_coral_weights', 'stop_on_loss_balance'}
    for setting, (config, _, _) in runs.items():
        differences = [key for key in config.keys() | base_config.keys()
                       if key not in varying and config.get(key) != base_config.get(key)]
        environment = json.loads((directories[setting] / 'environment.json').read_text(encoding='utf-8'))
        if differences or environment_differences(environment, reference_env):
            raise ValueError(f'Incompatible ResNet comparison: {setting}: {differences}')
    comparison = root / 'comparison'
    comparison.mkdir(parents=True, exist_ok=True)
    summaries, epoch_rows = [], []
    for setting, (_, summary, data) in runs.items():
        stop = summary['stop_epoch']
        matched = float(baseline['target_accuracy'][stop - 1])
        summaries.append(dict(setting=setting, stop_epoch=stop, stop_reason=summary['stop_reason'],
                              target_accuracy=summary['final_target_accuracy'],
                              correct=int(data['target_correct'][-1]), baseline_same_epoch_accuracy=matched,
                              improvement_pp=100 * (summary['final_target_accuracy'] - matched),
                              best_target_accuracy=summary['best_target_accuracy'],
                              best_target_epoch=summary['best_target_epoch'],
                              train_steps=summary['steps_completed'], elapsed_seconds=summary['elapsed_seconds'],
                              peak_gpu_memory_bytes=summary['peak_gpu_memory_bytes']))
        for index, epoch in enumerate(data['epoch']):
            epoch_rows.append(dict(setting=setting, **{key: float(value[index]) for key, value in data.items()}))
        plots = directories[setting] / 'plots'
        accuracy = [(data['epoch'], data[f'{domain}_accuracy'], domain, {}) for domain in ('source', 'target')]
        if setting != 'no_coral':
            accuracy.append((baseline['epoch'], baseline['target_accuracy'], 'no CORAL: target', {'linestyle': '--'}))
        save_plot(plots, 'accuracy', f'ResNet-50 V2: {setting}', 'Accuracy', accuracy)
        save_plot(plots, 'losses', setting, 'Loss', [(data['epoch'], data[key], label, {}) for key, label in
                  (('classification_loss', 'CE'), ('weighted_CORAL_loss', 'weighted CORAL'), ('total_loss', 'total'))])
        for layer in RESNET_LAYERS:
            save_plot(plots, f'coral_{layer}', f'{setting}: {layer}', 'CORAL distance',
                      [(data['epoch'], data[f'CORAL_{layer}'], setting, {}),
                       (baseline['epoch'], baseline[f'CORAL_{layer}'], 'no CORAL', {'linestyle': '--'})])
        if 'balance_gap' in data:
            save_plot(plots, 'stopping_signal', f'{setting}: stopping signal', 'Relative gap',
                      [(data['epoch'], data['balance_gap'], 'EMA gap', {}),
                       (data['epoch'], data['balance_raw_gap'], 'raw gap', {'alpha': 0.5}),
                       (data['epoch'], np.full(len(data['epoch']), 0.25), 'threshold', {'linestyle': '--'})])
    write_csv(comparison / 'summary.csv', summaries)
    # Baseline has no stop columns: preserve them as missing, never zero-fill.
    columns = list(dict.fromkeys(key for row in epoch_rows for key in row))
    with (comparison / 'epoch_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(epoch_rows)
    save_plot(comparison, 'target_accuracy', 'ResNet-50 V2 target accuracy', 'Accuracy',
              [(data['epoch'], data['target_accuracy'], setting, {'linestyle': '--'} if setting == 'no_coral' else {})
               for setting, (_, _, data) in runs.items()])
    for layer in RESNET_LAYERS:
        save_plot(comparison, f'coral_{layer}', f'CORAL distance: {layer}', 'CORAL distance',
                  [(data['epoch'], data[f'CORAL_{layer}'], setting, {}) for setting, (_, _, data) in runs.items()])
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), layout='constrained')
    colors = dict(zip(SETTINGS[1:], ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd')))
    markers = dict(zip(SETTINGS[1:], ('o', 's', '^', 'D', '*')))
    for ax, key, title in zip(axes, ('balance_raw_gap', 'balance_gap'), ('Raw loss gap', 'EMA loss gap')):
        for setting in SETTINGS[1:]:
            data = runs[setting][2]
            ax.plot(data['epoch'], 100 * data[key], label=setting, color=colors[setting],
                    marker=markers[setting], markevery=10, markersize=4)
            ax.plot(data['epoch'][-1], 100 * data[key][-1], marker=markers[setting], color=colors[setting])
        ax.axhline(25, color='gray', linestyle='--', label='25% threshold')
        ax.axvline(20, color='gray', linestyle=':', label='checks start at epoch 20')
        ax.set(title=title, xlabel='Epoch', ylabel='Relative gap (%)', ylim=(0, 105))
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=2, loc='lower right')
    fig.savefig(comparison / 'stopping_signals.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout='constrained')
    positions = np.arange(len(summaries))
    actual = [100 * row['target_accuracy'] for row in summaries]
    matched = [100 * row['baseline_same_epoch_accuracy'] for row in summaries]
    bars = axes[0].bar(positions - 0.2, actual, 0.4, label='setting at stop')
    axes[0].bar_label(bars, labels=[f'{value:.2f}%' for value in actual], padding=2)
    axes[0].bar(positions + 0.2, matched, 0.4, label='no CORAL at same epoch', color='gray')
    axes[0].set(ylabel='Target accuracy (%)', ylim=(0, 105), title='Training-stop accuracy and matched baseline')
    axes[0].legend()
    gains = [row['improvement_pp'] for row in summaries]
    bars = axes[1].bar(positions, gains)
    axes[1].bar_label(bars, labels=[f'{value:+.2f}' for value in gains], padding=3)
    axes[1].axhline(0, color='gray')
    axes[1].set(ylabel='Gain (percentage points)', title='CORAL gain at the same epoch')
    axes[1].set_ylim(min(0, min(gains)) - 0.45, max(0, max(gains)) + 0.55)
    for ax in axes:
        ax.set_xticks(positions, [f'{row["setting"]}\nepoch {row["stop_epoch"]}' for row in summaries])
        ax.grid(axis='y', alpha=0.2)
    fig.savefig(comparison / 'stop_comparison.png', dpi=180)
    plt.close(fig)
    best = max(summaries, key=lambda row: row['target_accuracy'])
    plot_reference_figure(comparison, runs, best['setting'])
    provenance = dict(protocol='reports/resnet50-protocol.md', pretrained='IMAGENET1K_V2',
                      primary_metric='training_stop_target_accuracy', delta_control='no_coral_same_epoch',
                      validation='CSV/log/summary/predictions/early-stop replay and runtime BatchNorm checks',
                      reference_figure=dict(file='plots.png', setting=best['setting'],
                                            first_epoch=1, last_epoch=best['stop_epoch'],
                                            baseline_truncated_to_same_epoch=True,
                                            loss_metric='weighted_CORAL_loss', distance_layer='logits',
                                            selection='highest stop accuracy for descriptive visualization'),
                      historical_alexnet_used_as_matched_control=False, environment=reference_env,
                      source_runs=[dict(setting=s, directory=str(d), config=runs[s][0], summary=runs[s][1])
                                   for s, d in directories.items()])
    runtime_notes = json.loads((root / 'runtime_notes.json').read_text(encoding='utf-8')) if (root / 'runtime_notes.json').exists() else {}
    provenance['runtime_notes'] = runtime_notes
    (comparison / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    table = '\n'.join(f'| {r["setting"]} | {r["stop_epoch"]} | {r["stop_reason"]} | {100*r["target_accuracy"]:.2f}% '
                      f'({r["correct"]}/795) | {100*r["baseline_same_epoch_accuracy"]:.2f}% | {r["improvement_pp"]:+.2f} | '
                      f'{100*r["best_target_accuracy"]:.2f}% / {r["best_target_epoch"]} |' for r in summaries)
    costs = '\n'.join(f'| {r["setting"]} | {r["train_steps"]} | {r["elapsed_seconds"]/60:.2f} | '
                      f'{r["peak_gpu_memory_bytes"]/2**30:.2f} |' for r in summaries)
    gain = max(summaries[1:], key=lambda row: row['improvement_pp'])
    baseline_correct = int(baseline['target_correct'][gain['stop_epoch'] - 1])
    target_samples = base_config['target_samples']
    control_errors = target_samples - baseline_correct
    setting_errors = target_samples - gain['correct']
    extra_correct = gain['correct'] - baseline_correct
    error_reduction = 100 * extra_correct / control_errors if control_errors else 0.0
    balance_table = '\n'.join(
        f'| {setting} | {summary["stop_epoch"]} | {data["classification_loss"][-1]:.6f} | '
        f'{data["weighted_CORAL_loss"][-1]:.6f} | {100*data["balance_raw_gap"][-1]:.2f}% | '
        f'{100*data["balance_gap"][-1]:.2f}% | {int(data["balance_streak"][-1])} |'
        for setting, (_, summary, data) in runs.items() if setting != 'no_coral')
    weights_table = '\n'.join(
        f'| {setting} | {", ".join(config["coral_layers"]) or "无"} | '
        f'{", ".join(f"{layer}: {weight:g}" for layer, weight in config["effective_coral_weights"].items() if weight) or "0"} |'
        for setting, (config, _, _) in runs.items())
    alignment_ratios = {setting: 100 * runs[setting][2]['weighted_CORAL_loss'][-1]
                        / runs[setting][2]['classification_loss'][-1] for setting in ('pooled', 'layer4')}
    protocol_link = (ROOT / 'reports' / 'resnet50-protocol.md').as_posix()
    report = f'''# ResNet-50 V2 + CORAL 六组实验

日期：2026-10-08。任务：Office-31 Amazon→Webcam，31 类，单种子探索性比较。

## 研究问题与实验设计

本实验检验：在 ImageNet-1K V2 预训练 ResNet-50、两域共同更新 BatchNorm 的固定配置下，CORAL 是否提供额外收益，以及收益是否依赖对齐位置。
它没有分离网络结构、预训练方案和 BatchNorm 的各自贡献，历史 AlexNet 结果仅作背景，不能作为匹配的跨架构对照。

使用 ImageNet-1K V2 预训练、seed=42、Amazon→Webcam、batch=32，每轮 24 步。全网络 SGD 微调，骨干 lr=0.001、分类头 lr=0.01，λ=0.5，多层取均值；双层实际各 0.25。
BatchNorm 每步按源域、目标域顺序更新，评估冻结统计量。无 CORAL 对照同样使用目标域更新统计量，因此本表衡量相同共享 BatchNorm 策略下 CORAL 的额外收益，不是纯 source-only 比较。

| 设置 | 对齐对象 | 实际损失系数 |
| --- | --- | --- |
{weights_table}

模型显式使用 `ResNet50_Weights.IMAGENET1K_V2`，替换为 31 类头（权重标准差 0.005、偏置 0.01）。保持原始 JPG、`Resize(224)` 和各域归一化，不增加增强。
SGD momentum=0.9、weight decay=0.0005，学习率与 λ 恒定；workers=0、FP32、严格确定性、关闭 TF32，无激活检查点。目标域标签只用于评估和补充预测归档，不进入训练损失或早停信号。完整设计见 [锁定协议]({protocol_link})。

## 停止轮结果

| 设置 | 停止轮 | 原因 | 目标准确率 | 对照同轮准确率 | 增益（百分点） | 最佳准确率 / 轮次（补充） |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
{table}

停止轮准确率最高为 **{best['setting']}：{100*best['target_accuracy']:.2f}%**。CORAL 组中同轮增益最大为 **{gain['setting']}：{gain['improvement_pp']:+.2f} 个百分点**。这两项回答不同问题，不将停止轮不同的原始准确率差全部归因于 CORAL。
增益定义为该设置停止轮准确率减去 no_coral 在相同 epoch 的准确率，而非统一减去对照第 100 轮成绩。未触发早停的组完整达到 100 轮；最佳轮及其准确率只作补充，不用于训练停止或参数选择。

![停止轮及同轮对照](stop_comparison.png)
![目标准确率曲线](target_accuracy.png)

## 四宫格训练曲线

参考仓库 `plots.png` 的布局，以停止轮准确率最高的 {best['setting']} 作描述性展示，对照截取至同一轮。
(a) 两域全量评估准确率；(b) CE 与实际加权 CORAL 损失；(c) 无 CORAL 对照的 logits 距离；(d) 两组 CE。
损失面板使用对数纵轴以保留初期高损失和后期变化；不平滑、不补第 0 轮或早停后的数据。

![四宫格训练曲线](plots.png)

## 结果解释：迁移表现与 CORAL 的额外收益

**较好的无 CORAL 成绩支持配置有效，尚不能证明是结构本身的跨域泛化优势。** no_coral 第 100 轮达到 {100*summaries[0]['target_accuracy']:.2f}%，说明这套预训练、微调与共享 BN 配置已有较好的目标准确率。
但目标域图像参与了 BN 统计量更新，因此这一成绩包含结构、V2 预训练和目标域统计信息的共同作用。实验没有纯源域 BN 对照，也没有一致预训练、采样和训练协议下的 AlexNet 对照，不能识别各因素的独立贡献。

**CORAL 收益依赖对齐位置，不能笼统归纳为提升不明显。** 单独 pooled、layer4 的同轮增益分别为 {summaries[2]['improvement_pp']:+.2f}、{summaries[4]['improvement_pp']:+.2f} 个百分点，收益有限。
包含 logits 的三组同轮增益为 {summaries[1]['improvement_pp']:+.2f}、{summaries[3]['improvement_pp']:+.2f}、{summaries[5]['improvement_pp']:+.2f} 个百分点，均为正向观察。
以 {gain['setting']} 第 {gain['stop_epoch']} 轮为例，相同 epoch 的对照判对 {baseline_correct}/{target_samples} 张，CORAL 组判对 {gain['correct']}/{target_samples} 张，多判对 {extra_correct} 张；错误数从 {control_errors} 减至 {setting_errors}，相对减少 {error_reduction:.2f}%。这一差异不宜仅因准确率曲线接近而忽略，但其跨种子稳定性尚未验证。

**联合设置的优势不能单独归因于新增 layer4。** logits 单层系数为 0.5，而联合组的 logits 与 layer4 各为 0.25；同时改变了对齐位置和每层权重。因此本实验识别的是六种完整配置的表现，不能证明添加 layer4 在保持 logits 权重时仍有同样收益。

四宫格中，CORAL 组的源域 CE 后期略高而目标准确率较高，与对齐约束抑制源域过度专门化的解释相容，但曲线不能证明这一机制。
无 CORAL 对照的 logits 协方差距离持续增大，也不能直接等同于目标准确率同比下降；CORAL 距离是二阶统计差异，不是分类误差本身。不同层的统计对象及归一化尺度不同，只在同一层内比较原始距离。

本轮可支持的结论是：**预训练 ResNet-50 在共享 BN 协议下已有较好的迁移表现，CORAL 的额外收益对对齐位置敏感；单独特征对齐收益有限，包含 logits 的对齐观察到正向改善。**

## 损失平衡与早停

CE 与实际加权 CORAL 分别维护 EMA（decay=0.5）。相对差距为 `abs(CE-weighted_CORAL)/max(CE,weighted_CORAL,1e-12)`；第 20 轮起，当轮与 EMA 均 ≤25%，连续两轮满足即停止。以下为各组最后一轮记录，连续次数按两项条件同时满足计数。

| 设置 | 停止轮 | CE | 加权 CORAL | 当轮差距 | EMA 差距 | 连续次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{balance_table}

![当轮与 EMA 早停信号](stopping_signals.png)

最后一轮，pooled 的加权对齐项为 CE 的 {alignment_ratios['pooled']:.2f}%，layer4 为 {alignment_ratios['layer4']:.2f}%，均低于满足当轮条件所需的 75%。日志回放确认，它们在 100 轮内没有连续两轮同时满足当轮与 EMA 条件。
这反映当前统计对象、损失尺度、固定权重和训练过程的共同影响，不能证明特征不存在域差异，也不能推出所有特征对齐方法无效。
三个含 logits 的组触发早停，但损失平衡不等于目标准确率峰值：它们的最佳目标准确率都在第 22 轮，而停止发生于第 37 或 42 轮。停止规则未读取这些最佳轮信息。

## 训练成本

| 设置 | 更新步数 | 训练及评估耗时（分钟） | 峰值 GPU 分配显存（GiB） |
| --- | ---: | ---: | ---: |
{costs}

耗时包含训练和两域评估，反映本机实际运行条件，不用于严格计算效率排名。六组共完成 {sum(r['train_steps'] for r in summaries):,} 次模型更新，未修改 batch 或精度，未保存模型 checkpoint。

## 方法与验证

五个 CORAL 组最多 100 轮，第 20 轮起，当轮和 EMA（decay=0.5）的 CE/加权 CORAL 相对差距都 ≤25%，连续两轮后停止。no_coral 跑满 100 轮。停止仅依赖训练损失；最佳轮标签只用于补充归档。
六组逐轮 CSV、train.log、摘要、最佳/停止轮预测及 NPZ 已核对，早停已回放到最早触发点；每轮 24 步、BatchNorm 每模块更新 48 次，评估未修改统计量。每次评估覆盖 Amazon 2817 张和 Webcam 795 张。
layer4 为残差相加及最后 ReLU 后的空间特征，pooled 为其空间 mean，logits 为 31 类分类输出。按层查看 [layer4](coral_layer4.png)、[pooled](coral_pooled.png)、[logits](coral_logits.png)，不跨层排名原始距离。

这是单种子、单域对结果。batch=32 的图像协方差秩最多 31，空间位置相关；多层均值同时改变每层权重。历史 AlexNet 的 batch、骨干及预训练协议不同，本轮没有匹配的跨架构对照，不能据此证明骨干优越或统计显著性。
目标准确率用于事后描述和选择四宫格展示对象，因此不能把该设置在本次比较中的领先解释成对其他域对或新随机种子已验证的优势。
验证结构贡献需要匹配预训练和训练协议的骨干对照；分离 BN 贡献需要源域独立更新与共享更新的对照；验证 CORAL 收益稳健性需要多种子和更多域对。本报告没有执行这些额外实验。

## 归档与复现

数据见 [summary.csv](summary.csv)、[epoch_metrics.csv](epoch_metrics.csv)；配置及运行环境见 [provenance.json](provenance.json)。
本目录保存可提交的报告、CSV/JSON 快照与 PNG；本地完整日志及最佳/停止轮两域预测保存在 `results/resnet50/<setting>/`，已有 AlexNet 归档保持原样。

```powershell
# 从完整本地结果重新校验并生成本报告和图表，不训练
uv run --no-sync python -m src.plotting.plot_resnet50
# 配置及环境匹配时跳过完整运行；缺失组才会训练
uv run --no-sync python -m src.experiments.resnet50.run --resume
```
'''
    (comparison / 'report.md').write_text(report, encoding='utf-8')
    if root.resolve() == suite_results_dir('resnet50').resolve():
        archive = ROOT / 'reports' / 'resnet50'
        archive.mkdir(parents=True, exist_ok=True)
        for name in ('report.md', 'summary.csv', 'epoch_metrics.csv', 'provenance.json',
                     'stop_comparison.png', 'target_accuracy.png', 'plots.png',
                     'stopping_signals.png',
                     'coral_layer4.png', 'coral_pooled.png', 'coral_logits.png'):
            if name == 'report.md':
                (archive / name).write_text(report.replace(f'({protocol_link})', '(../resnet50-protocol.md)'),
                                           encoding='utf-8')
            else:
                shutil.copyfile(comparison / name, archive / name)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_dir', type=Path, default=suite_results_dir('resnet50'))
    args = parser.parse_args()
    print(plot_experiments(args.results_dir))


if __name__ == '__main__':
    main()
