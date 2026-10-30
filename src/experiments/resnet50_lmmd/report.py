"""Validate and archive the single run, keeping external comparisons explicit."""

import json
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.common.paths import ROOT
from src.experiments.resnet50_lmmd.protocol import REPORT_DIR, RUN_DIR
from src.experiments.resnet50_lmmd.verify import read_csv, validate_run
from src.training.evaluation import write_csv
from src.training.train import write_json


def build_report():
    summary_path = RUN_DIR / 'summary.json'
    if not summary_path.exists() or not json.loads(summary_path.read_text(encoding='utf-8'))['training_complete']:
        raise RuntimeError('The single training run must finish before report generation')
    # The common prediction writer has a legacy fixed/adaptive tag for CORAL.
    # Record this experiment's actual schedule and selection semantics explicitly.
    for name in ('best_epoch', 'stop_epoch'):
        path = RUN_DIR / f'{name}.json'
        metadata = json.loads(path.read_text(encoding='utf-8'))
        metadata.update(phase='progressive_lmmd', target_labels_in_loss=False,
                        target_labels_in_selection_and_stopping=True,
                        selection_metric='target_accuracy' if name == 'best_epoch'
                        else 'target_accuracy_patience_or_epoch_budget')
        write_json(path, metadata)
    config, summary, rows, verification = validate_run()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    assets = REPORT_DIR / 'assets'
    data = REPORT_DIR / 'data'
    assets.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    for name in ('metrics.csv', 'step_metrics.csv', 'train.log', 'config.json', 'environment.json',
                 'summary.json', 'verification.json', 'best_epoch.json', 'stop_epoch.json'):
        shutil.copy2(RUN_DIR / name, data / name)
    for name in ('best_epoch', 'stop_epoch'):
        shutil.copytree(RUN_DIR / name, data / name, dirs_exist_ok=True)
    epochs = np.array([int(row['epoch']) for row in rows])
    series = lambda key: np.array([float(row[key]) for row in rows])
    best_epoch, stop_epoch = summary['selected_epoch'], summary['stop_epoch']
    best, stopped = summary['selected_target_accuracy'], summary['stop_target_accuracy']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained')
    ax = axes[0]
    ax.plot(epochs, 100 * series('source_accuracy'), color='#6c7885', label='Source training images')
    ax.plot(epochs, 100 * series('target_accuracy'), color='#2463a0', label='Target: Webcam (795 images)')
    ax.scatter([best_epoch], [100 * best], color='#c56c22', zorder=3,
               label=f'Selected best: {100 * best:.2f}% @ {best_epoch}')
    ax.axvline(stop_epoch, color='#6c7885', linestyle=':', label=f'Stop: epoch {stop_epoch}')
    ax.set(xlabel='Epoch (87 updates each)', ylabel='Accuracy (%)', ylim=(0, 100), title='ResNet-50 V2 + LMMD')
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8, loc='lower right')
    ax = axes[1]
    ax.plot(epochs, series('classification_loss'), label='Source cross-entropy', color='#2463a0')
    ax.plot(epochs, series('lmmd_loss'), label='Raw LMMD', color='#6c7885', linestyle='--')
    ax.plot(epochs, series('weighted_lmmd_loss'), label='Weighted LMMD', color='#c56c22')
    ax.set(xlabel='Epoch', ylabel='Mean training loss', title='Losses (different numerical scales)')
    ax.grid(alpha=0.18)
    ax.legend(fontsize=9)
    fig.savefig(assets / 'accuracy_losses.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    axes[0].plot(epochs, series('backbone_lr'), label='Backbone', color='#2463a0')
    axes[0].plot(epochs, series('head_lr'), label='Bottleneck / classifier', color='#c56c22')
    axes[0].set(yscale='log', xlabel='Epoch', ylabel='Learning rate', title='Author learning-rate schedule')
    axes[0].legend(fontsize=8)
    axes[1].plot(epochs, series('lmmd_weight'), color='#c56c22')
    axes[1].set(xlabel='Epoch', ylabel='LMMD coefficient', ylim=(0, 0.52), title='Progressive adaptation')
    axes[2].plot(epochs, series('shared_classes'), color='#2463a0')
    axes[2].set(xlabel='Epoch', ylabel='Mean shared classes / batch', ylim=(0, 31), title='Class support (batch = 32)')
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.savefig(assets / 'schedule_support.png', dpi=180)
    plt.close(fig)

    arrays = np.load(RUN_DIR / 'best_epoch' / 'evaluation.npz', allow_pickle=False)
    labels, logits = arrays['target_labels'], arrays['target_logits']
    predictions, names = logits.argmax(1), arrays['classes']
    confusion = np.zeros((31, 31), dtype=int)
    np.add.at(confusion, (labels, predictions), 1)
    normalized = confusion / confusion.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(12, 10), layout='constrained')
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(np.arange(31), names, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(31), names, fontsize=7)
    ax.set(xlabel='Prediction', ylabel='True class', title=f'Selected epoch {best_epoch}: target confusion (row normalized)')
    fig.colorbar(im, ax=ax, label='Fraction of true class', shrink=0.8)
    fig.savefig(assets / 'confusion.png', dpi=180)
    plt.close(fig)
    class_rows = [{'class_index': i, 'label': str(name), 'target_samples': int(confusion[i].sum()),
                       'target_correct': int(confusion[i, i]), 'target_accuracy': float(normalized[i, i])}
                  for i, name in enumerate(names)]
    write_csv(data / 'class_accuracy.csv', class_rows)
    write_csv(data / 'confusion_counts.csv',
              [dict(true_class=str(name), **{str(n): int(confusion[i, j]) for j, n in enumerate(names)})
               for i, name in enumerate(names)])
    comparisons = [{'origin': 'new_run', 'method': 'ResNet50 V2 + LMMD', 'metric': 'selected_best',
                        'epoch': best_epoch, 'target_accuracy': best},
                   {'origin': 'new_run', 'method': 'ResNet50 V2 + LMMD', 'metric': 'training_stop',
                        'epoch': stop_epoch, 'target_accuracy': stopped}]
    for row in read_csv(ROOT / 'reports' / 'resnet50' / 'summary.csv'):
        comparisons.append({'origin': 'historical_coral_different_protocol', 'method': row['setting'],
                                'metric': 'training_stop', 'epoch': int(row['stop_epoch']),
                                'target_accuracy': float(row['target_accuracy'])})
    comparisons.extend([{'origin': 'paper_three_trials', 'method': method, 'metric': 'reported_mean', 'epoch': '',
                             'target_accuracy': accuracy} for method, accuracy in
                        [('ResNet50 source-only', 0.684), ('D-CORAL', 0.777), ('DSAN', 0.936)]])
    write_csv(data / 'comparison.csv', comparisons)
    with (REPORT_DIR / 'provenance.json').open('w', encoding='utf-8') as stream:
        json.dump({'date': '2026-10-12', 'source_run': str(RUN_DIR.relative_to(ROOT)),
                       'training_performed_for_experiment': True, 'training_performed_for_report': False,
                       'controls_retrained': False, 'historical_coral': 'reports/resnet50/summary.csv',
                       'paper': 'https://arxiv.org/pdf/2106.09388', 'paper_table': 'II',
                       'matching_loss_ablation': False, 'config': config}, stream, indent=2, ensure_ascii=False)
    best_row = rows[best_epoch - 1]
    report = f'''# ResNet-50 V2 + LMMD：DSAN 作者代码口径

2026-10-12，Office-31 Amazon→Webcam，seed=42，只新增一组正式训练。已完成 **{stop_epoch} 轮 / {summary['steps_completed']:,} 次更新**，停止原因 `{summary['stop_reason']}`。

按作者代码的目标评估选择口径，最佳轮为 **第 {best_epoch} 轮，{best:.2%}（{round(best * 795)}/795）**；停止轮为 **第 {stop_epoch} 轮，{stopped:.2%}（{round(stopped * 795)}/795）**。最佳轮目标宏平均类别准确率为 {float(best_row['target_macro_accuracy']):.2%}。

**目标真实标签没有进入训练损失和类别权重，但参与每轮评估、最佳模型选择和早停。** 因此本报告的主指标不是预先固定轮次或训练损失决定轮次的指标；其选择口径也不同于旧 CORAL 实验。单种子仅作探索，不宣称统计显著性。

## 实际训练设置

[训练前协议](protocol.md)。ResNet-50 `IMAGENET1K_V2` → 2048 维池化 → Linear(2048,256) → Linear(256,31)，在256维 bottleneck 上进行五核 RBF LMMD。新增 Linear 使用默认初始化，没有额外激活、Dropout 或 BN。共享 BN 每步按源、目标顺序更新，评估冻结；全网络 SGD 微调。

作者基础学习率 0.001/0.01/0.01，按 `(1+10*(epoch-1)/200)^(-0.75)` 衰减；LMMD 系数为 `0.5*(2/(1+exp(-10*epoch/200))-1)`。batch=32，按源域驱动每轮87步，目标域循环；无 CORAL 预热。训练 Resize([256,256])、随机裁剪224、随机水平翻转、ToTensor；评估 Resize([224,224])、ToTensor。跟随作者输入管线，不加 Normalize。

相对原作者改变为 V2 权重、seed=42、当前库版本、workers=0、严格确定性 FP32/禁用TF32、独立 loader 随机数和等价空间 mean 池化。每轮完整评估源域2817张和目标域795张；源准确率是在源训练图片上计算。

本次复用现有GPU虚拟环境 Python 3.12.15；历史CORAL环境为Python 3.14.7，两边PyTorch 2.14.1+cu132、torchvision 0.29.1+cu132和CUDA 13.2相同。没有修改项目Python版本文件或已有依赖变更。作者代码读取快照保存在 `reference/`。

![两域准确率及损失](assets/accuracy_losses.png)

![学习率、LMMD权重和共享类别](assets/schedule_support.png)

## 与已有结果的外部对照

| 来源与指标 | A→W准确率 | 累计更新数 | 轮次/统计口径 |
|---|---:|---:|---|
| 本次 V2＋LMMD 最佳轮 | {best:.2%} | {best_epoch * 87:,} | {best_epoch}，目标评估选择 |
| 本次 V2＋LMMD 停止轮 | {stopped:.2%} | {summary['steps_completed']:,} | {stop_epoch}，30轮无提升早停 |
| 历史 V2＋CORAL layer4＋logits 停止轮 | 81.76% | 1,008 | 42，训练损失平衡停止 |
| 历史 V2＋CORAL layer4＋logits 峰值（补充） | 82.77% | 528 | 22，仅事后描述 |
| 历史 V2 无CORAL、共享BN对照 | 80.00% | 2,400 | 100 |
| DSAN论文 ResNet50 | 68.4±0.5% | — | 三次均值±标准误 |
| DSAN论文 D-CORAL | 77.7±0.3% | — | 三次均值±标准误 |
| DSAN论文 DSAN | 93.6±0.2% | — | 三次均值±标准误 |

论文数据来自[原文 Table II](https://arxiv.org/pdf/2106.09388)，历史数据来自[ResNet CORAL报告](../resnet50/report.md)。本次最佳轮相对历史 CORAL 停止轮的数值差为 **{(best - 650 / 795) * 100:+.2f} 个百分点**，相对论文 DSAN 均值为 **{(best - 0.936) * 100:+.2f} 个百分点**。这些是不同训练配置/选择口径的成绩差，不是损失函数的因果增益。

本次还改变了 bottleneck、增强/归一化、学习率调度、权重调度、每轮更新数和模型选择规则；没有重跑匹配的 CORAL 或 source-only 控制组，也没有复现三次论文试验。不能仅凭这一组拆分各因素贡献，或判断 V2 优于原文权重。

## 分类细节、成本和验收

![最佳轮目标混淆矩阵](assets/confusion.png)

逐类准确率见 [class_accuracy.csv](data/class_accuracy.csv)，混淆计数见 [confusion_counts.csv](data/confusion_counts.csv)。训练及两域逐轮评估总耗时 **{summary['elapsed_seconds'] / 60:.2f} 分钟**，峰值GPU分配显存 **{summary['peak_gpu_memory_bytes'] / 2**30:.2f} GiB**；耗时不包括进程启动和模型初始化，不能跨配置作严格效率排名。

验收通过：{verification['epochs']}条逐轮记录、{verification['steps']:,}条更新记录、{verification['class_rows']:,}条逐类记录；87步/轮、调度、损失聚合、日志/CSV/摘要、最佳/停止轮NPZ与逐图预测、checkpoint轮次/结构及最早早停位置一致。重新加载最佳权重，对全部795张目标图像独立推理，logits最大差为 {verification['selected_checkpoint_predictions']['maximum_logit_error']:.3g}，正确数一致。正式运行校验了参数覆盖、bottleneck梯度、BN训练更新与评估冻结、有限损失/梯度。256维特征的独立作者实现对照见 [author_validation.json](author_validation.json)，特征梯度最大差 {verification['author_formula']['maximum_feature_gradient_error']:.3g}。

可审查文件：[逐轮指标](data/metrics.csv)、[逐步指标](data/step_metrics.csv)、[配置](data/config.json)、[环境](data/environment.json)、[摘要](data/summary.json)、[验收](data/verification.json)、[外部对照表](data/comparison.csv)、[来源](provenance.json)。最佳/停止轮预测和logits位于 `data/best_epoch/`、`data/stop_epoch/`。完整逐类训练统计及模型权重位于本地 `results/resnet50_lmmd/author_code_seed42/`，大模型文件不复制到报告归档。

## 运行命令

```powershell
uv run --no-sync --python 3.12 python -m src.experiments.resnet50_lmmd.run
# 只重新核验与生成报告，不训练
uv run --no-sync --python 3.12 python -m src.experiments.resnet50_lmmd.run --report_only
```
'''
    (REPORT_DIR / 'report.md').write_text(report, encoding='utf-8')
    return REPORT_DIR / 'report.md'


if __name__ == '__main__':
    print(build_report())
