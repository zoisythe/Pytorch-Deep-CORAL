"""Compare CSV metrics using the single archived original control pair."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.common.experiment_protocol import SUITES
from src.common.paths import CONV_EXPERIMENTS_DIR, EXPERIMENTS_DIR, ORIGINAL_DIR, RESULTS_DIR, ROOT, SEED, setting_dir
from src.plotting.plot_accuracies_losses import assert_comparable, read_run, validate_environments

LAYERS = ('conv5', 'fc6', 'fc7', 'fc8')
STYLES = {
    'no_adaptation': ('#626970', '--', None),
    'fc6': ('#AD7A23', '-', 's'),
    'fc7': ('#C66531', '-', '^'),
    'fc8': ('#2459A8', '-', 'o'),
    'conv5': ('#617F43', '-', 'D'),
    'fc6_fc7': ('#AD7A23', '-.', 's'),
    'fc6_fc8': ('#A44E7A', '-.', '^'),
    'fc7_fc8': ('#2459A8', '--', 'v'),
    'fc6_fc7_fc8': ('#A44E7A', ':', 'X'),
    'conv5_fc8': ('#617F43', '--', 'D'),
}
SINGLE = ('no_adaptation', 'fc6', 'fc7', 'fc8', 'conv5')
MULTI = ('no_adaptation', 'fc8', 'fc6_fc7', 'fc6_fc8', 'fc7_fc8', 'fc6_fc7_fc8', 'conv5_fc8')


def label(setting):
    return 'No adaptation' if setting == 'no_adaptation' else setting.replace('_', '+')


def load_suite(root, suite, seed):
    if seed != SEED:
        raise ValueError('All experiments require seed=42')
    runs = {}
    for setting in ('no_adaptation', *('_'.join(layers) for layers in SUITES[suite]['settings'])):
        directory = setting_dir(root, setting)
        config, summary, csv_data = read_run(directory)
        if config['setting'] != setting:
            raise ValueError(f'Incorrect setting directory: {directory}')
        if config.get('schema_version') != 3:
            raise ValueError(f'Rerun {setting} with the all-seed-42 protocol before comparing')
        data = csv_data
        environment = json.loads((directory / 'environment.json').read_text(encoding='utf-8'))
        runs[setting] = {'config': config, 'summary': summary, 'data': data,
                         'environment': environment, 'directory': directory,
                         'suite': 'original' if setting in ('no_adaptation', 'fc8') else suite}
    baseline = runs['no_adaptation']
    if baseline['config']['lambda_coral'] != 0:
        raise ValueError('The baseline must have zero adaptation weight')
    for setting, run in runs.items():
        assert_comparable(run['config'], baseline['config'])
        validate_environments(run['directory'], baseline['directory'])
    return runs


def validate_bridge(fc, conv):
    """Both suites must use the very same frozen control directories and data."""
    for setting in ('no_adaptation', 'fc8'):
        if fc[setting]['directory'] != conv[setting]['directory']:
            raise ValueError('Both suites must reference results/original')
        for key in fc[setting]['data']:
            if not np.array_equal(fc[setting]['data'][key], conv[setting]['data'][key]):
                raise ValueError(f'Archived control differs between readers: {setting}/{key}')
    weights = {run['config']['lambda_coral'] for setting, run in (*fc.items(), *conv.items())
               if setting != 'no_adaptation'}
    if len(weights) != 1:
        raise ValueError('Comparison requires a common total adaptation coefficient')
    return {'shared_controls': {setting: str(fc[setting]['directory']) for setting in ('no_adaptation', 'fc8')}}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(runs):
    baseline_accuracy = runs['no_adaptation']['data']['target_accuracy'][-1]
    fc8_accuracy = runs['fc8']['data']['target_accuracy'][-1]
    rows = []
    for setting, run in runs.items():
        config, summary, data = run['config'], run['summary'], run['data']
        active = config['coral_layers'] if config['lambda_coral'] > 0 else []
        best_index = int(data['target_accuracy'].argmax())
        rows.append({
            'setting': setting, 'suite': run['suite'], 'seed': config['seed'],
            'epochs': len(data['epoch']), 'final_source_accuracy': data['source_accuracy'][-1],
            'final_target_accuracy': data['target_accuracy'][-1],
            'final_target_correct': int(data['target_correct'][-1]),
            'target_samples': int(data['target_samples'][-1]),
            'vs_baseline_pp': 100 * (data['target_accuracy'][-1] - baseline_accuracy),
            'vs_fc8_pp': 100 * (data['target_accuracy'][-1] - fc8_accuracy),
            'best_target_accuracy': data['target_accuracy'][best_index],
            'best_target_epoch': int(data['epoch'][best_index]),
            'final_classification_loss': data['classification_loss'][-1],
            'final_weighted_CORAL_loss': data['weighted_CORAL_loss'][-1],
            **{f'final_CORAL_{layer}': data[f'CORAL_{layer}'][-1] if f'CORAL_{layer}' in data else None
               for layer in LAYERS},
            **{f'weight_{layer}': config['lambda_coral'] / len(active) if layer in active else 0.0
               for layer in LAYERS},
            'elapsed_seconds': summary['elapsed_seconds'],
            'peak_gpu_memory_bytes': summary['peak_gpu_memory_bytes'],
            'metrics_path': str(run['directory'] / 'metrics.csv'),
            'source_shuffle_seed': config['source_initial_shuffle_seed'],
            'target_shuffle_seed': config['target_initial_shuffle_seed'],
            'sampling_matches_original': config['target_initial_shuffle_seed'] == runs['no_adaptation']['config']['target_initial_shuffle_seed'],
        })
    rows.sort(key=lambda row: runs[row['setting']]['data']['target_accuracy'][-1], reverse=True)
    for rank, row in enumerate(rows, 1):
        row['display_order'] = rank  # Equal accuracies remain ties, not strict ranks.
    return rows


def prepare_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#26313B', 'text.color': '#26313B',
                         'savefig.facecolor': 'white'})


def save_figure(fig, output, name, caption):
    fig.supxlabel(caption, fontsize=9)
    fig.savefig(output / f'{name}.png', dpi=200)
    plt.close(fig)


def plot_rankings(rows, output, scope):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout='constrained', sharey=True,
                             gridspec_kw={'width_ratios': [1.3, 1]})
    positions = np.arange(len(rows))
    accuracy = [100 * row['final_target_accuracy'] for row in rows]
    deltas = [row['vs_fc8_pp'] for row in rows]
    bars = axes[0].barh(positions, accuracy, color='#3875A8', height=0.65)
    axes[0].bar_label(bars, labels=[f'{value:.2f}%' for value in accuracy], padding=5)
    baseline = next(row for row in rows if row['setting'] == 'no_adaptation')
    axes[0].axvline(100 * baseline['final_target_accuracy'], color='#626970', linestyle='--', linewidth=1.5,
                   label=f'Baseline: {100 * baseline["final_target_accuracy"]:.2f}%')
    axes[0].set(xlim=(0, 65), xlabel='Final target accuracy (%)', title='Accuracy at epoch 50',
                yticks=positions, yticklabels=[label(row['setting']) for row in rows])
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower left', frameon=False)
    delta_bars = axes[1].barh(positions, deltas, height=0.65,
                              color=['#3875A8' if value >= 0 else '#B6C4D2' for value in deltas])
    axes[1].bar_label(delta_bars, labels=[f'{value:+.2f} pp' for value in deltas], padding=5)
    axes[1].axvline(0, color='#626970', linewidth=1)
    axes[1].set(xlim=(min(deltas) - 2.0, max(deltas) + 2.0),
                xlabel='Difference from fc8 (percentage points)', title='Comparison with fc8')
    for ax in axes:
        ax.grid(axis='x', alpha=0.2)
        ax.set_axisbelow(True)
    save_figure(fig, output, 'final_target_comparison', scope + '; one seed, no uncertainty estimate')


def plot_lines(ax, runs, settings, key, percent=False, log=False):
    for setting in settings:
        data = runs[setting]['data']
        if key not in data:
            continue
        values = data[key] * (100 if percent else 1)
        if log:
            # Preserve zero as zero in exported data; omit it on a logarithmic axis.
            values = np.where(values > 0, values, np.nan)
            if np.isnan(values).all():
                continue
        color, linestyle, marker = STYLES[setting]
        ax.plot(data['epoch'], values, label=label(setting), color=color, linestyle=linestyle,
                marker=marker, markevery=10, markersize=4, linewidth=1.6)
    ax.set(xlim=(1, 50), xticks=[1, 10, 20, 30, 40, 50], xlabel='Epoch')
    if log:
        ax.set_yscale('log')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, ncol=2, frameon=False)


def plot_accuracy_curves(runs, output, scope):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout='constrained', sharex=True, sharey='row')
    target_values = np.concatenate([run['data']['target_accuracy'] for run in runs.values()]) * 100
    target_limits = (5 * np.floor(target_values.min() / 5), 5 * np.ceil(target_values.max() / 5))
    source_values = np.concatenate([run['data']['source_accuracy'] for run in runs.values()]) * 100
    source_limits = (5 * np.floor(source_values.min() / 5), 100.5)
    for column, (settings, title) in enumerate(((SINGLE, 'Single-layer settings'), (MULTI, 'Multi-layer settings'))):
        for row, domain in enumerate(('target', 'source')):
            ax = axes[row, column]
            plot_lines(ax, runs, settings, f'{domain}_accuracy', percent=True)
            ax.set(title=f'{title}: {domain} domain', ylabel=f'{domain.capitalize()} accuracy (%)')
            if domain == 'target':
                ax.set_ylim(*target_limits)
            else:
                ax.set_ylim(*source_limits)
    save_figure(fig, output, 'accuracy_curves', scope + '; target axes show a detail range; source is training-domain evaluation')


def plot_loss_curves(runs, output, scope):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout='constrained', sharex=True, sharey='row')
    for column, (settings, title) in enumerate(((SINGLE, 'Single-layer settings'), (MULTI, 'Multi-layer settings'))):
        for row, (key, name) in enumerate((('classification_loss', 'Classification loss'),
                                         ('weighted_CORAL_loss', 'Weighted adaptation loss'))):
            ax = axes[row, column]
            plot_lines(ax, runs, settings, key, log=True)
            ax.set(title=f'{title}: {name.lower()}', ylabel=f'{name} (log scale)')
    save_figure(fig, output, 'loss_curves', scope + '; epoch training-batch means; zero baseline adaptation omitted on log axes')


def plot_layer_distances(runs, output, scope):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout='constrained')
    for layer, ax in zip(LAYERS, axes.flat):
        plot_lines(ax, runs, runs.keys(), f'CORAL_{layer}', log=True)
        ax.set(title=f'{layer}: recorded CORAL distances', ylabel='CORAL distance (log scale)')
    save_figure(fig, output, 'coral_by_layer', scope + '; independent layer scales; unlogged layers omitted, not filled with zero')


def write_provenance(output, source_runs, duplicates, args, rows):
    manifests = []
    for run in source_runs:
        directory = run['directory']
        manifests.append({'suite': run['suite'], 'setting': run['config']['setting'], 'directory': str(directory),
                          'input_files': ['metrics.csv', 'config.json', 'summary.json', 'environment.json'],
                          'source_shuffle_seed': run['config']['source_initial_shuffle_seed'],
                          'target_shuffle_seed': run['config']['target_initial_shuffle_seed']})
    manifest = {
        'schema_version': 2, 'primary_source': 'metrics.csv', 'csv_validation': 'finite epochs, loss identities, sample counts, summary equality',
        'source_run_count': len(manifests), 'unique_setting_count': len(rows), 'duplicates': duplicates,
        'bridge': 'Shared archived controls; common hyperparameters, all RNG seeds and runtime match.',
        'canonical_fc8': str(ORIGINAL_DIR / 'fc8'),
        'baseline': str(ORIGINAL_DIR / 'no_adaptation'),
        'historical_sampling_difference': any(not row['sampling_matches_original'] for row in rows),
        'source_runs': manifests,
        'renderer': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__},
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    execution_path = RESULTS_DIR / 'execution.json'
    if execution_path.exists():
        manifest['execution'] = json.loads(execution_path.read_text(encoding='utf-8'))
    (output / 'provenance.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_summary(output, rows, scope):
    by_setting = {row['setting']: row for row in rows}
    def accuracy(setting):
        return f'{100 * by_setting[setting]["final_target_accuracy"]:.2f}%'
    best = rows[0]
    target_count = best['target_samples']
    joint = by_setting['conv5_fc8']
    extra_correct = joint['final_target_correct'] - by_setting['fc8']['final_target_correct']
    equal_fc8 = '，两者本轮并列' if by_setting['fc8']['final_target_accuracy'] == by_setting['fc7_fc8']['final_target_accuracy'] else ''
    hidden_settings = ('fc6', 'fc7', 'fc6_fc7')
    hidden_result = '本轮三组均低于基线' if all(by_setting[name]['vs_baseline_pp'] < 0 for name in hidden_settings) else '各组相对基线的方向见上表'
    contains_fc8 = ('fc8', 'fc6_fc8', 'fc7_fc8', 'fc6_fc7_fc8', 'conv5_fc8')
    fc8_result = '本轮含 fc8 的全部配置高于基线' if all(by_setting[name]['vs_baseline_pp'] > 0 for name in contains_fc8) else '含 fc8 配置的相对基线结果见上表'
    table = '\n'.join(f'| {label(row["setting"])} | {100 * row["final_source_accuracy"]:.2f}% | '
                      f'{100 * row["final_target_accuracy"]:.2f}% | {row["final_target_correct"]}/{target_count} | '
                      f'{row["vs_baseline_pp"]:+.2f} | {row["vs_fc8_pp"]:+.2f} |'
                      for row in rows)
    distances = '\n'.join('| ' + label(row['setting']) + ' | ' + ' | '.join(
        '未记录' if row[f'final_CORAL_{layer}'] is None else f'{row[f"final_CORAL_{layer}"]:.6f}'
        for layer in LAYERS) + ' |' for row in rows)
    text = f'''# CORAL 层位置实验综合比较

直接读取十种设置的 `metrics.csv`，核对轮次、损失恒等式、评估样本数及 summary。no_adaptation 和 fc8 直接来自此次重跑的 `results/original/`，额外实验不再保留对照副本。条件：{scope}；batch size=128、每轮六次更新、SGD lr=0.001、fc8 lr=0.01、momentum=0.9、weight decay=0.0005、严格确定性。

原始两组及全部进阶组均已按全 seed=42 协议重跑，源域、目标域、评估和 worker seed 均为 42。

## 最终准确率

主指标为第 50 轮目标准确率；排序只描述本轮数值，相同准确率为并列。

| 设置 | 源域准确率 | 目标准确率 | 目标正确数 | 相对基线（百分点） | 相对 fc8（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

## 观察与解释

1. **本轮最终最高为 {label(best['setting'])}：{100 * best['final_target_accuracy']:.2f}%**。conv5+fc8 相对基线 {joint['vs_baseline_pp']:+.2f} 个百分点，相对 fc8 {joint['vs_fc8_pp']:+.2f} 个百分点，对应目标正确数差 {extra_correct:+d}/{target_count}。
2. fc8 为 **{accuracy('fc8')}**，fc7+fc8 为 **{accuracy('fc7_fc8')}{equal_fc8}**；fc6+fc8 为 {accuracy('fc6_fc8')}，fc6+fc7+fc8 为 {accuracy('fc6_fc7_fc8')}。本轮加入更多 fc 层没有带来单调提高。conv5 单层为 {accuracy('conv5')}。
3. fc6、fc7、fc6+fc7 最终分别为 {accuracy('fc6')}、{accuracy('fc7')}、{accuracy('fc6_fc7')}；{hidden_result}。{fc8_result}。
4. conv5 单层的局部通道距离为 {by_setting['conv5']['final_CORAL_conv5']:.6f}，联合组为 {joint['final_CORAL_conv5']:.6f}；对应目标准确率分别为 {accuracy('conv5')} 和 {accuracy('conv5_fc8')}。局部距离更低不直接意味着目标分类更好。源域最终准确率均接近 100%，不能据此认为目标域同样具有良好泛化。

单层损失系数 {by_setting['fc8']['weight_fc8']:.3g}；双层各 {joint['weight_fc8']:.3g}；三层各 {by_setting['fc6_fc7_fc8']['weight_fc8']:.6g}。总系数相同并不保证实际正则强度或梯度尺度相同。这里同时改变层配置与各层权重，因此不将结果解释为某一新增层的独立因果作用。本轮仅 seed={best['seed']}、仅 Amazon→Webcam，未估计统计显著性或跨域稳健性。源域评估使用源域训练数据，目标标签仅用于评估，不用于模型选择。

## 各层距离

以下为第 50 轮六个训练批次的均值，不是冻结最终模型后的全域距离。各层维数、激活尺度及统计对象不同，只在同一层内部比较。未记录的距离保持缺失。

| 设置 | conv5 CORAL | fc6 CORAL | fc7 CORAL | fc8 CORAL |
| --- | ---: | ---: | ---: | ---: |
{distances}

conv5 使用 `[BHW,256]` 局部通道观测与 `256×256` 协方差；fc 层使用图像为样本的二维特征与原 Gram 公式。空间观测相关，`BHW` 不等于独立图像数。原始组监测全部四层；不同设置的原始 CORAL 均值不一定包含相同层集合，应比较同层距离和实际加权适应项。

## 来源与图表

配置 schema、数据路径和实际采样 seed 保留在 `provenance.json`。共同超参数和运行环境均匹配。所有基线/fc8 指标取自新归档，不拼接旧对照数据。原训练入口的续跑要求配置和运行环境匹配。

- `final_target_comparison.png`：十种设置的最终准确率及相对 fc8 的差值。
- `accuracy_curves.png`：单层与多层配置的目标、源域准确率曲线；目标轴为标明范围的细节视图。
- `loss_curves.png`：分类损失及实际加权适应项，使用对数纵轴；零适应基线不画在适应项对数轴上。
- `coral_by_layer.png`：conv5、fc6、fc7、fc8 分层距离，各层独立对数尺度。
- `summary.csv`：准确率、差值、最佳轮次、最终损失、每层距离、有效权重与实际运行成本；缺失距离为空。
- `epoch_metrics.csv`：去重后按设置、epoch 导出的全部已记录指标，缺失值为空。

图表均不平滑、不补 epoch 0、不加入未估计的误差棒。耗时和显存为此次单次运行记录；早期进阶设置使用两个独立进程，出现一次进程退出后，未完成设置改为串行重跑（执行记录见 `results/execution.json`）。监测层数与资源竞争不同，不据此得出算法效率结论。

在仓库根目录复现：

```powershell
uv run --no-sync python -m src.plotting.plot_all_experiments
```
'''
    (output / 'report.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fc_results_dir', type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument('--conv_results_dir', type=Path, default=CONV_EXPERIMENTS_DIR)
    parser.add_argument('--output_dir', type=Path, default=RESULTS_DIR / 'combined_experiments')
    parser.add_argument('--seed', type=int, choices=(SEED,), default=SEED)
    args = parser.parse_args()
    fc = load_suite(args.fc_results_dir, 'fc_layers', args.seed)
    conv = load_suite(args.conv_results_dir, 'conv5_fc8', args.seed)
    duplicates = validate_bridge(fc, conv)
    reference = fc['no_adaptation']['config']
    if reference['epochs'] != 50 or reference['target_samples'] != 795 or reference['source'] != 'amazon' or reference['target'] != 'webcam':
        parser.error('This comparison is defined for the 50-epoch Amazon-to-Webcam experiment with 795 target images')
    runs = {setting: {**run, 'data': dict(run['data'])} for setting, run in fc.items()}
    runs.update({setting: conv[setting] for setting in ('conv5', 'conv5_fc8')})
    rows = summary_rows(runs)
    output = args.output_dir
    input_roots = (args.fc_results_dir.resolve(), args.conv_results_dir.resolve(), ORIGINAL_DIR.resolve())
    if any(output.resolve().is_relative_to(root) for root in input_roots):
        parser.error('Use a separate output directory to preserve the original experiment outputs')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'summary.csv', rows)
    metric_keys = list(dict.fromkeys(key for run in runs.values() for key in run['data']))
    epoch_rows = [{'setting': setting, **{key: data[key][index] if key in data else None for key in metric_keys}}
                  for setting, run in runs.items() for data in [run['data']] for index in range(len(data['epoch']))]
    write_csv(output / 'epoch_metrics.csv', epoch_rows)
    scope = f'Office-31 Amazon to Webcam | seed {args.seed} | 50 epochs | target n=795'
    prepare_style()
    plot_rankings(rows, output, scope)
    plot_accuracy_curves(runs, output, scope)
    plot_loss_curves(runs, output, scope)
    plot_layer_distances(runs, output, scope)
    write_provenance(output, list(runs.values()), duplicates, args, rows)
    write_summary(output, rows, scope)
    print(f'Validated {len(rows)} CSV runs using shared original controls; outputs: {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}')
    print(f'Best final setting: {rows[0]["setting"]}, target={100 * rows[0]["final_target_accuracy"]:.2f}%')


if __name__ == '__main__':
    main()
