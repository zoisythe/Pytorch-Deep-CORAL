"""Compare delayed adaptive CORAL with the matching fixed-weight layer group."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.common.paths import ORIGINAL_DIR, RESULTS_DIR, SEED
from src.plotting.plot_accuracies_losses import assert_comparable, environment_differences, read_run


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows or any(not all(math.isfinite(value) for value in row.values()) for row in rows):
        raise ValueError(f'Empty or non-finite metrics: {path}')
    return rows


def close(left, right):
    return math.isclose(left, right, rel_tol=2e-5, abs_tol=1e-7)


def read_adaptive(directory):
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    epochs, steps = read_csv(directory / 'metrics.csv'), read_csv(directory / 'step_metrics.csv')
    layers = tuple(config['coral_layers'])
    if config['schema_version'] not in (4, 5) or layers not in (('fc8',), ('conv5', 'fc8')) or config['monitor_layers'] != list(layers):
        raise ValueError('Expected adaptive fc8 or conv5+fc8 CORAL')
    if config['lambda_mode'] not in ('gradient_norm_balance', 'fixed') or config['seed_policy'] != 'all_rngs_42':
        raise ValueError('Unexpected weight or sampling protocol')
    if config['seed'] != SEED or any(config[key] != SEED for key in (
            'source_initial_shuffle_seed', 'target_initial_shuffle_seed', 'worker_seed')):
        raise ValueError('Every experiment seed must be 42')
    if len(epochs) != config['epochs'] or summary['epochs_completed'] != len(epochs):
        raise ValueError('Incomplete adaptive experiment')
    batches = config['train_steps_per_epoch']
    if len(steps) != len(epochs) * batches or summary['steps_completed'] != len(steps):
        raise ValueError('Incomplete step metrics')
    previous = config['lambda_coral']
    start_epoch = config.get('lambda_learning_start_epoch', 1)
    for index, row in enumerate(steps):
        if (row['global_step'], row['epoch'], row['step_in_epoch']) != (
                index + 1, index // batches + 1, index % batches + 1):
            raise ValueError('Incorrect step sequence')
        if not close(row['lambda_used'], previous):
            raise ValueError('Incorrect pre-update model weight')
        if config['schema_version'] == 5:
            learning = int(row['epoch'] >= start_epoch)
            if row['lambda_learning'] != learning:
                raise ValueError('Incorrect lambda learning schedule')
            if not learning and (row['lambda_used'] != config['lambda_coral']
                                 or row['lambda_next'] != config['lambda_coral']):
                raise ValueError('Lambda changed during the fixed phase')
        if not config['lambda_min'] * (1 - 1e-6) <= row['lambda_used'] <= config['lambda_max'] * (1 + 1e-6):
            raise ValueError('Adaptive weight outside its bounds')
        if not close(row['weighted_CORAL_loss'], row['lambda_used'] * row['CORAL_loss']):
            raise ValueError('Incorrect per-step CORAL weight')
        if any(row[f'CORAL_{layer}'] < 0 for layer in layers) or not close(
                row['CORAL_loss'], sum(row[f'CORAL_{layer}'] for layer in layers) / len(layers)):
            raise ValueError('Incorrect CORAL layer reduction')
        if not close(row['total_loss'], row['classification_loss'] + row['weighted_CORAL_loss']):
            raise ValueError('Incorrect per-step model objective')
        if not close(row['weighted_gradient_CORAL_ema'], row['lambda_used'] * row['gradient_CORAL_ema']):
            raise ValueError('Incorrect gradient balance measurement')
        expected = (row['weighted_gradient_CORAL_ema'] - row['gradient_CE_ema']) ** 2
        if not close(row['weight_objective'], expected):
            raise ValueError('Incorrect adaptive weight objective')
        previous = row['lambda_next']
    for index, row in enumerate(epochs):
        group = steps[index * batches:(index + 1) * batches]
        if row['epoch'] != index + 1 or row['train_steps'] != batches:
            raise ValueError('Incorrect epoch sequence or batch count')
        for key in group[0].keys() - {'global_step', 'epoch', 'step_in_epoch'}:
            if not close(row[key], sum(step[key] for step in group) / batches):
                raise ValueError(f'Incorrect epoch average for {key}')
        for key, expected in [('lambda_min', min(step['lambda_used'] for step in group)),
                              ('lambda_max', max(step['lambda_used'] for step in group)),
                              ('lambda_end', group[-1]['lambda_next'])]:
            if not close(row[key], expected):
                raise ValueError(f'Incorrect {key}')
        for domain in ('source', 'target'):
            if row[f'{domain}_samples'] != config[f'{domain}_samples'] or not close(
                    row[f'{domain}_accuracy'], row[f'{domain}_correct'] / row[f'{domain}_samples']):
                raise ValueError('Incorrect full-domain evaluation')
    best = max(epochs, key=lambda row: row['target_accuracy'])
    if any(summary[f'final_{domain}_accuracy'] != epochs[-1][f'{domain}_accuracy']
           for domain in ('source', 'target')) or summary['final_lambda'] != steps[-1]['lambda_next']:
        raise ValueError('Summary differs from final metrics')
    if (summary['best_target_epoch'], summary['best_target_accuracy']) != (best['epoch'], best['target_accuracy']):
        raise ValueError('Incorrect best epoch summary')
    if config['schema_version'] == 5:
        saved = json.loads((directory / 'best_epoch.json').read_text(encoding='utf-8'))
        if saved['metrics'] != best or saved['epoch'] != summary['selected_epoch']:
            raise ValueError('Saved best epoch differs from the training metrics')
        if summary['lambda_updates'] != (config['epochs'] - start_epoch + 1) * batches:
            raise ValueError('Incorrect lambda update count')
    return (config, summary, {key: np.array([row[key] for row in epochs]) for key in epochs[0]}), \
        {key: np.array([row[key] for row in steps]) for key in steps[0]}


def save(figure, directory, name):
    figure.savefig(directory / f'{name}.png', dpi=180)
    plt.close(figure)


def read_fixed_reference(directory):
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if config['schema_version'] == 3:
        return read_run(directory)
    if config.get('lambda_mode') != 'fixed':
        raise ValueError('Comparison reference must use a fixed lambda')
    return read_adaptive(directory)[0]


def plot_experiment(directory):
    directory = Path(directory)
    adaptive, steps = read_adaptive(directory)
    layers = adaptive[0]['coral_layers']
    setting, display = '_'.join(layers), '+'.join(layers)
    fixed_key, adaptive_key = f'{setting}_fixed', f'{setting}_adaptive'
    fixed_reference = Path(adaptive[0].get('fixed_reference', ORIGINAL_DIR / 'fc8'))
    runs = {'no_adaptation': read_run(ORIGINAL_DIR / 'no_adaptation'),
            fixed_key: read_fixed_reference(fixed_reference), adaptive_key: adaptive}
    directories = {'no_adaptation': ORIGINAL_DIR / 'no_adaptation',
                   fixed_key: fixed_reference, adaptive_key: directory}
    if runs[fixed_key][0]['coral_layers'] != layers or runs[fixed_key][0]['lambda_coral'] != adaptive[0]['lambda_coral']:
        raise ValueError('Fixed reference uses different layers or initial lambda')
    reference_env = json.loads((fixed_reference / 'environment.json').read_text(encoding='utf-8'))
    for name, (config, _, _) in runs.items():
        assert_comparable(config, runs[fixed_key][0], allow_epoch_difference=True)
        environment = json.loads((directories[name] / 'environment.json').read_text(encoding='utf-8'))
        if environment_differences(environment, reference_env):
            raise ValueError(f'{name} has an incompatible runtime environment')
    output = directory / 'comparison'
    output.mkdir(parents=True, exist_ok=True)
    labels = {'no_adaptation': f'No adaptation ({runs["no_adaptation"][0]["epochs"]} epochs)',
              fixed_key: f'{display}: fixed ({runs[fixed_key][0]["epochs"]} epochs)',
              adaptive_key: f'{display}: adaptive ({adaptive[0]["epochs"]} epochs)'}
    colors = {'no_adaptation': '#64748b', fixed_key: '#2563eb', adaptive_key: '#e86f25'}
    start_epoch = adaptive[0].get('lambda_learning_start_epoch', 1)
    start_step = (start_epoch - 1) * adaptive[0]['train_steps_per_epoch'] + 1
    selected = {name: int(summary['best_target_epoch']) - 1 for name, (_, summary, _) in runs.items()}
    for domain in ('source', 'target'):
        fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
        for name, (_, _, data) in runs.items():
            ax.plot(data['epoch'], 100 * data[f'{domain}_accuracy'], label=labels[name], color=colors[name])
            index = selected[name]
            ax.scatter(data['epoch'][index], 100 * data[f'{domain}_accuracy'][index], color=colors[name], s=35, zorder=3)
        ax.axvline(start_epoch - 0.5, color='#7c3aed', linestyle=':', label=f'Lambda learning from epoch {start_epoch}')
        values = np.concatenate([100 * run[2][f'{domain}_accuracy'] for run in runs.values()])
        ax.set(xlabel='Epoch', ylabel='Accuracy (%)', title=f'{domain.capitalize()} domain: seed=42',
               ylim=(5 * np.floor(values.min() / 5), min(100.5, 5 * np.ceil(values.max() / 5) + 0.5)))
        ax.grid(alpha=0.25)
        ax.legend()
        save(fig, output, f'{domain}_accuracy')
    data = adaptive[2]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for ax, key, title in zip(axes.flat, ('classification_loss', 'CORAL_loss', 'weighted_CORAL_loss', 'total_loss'),
                              ('Classification loss', 'Raw CORAL mean', 'Weighted CORAL mean', 'Total model loss')):
        for name in (fixed_key, adaptive_key):
            values = runs[name][2]
            ax.plot(values['epoch'], values[key], label=labels[name], color=colors[name])
        ax.set(xlabel='Epoch', ylabel='Loss', title=title, ylim=(0, None))
        ax.axvline(start_epoch - 0.5, color='#7c3aed', linestyle=':')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    save(fig, output, 'losses')
    fig, axes = plt.subplots(len(layers), 1, figsize=(10, 4 * len(layers)), layout='constrained', squeeze=False)
    for ax, layer in zip(axes.flat, layers):
        for name in (fixed_key, adaptive_key):
            values = runs[name][2]
            ax.plot(values['epoch'], values[f'CORAL_{layer}'], label=labels[name], color=colors[name])
        ax.axvline(start_epoch - 0.5, color='#7c3aed', linestyle=':')
        ax.set(xlabel='Epoch', ylabel=f'Raw CORAL: {layer}', ylim=(0, None), title=f'{layer} alignment distance')
        ax.grid(alpha=0.25)
        ax.legend()
    save(fig, output, 'coral_by_layer')
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    ax.plot(steps['global_step'], steps['lambda_used'], color=colors[adaptive_key], label='Lambda used by model')
    if len(layers) > 1:
        ax.plot(steps['global_step'], steps['lambda_used'] / len(layers), color='#64748b', label=f'Each layer weight: lambda/{len(layers)}')
    ax.plot(steps['global_step'], steps['gradient_CE_ema'] / np.maximum(steps['gradient_CORAL_ema'], 1e-12),
            color='#7c3aed', alpha=0.65, label='EMA norm ratio: g_CE / g_CORAL')
    ax.axhline(0.5, color=colors[fixed_key], linestyle='--', label='Fixed lambda=0.5')
    ax.axvline(start_step - 0.5, color='#7c3aed', linestyle=':', label='Start learning lambda')
    ax.set(xlabel='Training step', ylabel='Lambda (log scale)', yscale='log', title=f'Adaptive {display} coefficient')
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, output, 'lambda')
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout='constrained')
    ax = axes[0]
    for key, label, color in (('gradient_CE_ema', 'CE gradient (EMA)', '#2563eb'),
                               ('gradient_CORAL_ema', 'Raw CORAL gradient (EMA)', '#64748b'),
                               ('weighted_gradient_CORAL_ema', 'Lambda * CORAL gradient (EMA)', '#e86f25')):
        ax.plot(steps['global_step'], steps[key], label=label, color=color)
    ax.set(ylabel='Gradient norm (log scale)', yscale='log', title=f'Shared conv5 parameters; CORAL on {display}')
    ax.axvline(start_step - 0.5, color='#7c3aed', linestyle=':')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    ratio = steps['weighted_gradient_CORAL_ema'] / np.maximum(steps['gradient_CE_ema'], 1e-12)
    axes[1].plot(steps['global_step'], ratio, color='#e86f25')
    axes[1].axhline(1, linestyle='--', color='#64748b')
    axes[1].axvline(start_step - 0.5, color='#7c3aed', linestyle=':')
    axes[1].set(xlabel='Training step', ylabel='Weighted CORAL / CE', title='Gradient balance ratio', yscale='log')
    axes[1].grid(alpha=0.25)
    save(fig, output, 'gradient_balance')
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(steps['global_step'], steps['gradient_cosine'], alpha=0.55, label='Per batch')
    ax.plot(data['epoch'] * adaptive[0]['train_steps_per_epoch'], data['gradient_cosine'],
            color='#e86f25', label='Epoch mean')
    ax.axhline(0, color='#64748b', linestyle='--')
    ax.axvline(start_step - 0.5, color='#7c3aed', linestyle=':')
    ax.set(xlabel='Training step', ylabel='Gradient cosine', ylim=(-1, 1), title=f'CE and {display} CORAL mean gradient directions')
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, output, 'gradient_cosine')
    summaries = []
    shared_epoch = min(run[0]['epochs'] for run in runs.values())
    for name, (config, summary, values) in runs.items():
        index = selected[name]
        summaries.append({
            'setting': name, 'epochs': config['epochs'], 'final_source_accuracy': summary['final_source_accuracy'],
            'final_target_accuracy': summary['final_target_accuracy'],
            'final_target_correct': int(values['target_correct'][-1]),
            'target_samples': config['target_samples'],
            'vs_no_adaptation_pp': 100 * (summary['final_target_accuracy'] - runs['no_adaptation'][1]['final_target_accuracy']),
            'vs_fixed_group_pp': 100 * (summary['final_target_accuracy'] - runs[fixed_key][1]['final_target_accuracy']),
            'best_target_accuracy': summary['best_target_accuracy'], 'best_target_epoch': summary['best_target_epoch'],
            'selected_epoch': int(values['epoch'][index]),
            'selected_source_accuracy': float(values['source_accuracy'][index]),
            'selected_target_accuracy': float(values['target_accuracy'][index]),
            'selected_target_correct': int(values['target_correct'][index]),
            'selected_vs_no_adaptation_pp': 100 * (summary['best_target_accuracy'] - runs['no_adaptation'][1]['best_target_accuracy']),
            'selected_vs_fixed_group_pp': 100 * (summary['best_target_accuracy'] - runs[fixed_key][1]['best_target_accuracy']),
            'selected_lambda': float(values.get('lambda_used', np.full(config['epochs'], config['lambda_coral']))[index]),
            'shared_epoch': shared_epoch, 'shared_epoch_target_accuracy': float(values['target_accuracy'][shared_epoch - 1]),
            'initial_lambda': config['lambda_coral'], 'final_lambda': summary.get('final_lambda', config['lambda_coral']),
        })
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    bars = ax.bar([labels[row['setting']] for row in summaries],
                  [100 * row['final_target_accuracy'] for row in summaries], color=list(colors.values()))
    ax.bar_label(bars, labels=[f'{100 * row["final_target_accuracy"]:.2f}%\n'
                              f'{row["final_target_correct"]}/{row["target_samples"]}' for row in summaries], padding=4)
    ax.set(ylabel='Target accuracy (%)', ylim=(0, 100), title='Final epoch from each run: seed=42')
    save(fig, output, 'final_target_accuracy')
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    bars = ax.bar([labels[row['setting']] for row in summaries],
                  [100 * row['selected_target_accuracy'] for row in summaries], color=list(colors.values()))
    ax.bar_label(bars, labels=[f'{100 * row["selected_target_accuracy"]:.2f}%\n'
                              f'{row["selected_target_correct"]}/{row["target_samples"]} @ epoch {row["selected_epoch"]}'
                              for row in summaries], padding=4)
    ax.set(ylabel='Target accuracy (%)', ylim=(0, 100), title='Best epoch from each run: seed=42')
    save(fig, output, 'best_target_accuracy')
    provenance = {'input_runs': {name: {'directory': str(path), 'selected_epoch': selected[name] + 1,
                                      'input_files': ['config.json', 'environment.json', 'summary.json', 'metrics.csv']}
                                for name, path in directories.items()},
                  'selection_metric': 'best_target_accuracy', 'seed': SEED,
                  'original_controls_retrained': False, 'fixed_reference_retrained': False}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    result = adaptive[1]
    report = [
        f'# {display}：自适应 λ 实验', '',
        f'在 {display} 计算 CORAL，取所选层损失的均值，沿用 seed=42、{adaptive[0]["epochs"]} epochs、batch=128 和原 SGD 配置。'
        f'原始基线直接引用 `results/original/no_adaptation`；固定 {display} 来自 `{fixed_reference}`，'
        f'该固定运行共 {runs[fixed_key][0]["epochs"]} 轮。', '',
        f'前 {start_epoch - 1} 轮固定 λ=0.5，第 {start_epoch} 轮起学习 λ=exp(s)。'
        '在共享 conv5 weight/bias 上测量两项未加权梯度，'
        '以 decay=0.9 的 EMA 平滑，并使用独立 Adam（lr=0.01）最小化 '
        '`(λ·EMA(g_CORAL)−EMA(g_CE))²`。范数与模型使用的 λ 均 detach；λ 范围为 [0.0001,100]。', '',
        f'学习一个共享 λ，每层有效系数为 λ/{len(layers)}；固定阶段每层系数为 {0.5 / len(layers):.4f}。', '',
        '固定阶段维护 EMA，但不更新 λ 或 Adam 状态。以全部训练轮次中的最佳目标准确率保留本次结果，平分时选较早轮次。', '',
        f'| 设置 | 总轮数 | 最佳轮目标准确率 | 正确数 | 相对固定 {display} 最佳轮 | epoch | λ |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summaries:
        report.append(f'| {labels[row["setting"]]} | {row["epochs"]} | {100 * row["selected_target_accuracy"]:.2f}% | '
                      f'{row["selected_target_correct"]}/{row["target_samples"]} | {row["selected_vs_fixed_group_pp"]:+.2f} pp | '
                      f'{row["selected_epoch"]} | {row["selected_lambda"]:.4f} |')
    lambda_updates = result.get('lambda_updates', len(steps['global_step']) - start_step + 1)
    report.extend(['', f'完成 {result["epochs_completed"]} 轮、{result["steps_completed"]} 次模型更新、{lambda_updates} 次 λ 更新。'
                   f'λ：0.5 → {result["final_lambda"]:.6f}；实际使用范围 '
                   f'{result["minimum_lambda_used"]:.6f}～{result["maximum_lambda_used"]:.6f}。',
                   f'自适应阶段平均梯度平衡比（λ·EMA(g_CORAL)/EMA(g_CE)）：{float(np.mean(ratio[start_step - 1:])):.4f}；'
                   f'平均梯度方向余弦：{float(np.mean(steps["gradient_cosine"][start_step - 1:])):.4f}。', '',
                   f'最终轮补充：固定 {display} 为 {100 * runs[fixed_key][1]["final_target_accuracy"]:.2f}%，'
                   f'自适应 {display} 为 {100 * result["final_target_accuracy"]:.2f}%。', '',
                   f'共同轮次第 {shared_epoch} 轮：固定 {display} 为 '
                   f'{100 * runs[fixed_key][2]["target_accuracy"][shared_epoch - 1]:.2f}%，自适应为 '
                   f'{100 * data["target_accuracy"][shared_epoch - 1]:.2f}%。', '',
                   '逐批动态加权后再取轮均值，不能用 λ 均值乘原始 CORAL 均值替代。', '',
                   '![最佳轮比较](best_target_accuracy.png)', '![目标准确率](target_accuracy.png)', '![λ 变化](lambda.png)',
                   '![梯度平衡](gradient_balance.png)', '![损失](losses.png)', '![分层距离](coral_by_layer.png)', '',
                   '数据见 `summary.csv`、`../metrics.csv`、`../step_metrics.csv`；'
                   '数据来源见 `provenance.json`，运行环境见 `../environment.json`。', ''])
    if adaptive[0]['schema_version'] == 5:
        report.extend([f'本次保留第 {result["selected_epoch"]} 轮（{result["selected_phase"]} 阶段），'
                       f'该轮 λ 均值为 {result["selected_lambda"]:.6f}。'
                       f'仅在自适应阶段选取的最佳结果为第 {result["best_adaptive_target_epoch"]} 轮、'
                       f'{100 * result["best_adaptive_target_accuracy"]:.2f}%。', '',
                       '`../best_epoch.json`、`../best_epoch/metrics.csv` 和 `../best_epoch/step_metrics.csv` 保存最佳轮完整指标；'
                       '`../best_epoch/{source,target}_predictions.csv` 和 `../best_epoch/evaluation.npz` 保存逐图预测及 logits。', ''])
    (output / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, default=RESULTS_DIR / 'adaptive_lambda' / 'fc8')
    args = parser.parse_args()
    print(json.dumps(plot_experiment(args.run_dir), indent=2))


if __name__ == '__main__':
    main()
