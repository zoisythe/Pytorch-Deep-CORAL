"""Validate experiment metrics and generate PNG plots with a shared baseline."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.common.experiment_protocol import SUITES, feature_protocol
from src.common.paths import CONTROL_SETTINGS, ORIGINAL_DIR, SEED, setting_dir, suite_results_dir

COMMON_KEYS = ('schema_version', 'source', 'target', 'seed', 'epochs', 'batch_size',
               'lr', 'decay', 'momentum', 'num_workers', 'device', 'feature_location',
               'coral_reduction', 'covariance', 'pretrained', 'class_to_idx',
               'train_steps_per_epoch', 'source_samples', 'target_samples',
               'fc8_weight_std', 'fc8_bias', 'fc8_lr_multiplier', 'suite', 'monitor_layers',
               'feature_protocol', 'train_drop_last', 'source_initial_shuffle_seed',
               'target_initial_shuffle_seed', 'worker_seed', 'evaluation_loader_seeds', 'seed_policy')
SETTINGS = tuple(dict.fromkeys(('no_adaptation', *('_'.join(layers)
                   for suite in SUITES.values() for layers in suite['settings']))))


def monitored_layers(config):
    return config.get('monitor_layers', config['coral_layers'])


def validate_protocol(config):
    version = config.get('schema_version')
    if version == 1:
        if not config['coral_layers'] or not set(config['coral_layers']).issubset(SUITES['fc_layers']['monitor_layers']):
            raise ValueError('Historical schema 1 only supports fully connected layers')
        if config['feature_location'] != 'linear_output_before_relu' or config['covariance'] != 'unbiased_centered_gram':
            raise ValueError('Unknown historical CORAL protocol')
    elif version in (2, 3):
        suite = SUITES.get(config.get('suite'))
        if suite is None or config.get('monitor_layers') != list(suite['monitor_layers']):
            raise ValueError('Incorrect suite monitoring protocol')
        if config.get('feature_protocol') != feature_protocol(suite['monitor_layers']):
            raise ValueError('Incorrect feature/covariance protocol')
        active = config['coral_layers']
        if config['lambda_coral'] == 0:
            if active or config['setting'] != 'no_adaptation':
                raise ValueError('A baseline must have no optimization layers')
        elif tuple(active) not in suite['settings'] or config['setting'] != '_'.join(active):
            raise ValueError('Incorrect optimization layer setting')
        weights = {layer: config['lambda_coral'] / len(active) if layer in active else 0.0
                   for layer in suite['monitor_layers']}
        if config.get('effective_coral_weights') != weights:
            raise ValueError('Incorrect effective CORAL weights')
    else:
        raise ValueError(f'Unsupported config schema: {version}')
    if config['coral_reduction'] != 'mean' or not math.isfinite(config['lambda_coral']) or config['lambda_coral'] < 0:
        raise ValueError('Invalid CORAL reduction or weight')
    if config['seed'] != SEED:
        raise ValueError('Expected experiment seed=42')
    if version == 3 and (config.get('seed_policy') != 'all_rngs_42' or any(
            config.get(key) != SEED for key in ('source_initial_shuffle_seed', 'target_initial_shuffle_seed', 'worker_seed'))
            or config.get('evaluation_loader_seeds') != [SEED, SEED]):
        raise ValueError('New runs must use 42 for every RNG and loader')


def read_run(run_dir):
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    validate_protocol(config)
    with (run_dir / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows or summary['epochs_completed'] != len(rows):
        raise ValueError(f'Incomplete epoch metrics: {run_dir}')
    if config.get('stop_on_loss_balance'):
        from src.training.loss_balance import LossBalanceStopper
        if config['lambda_coral'] != 0.5 or config['suite'] == 'original' or not 1 <= len(rows) <= config['epochs']:
            raise ValueError(f'Invalid fixed-weight stopping protocol: {run_dir}')
        stopper = LossBalanceStopper(config['balance_tolerance'], config['balance_patience'],
                                     config['balance_min_epochs'], config['balance_ema_decay'])
        for index, row in enumerate(rows, start=1):
            expected = stopper.update(index, row['classification_loss'], row['weighted_CORAL_loss'])
            if any(not math.isclose(row[key], value, rel_tol=1e-12, abs_tol=1e-12)
                   for key, value in expected.items()) or expected['balance_stop'] and index != len(rows):
                raise ValueError(f'Incorrect loss-balance stopping history: {run_dir}')
        reason = 'loss_balance' if rows[-1]['balance_stop'] else 'max_epochs'
        if (summary.get('training_complete') is not True or summary.get('stop_epoch') != len(rows)
                or summary.get('stop_reason') != reason or summary.get('max_epochs') != config['epochs']
                or summary.get('stopped_early') != (len(rows) < config['epochs'])
                or reason == 'max_epochs' and len(rows) != config['epochs']):
            raise ValueError(f'Incomplete loss-balance experiment: {run_dir}')
    elif len(rows) != config['epochs']:
        raise ValueError(f'Incomplete epoch metrics: {run_dir}')
    for index, row in enumerate(rows, start=1):
        if row['epoch'] != index or not all(math.isfinite(v) for v in row.values()):
            raise ValueError(f'Invalid epoch or non-finite metric: {run_dir}')
        if row['train_steps'] != config['train_steps_per_epoch']:
            raise ValueError(f'Unexpected training step count: {run_dir}')
        for domain in ('source', 'target'):
            if row[f'{domain}_samples'] != config[f'{domain}_samples']:
                raise ValueError(f'Incomplete evaluation: {run_dir}')
            expected = row[f'{domain}_correct'] / row[f'{domain}_samples']
            if not 0 <= expected <= 1 or not math.isclose(row[f'{domain}_accuracy'], expected, abs_tol=1e-12):
                raise ValueError(f'Incorrect sample-weighted accuracy: {run_dir}')
        for layer in monitored_layers(config):
            if row[f'CORAL_{layer}'] < 0:
                raise ValueError(f'Negative CORAL distance: {run_dir}')
        reduction_layers = config['coral_layers'] or monitored_layers(config)
        mean_coral = sum(row[f'CORAL_{layer}'] for layer in reduction_layers) / len(reduction_layers)
        if not math.isclose(row['CORAL_loss'], mean_coral, rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError(f'Incorrect CORAL layer mean: {run_dir}')
        if not math.isclose(row['weighted_CORAL_loss'], config['lambda_coral'] * mean_coral,
                            rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError(f'Incorrect CORAL weight: {run_dir}')
        if not math.isclose(row['total_loss'], row['classification_loss'] + row['weighted_CORAL_loss'],
                            rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError(f'Incorrect total loss: {run_dir}')
    for domain in ('source', 'target'):
        if summary[f'final_{domain}_accuracy'] != rows[-1][f'{domain}_accuracy']:
            raise ValueError(f'Final summary differs from metrics: {run_dir}')
    best = max(rows, key=lambda row: row['target_accuracy'])
    if summary['best_target_accuracy'] != best['target_accuracy'] or summary['best_target_epoch'] != best['epoch']:
        raise ValueError(f'Best-epoch summary differs from metrics: {run_dir}')
    return config, summary, {key: np.array([row[key] for row in rows]) for key in rows[0]}


def assert_comparable(config, baseline_config, *, allow_epoch_difference=False):
    # Suite/schema/monitoring metadata describe different experiments, not training differences.
    metadata = {'schema_version', 'suite', 'monitor_layers', 'feature_protocol', 'feature_location', 'covariance'}
    if allow_epoch_difference:
        metadata.add('epochs')
    differences = [key for key in COMMON_KEYS if key not in metadata and config.get(key) != baseline_config.get(key)]
    if differences:
        raise ValueError(f'Run does not match its baseline: {differences}')


def comparison_provenance(runs):
    reference_config = runs['no_adaptation'][0]
    reference_env = json.loads((ORIGINAL_DIR / 'no_adaptation' / 'environment.json').read_text(encoding='utf-8'))
    reference_env.pop('source_sha256', None)
    sources = []
    for setting, (config, _, _) in runs.items():
        directory = setting_dir(suite_results_dir(config.get('suite', 'fc_layers')), setting)
        sources.append({'setting': setting, 'directory': str(directory),
                        'source_shuffle_seed': config['source_initial_shuffle_seed'],
                        'target_shuffle_seed': config['target_initial_shuffle_seed'],
                        'sampling_matches_original': config['target_initial_shuffle_seed'] == reference_config['target_initial_shuffle_seed'],
                        'schema_version': config['schema_version']})
    return {'original_directory': str(ORIGINAL_DIR), 'primary_source': 'metrics.csv',
            'historical_sampling_difference': any(not source['sampling_matches_original'] for source in sources),
            'comparison_scope': 'All RNGs and loaders use seed 42; single-seed descriptive comparison.',
            'reference_environment': reference_env, 'source_runs': sources}


def validate_frozen_controls(args, *, allow_epoch_difference=False):
    """Require a compatible completed archive before any advanced training."""
    from src.training.train import collect_environment, configure_device
    configure_device(args)
    environment = collect_environment(args)
    keys = ('source', 'target', 'epochs', 'batch_size', 'lr', 'decay', 'momentum', 'seed', 'num_workers', 'disable_cuda')
    if allow_epoch_difference:
        keys = tuple(key for key in keys if key != 'epochs')
    for setting in CONTROL_SETTINGS:
        config, _, _ = read_run(ORIGINAL_DIR / setting)
        differences = [key for key in keys if config.get(key) != getattr(args, key)]
        if differences or config.get('seed_policy') != 'all_rngs_42':
            raise ValueError(f'Frozen original {setting} is incompatible: {differences}; establish compatible originals explicitly')
        if setting == 'fc8' and args.lambda_coral > 0 and args.lambda_coral != config['lambda_coral']:
            raise ValueError('Adaptation coefficient must match the frozen original fc8 control')
        recorded = json.loads((ORIGINAL_DIR / setting / 'environment.json').read_text(encoding='utf-8'))
        if environment_differences(recorded, environment):
            raise ValueError(f'Frozen original {setting} uses a different runtime environment')


def environment_differences(environment, reference):
    """Compare runtime settings; historical code fingerprints are ignored."""
    def comparable(value):
        return {key: item for key, item in value.items()
                if key not in ('source_sha256', 'experiment_source_sha256')}

    current, recorded = comparable(environment), comparable(reference)
    return [key for key in current.keys() | recorded.keys() if current.get(key) != recorded.get(key)]


def validate_environments(directory, reference_directory):
    environment = json.loads((directory / 'environment.json').read_text(encoding='utf-8'))
    reference = json.loads((reference_directory / 'environment.json').read_text(encoding='utf-8'))
    differences = environment_differences(environment, reference)
    if differences:
        raise ValueError(f'Runtime environment differs from original controls: {directory}: {differences}')
    return environment


def save_plot(directory, name, title, ylabel, series, xlabel='Epoch'):
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for x, y, label, style in series:
        ax.plot(x, y, label=label, **style)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if ylabel == 'Accuracy':
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.savefig(directory / f'{name}.png', dpi=180)
    plt.close(fig)


def plot_run(run_dir, config, data, baseline):
    plots = run_dir / 'plots'
    epochs = data['epoch']
    setting = config['setting']
    is_baseline = setting == 'no_adaptation'
    accuracy_series = [(epochs, data[f'{domain}_accuracy'], f'{domain}: {setting}', {})
                       for domain in ('source', 'target')]
    if not is_baseline:
        accuracy_series.extend((epochs, baseline[f'{domain}_accuracy'], f'{domain}: no adaptation', {'linestyle': '--'})
                               for domain in ('source', 'target'))
    save_plot(plots, 'accuracies', f'Domain accuracy: {setting}', 'Accuracy', accuracy_series)
    classification_series = [(epochs, data['classification_loss'], setting, {})]
    if not is_baseline:
        classification_series.append((epochs, baseline['classification_loss'], 'no adaptation', {'linestyle': '--'}))
    save_plot(plots, 'classification_loss', f'Classification loss: {setting}', 'Loss', classification_series)
    save_plot(plots, 'losses_adapt', f'Training losses: {setting}', 'Loss',
              [(epochs, data[key], label, {}) for key, label in (
                  ('classification_loss', 'classification'), ('CORAL_loss', 'raw CORAL mean'),
                  ('weighted_CORAL_loss', 'weighted CORAL'), ('total_loss', 'total'))])
    layer_series = []
    for layer in monitored_layers(config):
        layer_series.append((epochs, data[f'CORAL_{layer}'], f'{layer}: {setting}', {}))
        if not is_baseline:
            layer_series.append((epochs, baseline[f'CORAL_{layer}'], f'{layer}: no adaptation', {'linestyle': '--'}))
    save_plot(plots, 'coral_by_layer', f'CORAL distance per layer: {setting}', 'CORAL distance', layer_series)
    save_plot(plots, 'coral_loss_no_adapt', f'Baseline CORAL distance for {setting}', 'CORAL distance',
              [(epochs, baseline[f'CORAL_{layer}'], layer, {}) for layer in monitored_layers(config)])


def plot_experiments(results_dir, seed, settings=None, suite='fc_layers'):
    if seed != SEED:
        raise ValueError('All experiments require seed=42')
    suite_settings = ('no_adaptation', *('_'.join(layers) for layers in SUITES[suite]['settings']))
    selected = list(suite_settings if settings is None else dict.fromkeys(('no_adaptation', *settings)))
    if not set(selected).issubset(suite_settings):
        raise ValueError(f'Settings do not belong to suite {suite}: {selected}')
    directories = {setting: setting_dir(results_dir, setting) for setting in selected}
    runs = {setting: read_run(directory) for setting, directory in directories.items()}
    baseline_config, baseline_summary, baseline_data = runs['no_adaptation']
    if baseline_config['lambda_coral'] != 0 or not set(SUITES[suite]['monitor_layers']).issubset(monitored_layers(baseline_config)):
        raise ValueError('Shared baseline must have lambda=0 and monitor the suite layers')
    if baseline_config.get('suite') != 'original':
        raise ValueError('Use the shared results/original archive')
    for setting, (config, summary, data) in runs.items():
        if config['setting'] != setting:
            raise ValueError(f'Setting name differs from its folder: {setting}')
        if config.get('schema_version') != 3:
            raise ValueError(f'Rerun {setting} with the all-seed-42 protocol before comparing')
        assert_comparable(config, baseline_config)
        validate_environments(directories[setting], ORIGINAL_DIR / 'no_adaptation')
        if setting != 'no_adaptation' and config['lambda_coral'] <= 0:
            raise ValueError(f'Adaptation requires a positive CORAL weight: {setting}')
    weights = {config['lambda_coral'] for setting, (config, _, _) in runs.items() if setting != 'no_adaptation'}
    if len(weights) > 1:
        raise ValueError('The comparison suite must use one shared adaptation weight')
    # Only write plots after the complete selected comparison has passed validation.
    for setting, (config, _, data) in runs.items():
        if suite == 'original' or setting not in CONTROL_SETTINGS:
            plot_run(directories[setting], config, data, baseline_data)

    comparison = results_dir / 'comparison'
    save_plot(comparison, 'target_accuracy', 'Target-domain accuracy by CORAL setting', 'Accuracy',
              [(data['epoch'], data['target_accuracy'], setting,
                {'linestyle': '--'} if setting == 'no_adaptation' else {})
               for setting, (_, _, data) in runs.items()])
    summaries = [{'setting': setting, 'final_source_accuracy': summary['final_source_accuracy'],
                  'final_target_accuracy': summary['final_target_accuracy'],
                  'improvement_pp': 100 * (summary['final_target_accuracy'] - baseline_summary['final_target_accuracy']),
                  'best_target_accuracy': summary['best_target_accuracy'],
                  'best_target_epoch': summary['best_target_epoch'],
                  'elapsed_seconds': summary['elapsed_seconds'],
                  'peak_gpu_memory_bytes': summary['peak_gpu_memory_bytes'],
                  **({'vs_fc8_pp': 100 * (summary['final_target_accuracy'] - runs['fc8'][1]['final_target_accuracy'])}
                     if 'fc8' in runs else {}),
                  'epochs_completed': summary['epochs_completed']}
                 for setting, (_, summary, _) in runs.items()]
    with (comparison / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    provenance = comparison_provenance(runs)
    for source in provenance['source_runs']:
        directory = directories[source['setting']]
        source['directory'] = str(directory)
        source['input_files'] = ['metrics.csv', 'config.json', 'summary.json', 'environment.json']
    (comparison / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    caveat = '原始组及全部进阶组均已按全 seed=42 协议重跑，配置和运行环境一致。'
    table = '\n'.join(f'| {row["setting"]} | {100 * row["final_source_accuracy"]:.2f}% | '
                      f'{100 * row["final_target_accuracy"]:.2f}% | {row["improvement_pp"]:+.2f} |'
                      for row in summaries)
    (comparison / 'report.md').write_text(
        f'# {suite} 实验比较\n\n对照组直接引用 `results/original/` 中的 no_adaptation 和 fc8，进阶绘图不重跑对照。\n\n'
        f'{caveat}\n\n| 设置 | 源域准确率 | 目标准确率 | 相对 no_adaptation（百分点） |\n'
        f'| --- | ---: | ---: | ---: |\n{table}\n\n'
        '主指标为最终轮次，目标标签只用于评估；来源见 `provenance.json`。\n',
        encoding='utf-8')
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    values = [row['final_target_accuracy'] for row in summaries]
    bars = ax.bar([row['setting'] for row in summaries], values,
                  color=['gray' if row['setting'] == 'no_adaptation' else 'steelblue' for row in summaries])
    ax.bar_label(bars, labels=[f'{100 * value:.2f}%' for value in values], padding=3)
    ax.set(title=f'Final target accuracy (epoch {baseline_config["epochs"]})', ylabel='Accuracy', ylim=(0, 1))
    ax.tick_params(axis='x', labelrotation=30)
    ax.axhline(baseline_summary['final_target_accuracy'], color='gray', linestyle='--', label='no adaptation')
    ax.legend()
    fig.savefig(comparison / 'final_target_accuracy.png', dpi=180)
    plt.close(fig)
    best = max(summaries, key=lambda row: row['final_target_accuracy'])
    print(f'Best final setting: {best["setting"]}; target accuracy={best["final_target_accuracy"]:.4%}; '
          f'improvement={best["improvement_pp"]:+.2f} percentage points', flush=True)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, default='fc_layers')
    parser.add_argument('--results_dir', type=Path)
    parser.add_argument('--seed', type=int, choices=(SEED,), default=SEED)
    parser.add_argument('--settings', nargs='+', choices=SETTINGS, help='Plot a subset together with its baseline')
    args = parser.parse_args()
    plot_experiments(args.results_dir or suite_results_dir(args.suite), args.seed, args.settings, args.suite)


if __name__ == '__main__':
    main()
