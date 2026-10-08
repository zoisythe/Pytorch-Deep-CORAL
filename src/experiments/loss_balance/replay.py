"""Replay the fixed-weight stopping rule on existing logs, without training."""

import argparse
import csv
import json
from pathlib import Path

from src.common.paths import ORIGINAL_DIR, RESULTS_DIR
from src.training.loss_balance import (DEFAULT_EMA_DECAY, DEFAULT_MIN_EPOCHS,
                                      DEFAULT_PATIENCE, DEFAULT_TOLERANCE, LossBalanceStopper)


def replay(directory, *, tolerance=DEFAULT_TOLERANCE, patience=DEFAULT_PATIENCE,
           min_epochs=DEFAULT_MIN_EPOCHS, ema_decay=DEFAULT_EMA_DECAY):
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if config['lambda_coral'] != 0.5 or config.get('lambda_mode', 'fixed') != 'fixed':
        raise ValueError('Replay requires a fixed lambda=0.5 experiment')
    with (directory / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if len(rows) != config['epochs']:
        raise ValueError('Replay requires the complete recorded run')
    log_path = directory / 'train.log'
    if log_path.exists():
        log_rows = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()
                    if line.strip().startswith('{')]
        if log_rows != rows:
            raise ValueError(f'Training log differs from CSV metrics: {directory}')
        rows = log_rows
    stopper = LossBalanceStopper(tolerance, patience, min_epochs, ema_decay)
    observed = []
    for row in rows:
        observed.append({**row, **stopper.update(int(row['epoch']), row['classification_loss'], row['weighted_CORAL_loss'])})
        if observed[-1]['balance_stop']:
            break
    final, best = observed[-1], max(observed, key=lambda row: row['target_accuracy'])
    full_best = max(rows, key=lambda row: row['target_accuracy'])
    full_stopper = LossBalanceStopper(tolerance, patience, min_epochs, ema_decay)
    full_states = [full_stopper.update(int(row['epoch']), row['classification_loss'], row['weighted_CORAL_loss'])
                   for row in rows]
    end = full_states[-1]
    return observed, {
        'source_run': str(directory.resolve()), 'execution_mode': 'historical_log_replay',
        'log_metrics_checked': log_path.exists(),
        'lambda_coral': 0.5, 'recorded_epochs': len(rows),
        'triggered': bool(final['balance_stop']),
        'stop_epoch': int(final['epoch']) if final['balance_stop'] else None,
        'epochs_observed': len(observed), 'stop_balance_gap': final['balance_gap'],
        'stop_balance_raw_gap': final['balance_raw_gap'],
        'last_observed_target_accuracy': final['target_accuracy'],
        'best_observed_target_accuracy': best['target_accuracy'], 'best_observed_epoch': int(best['epoch']),
        'full_best_target_accuracy': full_best['target_accuracy'], 'full_best_epoch': int(full_best['epoch']),
        'best_accuracy_lost_pp': 100 * (full_best['target_accuracy'] - best['target_accuracy']),
        'recording_end_balance_gap': end['balance_gap'],
        'recording_end_classification_loss': rows[-1]['classification_loss'],
        'recording_end_weighted_CORAL_loss': rows[-1]['weighted_CORAL_loss'],
        'recording_end_ema_weighted_ratio': end['weighted_CORAL_loss_ema'] / max(end['classification_loss_ema'], 1e-12),
        'full_best_balance_gap': full_states[int(full_best['epoch']) - 1]['balance_gap'],
        'epochs_saved_within_recording': len(rows) - len(observed),
    }


def default_run_directories(*, all_fixed=False):
    paths = [ORIGINAL_DIR / 'fc8']
    if all_fixed:
        for suite in ('fc_layers', 'conv5'):
            paths.extend(sorted(path.parent for path in (RESULTS_DIR / suite).glob('*/config.json')
                                if (path.parent / 'metrics.csv').exists()))
    fixed_joint = RESULTS_DIR / 'fixed_lambda/conv5_fc8'
    if (fixed_joint / 'metrics.csv').exists():
        paths.append(fixed_joint)
    elif not all_fixed:
        paths.append(RESULTS_DIR / 'conv5/conv5_fc8')
    return paths


def write_records(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_rules(directories, output, tolerances, patiences, min_epochs, ema_decay):
    results, aggregates = [], []
    for tolerance in tolerances:
        for patience in patiences:
            rule_results = []
            for directory in directories:
                _, result = replay(directory, tolerance=tolerance, patience=patience,
                                   min_epochs=min_epochs, ema_decay=ema_decay)
                rule_results.append(result)
                results.append({'tolerance': tolerance, 'patience': patience, **result})
            aggregates.append({'tolerance': tolerance, 'patience': patience,
                               'runs': len(rule_results), 'triggered_runs': sum(r['triggered'] for r in rule_results),
                               'epochs_saved': sum(r['epochs_saved_within_recording'] for r in rule_results),
                               'max_best_accuracy_lost_pp': max(r['best_accuracy_lost_pp'] for r in rule_results)})
    write_records(output / 'threshold_grid.csv', results)
    write_records(output / 'rule_summary.csv', aggregates)
    report = ['# 固定权重早停：阈值与连续次数', '',
              f'回放 {len(directories)} 条完整固定 λ=0.5 记录；EMA decay={ema_decay}，最少 {min_epochs} 轮。'
              '只使用逐轮训练损失触发停止，事后比较停止前最佳准确率与完整记录最佳准确率。', '',
              '训练阈值上限为 25%，允许损失部分准确率；默认选择 25% 相对差距、连续 2 轮。'
              '当轮差距与 EMA 差距均须满足阈值；实际训练时停止决策只读取损失。'
              '25% 表示两项 EMA 的较小值至少为较大值的 75%。', '',
              '联合组的 50/100 轮记录属于同一层设置，不能把共有前缀当作独立证据。'
              '参数选择同时检查每条记录；未触发表示该记录内没有达到条件。', '',
              '| 阈值 | 连续轮数 | 触发记录数 | 记录内节省轮数 | 最大最佳准确率损失 |',
              '|---|---:|---:|---:|---:|']
    for row in aggregates:
        report.append(f'| {100 * row["tolerance"]:.0f}% | {row["patience"]} | '
                      f'{row["triggered_runs"]}/{row["runs"]} | {row["epochs_saved"]} | '
                      f'{row["max_best_accuracy_lost_pp"]:.2f} pp |')
    report.extend(['', '## 连续 2 轮：联合组独立 100 轮记录', '',
                   '| 阈值 | 停止轮 | 停止轮准确率 | 停止前最佳准确率 | 最佳轮 | 最佳准确率损失 |',
                   '|---|---:|---:|---:|---:|---:|'])
    for row in results:
        if row['patience'] == 2 and row['recorded_epochs'] == 100 and Path(row['source_run']).name == 'conv5_fc8':
            report.append(f'| {100 * row["tolerance"]:.0f}% | {row["stop_epoch"] or "未触发"} | '
                          f'{100 * row["last_observed_target_accuracy"]:.2f}% | '
                          f'{100 * row["best_observed_target_accuracy"]:.2f}% | {row["best_observed_epoch"]} | '
                          f'{row["best_accuracy_lost_pp"]:.2f} pp |')
    report.extend(['', '## 25% 与连续 2 轮：各记录', '',
                   '| 设置 | 记录轮数 | 停止轮 | 末期相对差距 | 停止前最佳准确率 |',
                   '|---|---:|---:|---:|---:|'])
    for row in results:
        if row['tolerance'] == DEFAULT_TOLERANCE and row['patience'] == DEFAULT_PATIENCE:
            report.append(f'| {Path(row["source_run"]).name} | {row["recorded_epochs"]} | '
                          f'{row["stop_epoch"] or "未触发"} | {100 * row["recording_end_balance_gap"]:.2f}% | '
                          f'{100 * row["best_observed_target_accuracy"]:.2f}% |')
    report.extend(['', '## 其他 50 轮实验的 train.log', '',
                   '对照训练日志与 CSV 的每一轮完整指标后回放。下表列出第 50 轮损失、两项 EMA 比例及相对差距；'
                   '只用该设置实际参与优化的 CORAL 层均值，监测但未参与优化的层不进入停止判断。', '',
                   '| 设置 | 第 50 轮 CE | 第 50 轮加权 CORAL | EMA 加权 CORAL/CE | EMA 相对差距 | 最佳轮 | 最佳轮相对差距 |',
                   '|---|---:|---:|---:|---:|---:|---:|'])
    for row in results:
        if row['recorded_epochs'] == 50 and row['tolerance'] == DEFAULT_TOLERANCE and row['patience'] == DEFAULT_PATIENCE:
            report.append(f'| {Path(row["source_run"]).name} | {row["recording_end_classification_loss"]:.6f} | '
                          f'{row["recording_end_weighted_CORAL_loss"]:.6f} | {row["recording_end_ema_weighted_ratio"]:.4f} | '
                          f'{100 * row["recording_end_balance_gap"]:.2f}% | {row["full_best_epoch"]} | '
                          f'{100 * row["full_best_balance_gap"]:.2f}% |')
    smoothing_results = []
    report.extend(['', '## 25% 与连续 2 轮：平滑速度', '',
                   '| EMA decay | 设置 | 记录轮数 | 停止轮 | 停止前最佳准确率 | 最佳准确率损失 |',
                   '|---|---|---:|---:|---:|---:|'])
    for decay in (0.0, 0.3, 0.5, 0.7, 0.8):
        for directory in directories:
            _, row = replay(directory, tolerance=DEFAULT_TOLERANCE, patience=DEFAULT_PATIENCE,
                            min_epochs=min_epochs, ema_decay=decay)
            smoothing_results.append({'ema_decay': decay, **row})
            report.append(f'| {decay} | {Path(row["source_run"]).name} | {row["recorded_epochs"]} | '
                          f'{row["stop_epoch"] or "未触发"} | {100 * row["best_observed_target_accuracy"]:.2f}% | '
                          f'{row["best_accuracy_lost_pp"]:.2f} pp |')
    write_records(output / 'smoothing_comparison.csv', smoothing_results)
    report.extend(['', '保留最大轮数，未达到平衡时依预算停止。', '',
                   '完整逐记录比较与来源路径见 `threshold_grid.csv`；规则汇总见 `rule_summary.csv`。', ''])
    (output / 'threshold_report.md').write_text('\n'.join(report), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dirs', type=Path, nargs='+', help='Defaults to fixed fc8 and conv5+fc8')
    parser.add_argument('--all_fixed', action='store_true', help='Include other fixed-weight layer settings as additional references')
    parser.add_argument('--output_dir', type=Path, default=RESULTS_DIR / 'loss_balance/replay')
    parser.add_argument('--balance_tolerance', type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument('--balance_patience', type=int, default=DEFAULT_PATIENCE)
    parser.add_argument('--balance_min_epochs', type=int, default=DEFAULT_MIN_EPOCHS)
    parser.add_argument('--balance_ema_decay', type=float, default=DEFAULT_EMA_DECAY)
    parser.add_argument('--compare_rules', action='store_true', help='Compare thresholds and consecutive counts on all selected records')
    parser.add_argument('--tolerances', type=float, nargs='+',
                        default=[0.05, 0.1, 0.15, 0.2, 0.25])
    parser.add_argument('--patiences', type=int, nargs='+', default=[1, 2, 3])
    args = parser.parse_args()
    args.run_dirs = args.run_dirs or default_run_directories(all_fixed=args.all_fixed)
    entries = []
    # Finish reading every source before writing the retrospective output.
    records = [replay(directory, tolerance=args.balance_tolerance, patience=args.balance_patience,
                      min_epochs=args.balance_min_epochs, ema_decay=args.balance_ema_decay)
               for directory in args.run_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (rows, result) in enumerate(records, start=1):
        filename = f'{index}_{Path(result["source_run"]).name}_balance.csv'
        write_records(args.output_dir / filename, rows)
        entries.append({**result, 'balance_metrics': filename})
    payload = {'execution_mode': 'historical_log_replay', 'training_performed': False,
               'rule': {'tolerance': args.balance_tolerance, 'patience': args.balance_patience,
                        'min_epochs': args.balance_min_epochs, 'ema_decay': args.balance_ema_decay,
                        'require_raw_and_ema_balance': True},
               'runs': entries}
    (args.output_dir / 'summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report = ['# 固定 λ=0.5：损失平衡早停回放', '',
              '本报告只回放已有完整日志，没有重新训练，也没有替换原始或历史实验。', '',
              f'EMA decay={args.balance_ema_decay}，相对差距阈值={args.balance_tolerance}，'
              f'最少 {args.balance_min_epochs} 轮，当轮和 EMA 差距均满足，连续 {args.balance_patience} 轮后停止。', '',
              '| 设置 | 已有轮数 | 模拟停止轮 | 停止/末次观察准确率 | 停止前最佳准确率 | 最佳轮 | 节省轮数 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for result in entries:
        stop = result['stop_epoch'] if result['triggered'] else '未触发'
        report.append(f'| {Path(result["source_run"]).name} | {result["recorded_epochs"]} | {stop} | '
                      f'{100 * result["last_observed_target_accuracy"]:.2f}% | '
                      f'{100 * result["best_observed_target_accuracy"]:.2f}% | {result["best_observed_epoch"]} | '
                      f'{result["epochs_saved_within_recording"]} |')
    report.extend(['', '停止判断只接收训练损失；准确率在判断结束后作为结果列出。', '',
                   '损失贡献相近不能定位最佳准确率，因此新训练会分别保存停止轮与最佳轮数据。', '',
                   '逐轮 EMA、差距和连续满足计数见 CSV；规则与来源路径见 `summary.json`。', ''])
    (args.output_dir / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    if args.compare_rules:
        compare_rules(args.run_dirs, args.output_dir, args.tolerances, args.patiences,
                      args.balance_min_epochs, args.balance_ema_decay)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
