"""Run all eight extensions with a bounded number of independent processes."""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import subprocess
import sys
from collections import deque

from src.common.experiment_protocol import SUITES
from src.common.paths import CONTROL_SETTINGS, ORIGINAL_DIR, RESULTS_DIR, suite_results_dir
from src.plotting.plot_accuracies_losses import assert_comparable, read_run, validate_environments


class TrainingFailed(RuntimeError):
    def __init__(self, setting, result):
        self.details = {'setting': setting, 'exit_code': result.returncode,
                        'stderr_tail': result.stderr[-2000:]}
        super().__init__(f'{setting} failed with exit code {result.returncode}: {result.stderr[-2000:]}')


def control_hashes():
    return {f'{setting}/{name}': hashlib.sha256((ORIGINAL_DIR / setting / name).read_bytes()).hexdigest()
            for setting in CONTROL_SETTINGS
            for name in ('config.json', 'environment.json', 'metrics.csv', 'summary.json', 'train.log')}


def launch(suite, layers, overwrite):
    command = [sys.executable, '-m', 'src.training.train', '--suite', suite, '--coral_layers', *layers]
    if overwrite:
        command.append('--overwrite')
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise TrainingFailed('_'.join(layers), result)
    return '_'.join(layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=1,
                        help='Concurrent independent training processes (default serial)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Keep compatible completed extensions; --overwrite reruns only unfinished/old outputs')
    args = parser.parse_args()
    for setting in CONTROL_SETTINGS:
        read_run(ORIGINAL_DIR / setting)
    before = control_hashes()
    baseline_config = read_run(ORIGINAL_DIR / 'no_adaptation')[0]
    pending = deque()
    skipped = []
    interrupted = []
    for suite in ('fc_layers', 'conv5_fc8'):
        for layers in SUITES[suite]['settings']:
            setting = '_'.join(layers)
            if setting in CONTROL_SETTINGS:
                continue
            directory = suite_results_dir(suite) / setting
            config_path = directory / 'config.json'
            config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
            if args.resume and config.get('schema_version') == 3 and (directory / 'summary.json').exists():
                config, _, _ = read_run(directory)
                assert_comparable(config, baseline_config)
                validate_environments(directory, ORIGINAL_DIR / 'no_adaptation')
                if config['suite'] != suite or tuple(config['coral_layers']) != layers:
                    raise ValueError(f'Cannot resume an incompatible setting: {directory}')
                skipped.append(setting)
                print(f'Skipping completed extension: {setting}', flush=True)
                continue
            metrics_path = directory / 'metrics.csv'
            if config.get('schema_version') == 3 and metrics_path.exists():
                lines = metrics_path.read_bytes().splitlines(keepends=True)
                if lines and not lines[-1].endswith(b'\n'):
                    lines.pop()
                interrupted.append({'setting': setting, 'line_count': len(lines),
                                    'completed_epochs': max(0, len(lines) - 1),
                                    'metrics_prefix_sha256': hashlib.sha256(b''.join(lines)).hexdigest()})
            pending.append((suite, layers))
    destinations = [suite_results_dir(suite) / '_'.join(layers) for suite, layers in pending]
    if not args.overwrite and any(path.exists() and any(path.iterdir()) for path in destinations):
        parser.error('Existing extensions require --overwrite; use individual suite --resume to skip completed runs')
    execution_path = RESULTS_DIR / 'execution.json'
    previous = json.loads(execution_path.read_text(encoding='utf-8')) if execution_path.exists() else None
    manifest = {'original_training_jobs': 1, 'advanced_training_jobs': args.jobs,
                'advanced_processes': 'Independent Python processes, each with all RNG seeds=42',
                'timing_scope': 'Single-run elapsed time under shared resources; not an algorithm speed comparison',
                'original_controls_before': before, 'completed_settings': list(skipped),
                'reused_completed_settings': skipped, 'interrupted_attempts': interrupted,
                'previous_execution': previous, 'retry_prefix_checks': []}

    def persist():
        execution_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    persist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        active = {}
        try:
            while pending or active:
                while pending and len(active) < args.jobs:
                    suite, layers = pending.popleft()
                    future = executor.submit(launch, suite, layers, args.overwrite)
                    active[future] = (suite, layers)
                    print(f'Running extension: {suite}/{"_".join(layers)}', flush=True)
                done, _ = concurrent.futures.wait(active, timeout=30,
                                                  return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    setting = future.result()
                    del active[future]
                    manifest['completed_settings'].append(setting)
                    for attempt in interrupted:
                        if attempt['setting'] == setting:
                            directory = suite_results_dir(next(suite for suite in ('fc_layers', 'conv5_fc8')
                                                               if (suite_results_dir(suite) / setting).exists())) / setting
                            prefix = b''.join((directory / 'metrics.csv').read_bytes().splitlines(keepends=True)[:attempt['line_count']])
                            manifest['retry_prefix_checks'].append({**attempt,
                                'retry_metrics_prefix_sha256': hashlib.sha256(prefix).hexdigest(),
                                'exact_prefix_match': hashlib.sha256(prefix).hexdigest() == attempt['metrics_prefix_sha256']})
                    persist()
                    print(f'Completed extension: {setting} (50 epochs)', flush=True)
                progress = {}
                for suite, layers in active.values():
                    setting = '_'.join(layers)
                    directory = suite_results_dir(suite) / setting
                    config_path = directory / 'config.json'
                    metrics_path = directory / 'metrics.csv'
                    if config_path.exists() and metrics_path.exists():
                        try:
                            config = json.loads(config_path.read_text(encoding='utf-8'))
                            if config.get('schema_version') == 3:
                                with metrics_path.open(newline='', encoding='utf-8') as stream:
                                    progress[setting] = sum(1 for _ in csv.DictReader(stream))
                        except (OSError, json.JSONDecodeError):
                            # Progress is optional while a child creates its output.
                            pass
                print(f'Progress: {json.dumps(progress)}; completed={len(manifest["completed_settings"])}/8', flush=True)
        except BaseException as exc:
            manifest['failure'] = exc.details if isinstance(exc, TrainingFailed) else {'message': str(exc)}
            manifest['original_controls_after_failure'] = control_hashes()
            persist()
            print(f'Extension failure: {manifest["failure"]}', flush=True)
            for future in active:
                future.cancel()
            raise
    after = control_hashes()
    if after != before:
        raise ValueError('Advanced training modified the frozen original archive')
    manifest['original_controls_after'] = after
    manifest['original_controls_unchanged'] = True
    persist()
    for suite in ('fc_layers', 'conv5_fc8'):
        subprocess.run([sys.executable, '-m', 'src.plotting.plot_accuracies_losses', '--suite', suite], check=True)
    subprocess.run([sys.executable, '-m', 'src.plotting.plot_all_experiments'], check=True)


if __name__ == '__main__':
    main()
