"""Run originals explicitly, or advanced settings using frozen original controls."""

import json
import subprocess
import sys

from src.common.experiment_protocol import SUITES, feature_protocol
from src.common.paths import CONTROL_SETTINGS, experiment_dir
from src.training.train import (
    collect_environment,
    configure_device,
    make_parser,
    validate_args,
)

SETTINGS = SUITES['fc_layers']['settings']


def check_resume(run_dir, args, layers, weight, environment):
    from src.plotting.plot_accuracies_losses import environment_differences, read_run
    config, _, _ = read_run(run_dir)
    expected = {key: getattr(args, key) for key in
                ('suite', 'source', 'target', 'epochs', 'batch_size', 'lr', 'decay',
                 'momentum', 'seed', 'num_workers', 'disable_cuda')}
    active_layers = list(layers) if weight > 0 else []
    expected.update(schema_version=3, coral_layers=active_layers, lambda_coral=weight,
                    monitor_layers=list(args.monitor_layers), coral_reduction='mean',
                    feature_protocol=feature_protocol(args.monitor_layers),
                    effective_coral_weights={layer: weight / len(active_layers)
                                             if layer in active_layers else 0.0
                                             for layer in args.monitor_layers})
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Cannot resume incompatible configuration or protocol: {run_dir}')
    recorded_environment = json.loads((run_dir / 'environment.json').read_text(encoding='utf-8'))
    if environment_differences(recorded_environment, environment):
        raise ValueError(f'Cannot resume a different runtime environment: {run_dir}')


def main(suite='fc_layers'):
    parser = make_parser()
    parser.description = 'Run a shared baseline and a CORAL placement suite'
    parser.set_defaults(suite=suite)
    parser.add_argument('--resume', action='store_true', help='Skip matching, fully completed runs')
    args = validate_args(parser.parse_args())
    if args.stop_on_loss_balance:
        parser.error('Use src.experiments.loss_balance.run for variable-length fixed-weight runs')
    if args.lambda_coral <= 0:
        parser.error('The adaptation weight must be positive for the experiment suite')
    if args.resume and args.overwrite:
        parser.error('--resume and --overwrite are mutually exclusive')
    configure_device(args)
    environment = collect_environment(args)
    if args.suite == 'original':
        runs = [(args.monitor_layers, 0.0), (('fc8',), args.lambda_coral)]
    else:
        from src.plotting.plot_accuracies_losses import validate_frozen_controls
        validate_frozen_controls(args)
        runs = [(layers, args.lambda_coral) for layers in SUITES[args.suite]['settings']
                if '_'.join(layers) not in CONTROL_SETTINGS]
    # Check every existing destination before starting any expensive training.
    completed = set()
    for layers, weight in runs:
        run_dir = experiment_dir(args.results_dir, layers, args.seed, weight)
        if (args.resume or (args.suite == 'original' and not args.overwrite)) and (run_dir / 'summary.json').exists():
            check_resume(run_dir, args, layers, weight, environment)
            completed.add(run_dir)
        elif run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f'{run_dir} contains existing or incomplete output; choose a fresh root or --overwrite')
    for layers, weight in runs:
        run_dir = experiment_dir(args.results_dir, layers, args.seed, weight)
        if run_dir in completed:
            print(f'Skipping completed run: {run_dir}', flush=True)
            continue
        command = [sys.executable, '-m', 'src.training.train', '--coral_layers', *layers, '--lambda_coral', str(weight),
                   '--results_dir', str(args.results_dir), '--suite', args.suite]
        for key in ('source', 'target', 'epochs', 'batch_size', 'lr', 'decay', 'momentum', 'seed', 'num_workers'):
            command.extend([f'--{key}', str(getattr(args, key))])
        if args.disable_cuda:
            command.append('--disable_cuda')
        if args.overwrite:
            command.append('--overwrite')
        print(f'Running: {run_dir}', flush=True)
        subprocess.run(command, check=True)
    subprocess.run([sys.executable, '-m', 'src.plotting.plot_accuracies_losses',
                    '--results_dir', str(args.results_dir), '--seed', str(args.seed),
                    '--suite', args.suite], check=True)


if __name__ == '__main__':
    main()
