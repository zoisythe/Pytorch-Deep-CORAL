"""Run the six locked ResNet-50 V2 settings, with a local shared-BN baseline."""

import json
import subprocess
import sys
from pathlib import Path

from src.common.experiment_protocol import RESNET_DEFAULTS, SUITES
from src.common.paths import experiment_dir
from src.plotting.plot_accuracies_losses import environment_differences
from src.plotting.plot_resnet50 import plot_experiments, read_resnet_run
from src.training.train import collect_environment, configure_device, make_parser, validate_args, write_json


def main():
    parser = make_parser()
    parser.description = __doc__
    parser.set_defaults(suite='resnet50', coral_layers=['logits'], **RESNET_DEFAULTS)
    parser.add_argument('--resume', action='store_true', help='Skip compatible validated completed runs')
    args = validate_args(parser.parse_args())
    if args.suite != 'resnet50' or args.lambda_coral != 0.5:
        parser.error('This entry point runs the complete ResNet-50 V2 suite with lambda=0.5')
    if args.resume and args.overwrite:
        parser.error('--resume and --overwrite are mutually exclusive')
    configure_device(args)
    environment = collect_environment(args)
    runs = [((), 0.0), *((layers, 0.5) for layers in SUITES['resnet50']['settings'])]
    completed = set()
    # Validate every destination before launching any expensive work.
    for layers, weight in runs:
        directory = experiment_dir(args.results_dir, layers, args.seed, weight, 'resnet50')
        if args.resume and (directory / 'summary.json').exists():
            config, _, _ = read_resnet_run(directory)
            recorded = json.loads((directory / 'environment.json').read_text(encoding='utf-8'))
            if (config['coral_layers'] != list(layers) or config['lambda_coral'] != weight
                    or config['disable_cuda'] != args.disable_cuda
                    or Path(config['results_dir']).resolve() != args.results_dir.resolve()
                    or environment_differences(recorded, environment)):
                raise ValueError(f'Cannot resume incompatible run: {directory}')
            completed.add(directory)
        elif directory.exists() and any(directory.iterdir()) and not args.overwrite:
            raise FileExistsError(f'{directory} has existing output; use --resume for completed results or --overwrite to rerun')
    for layers, weight in runs:
        directory = experiment_dir(args.results_dir, layers, args.seed, weight, 'resnet50')
        if directory in completed:
            print(f'Skipping validated completed run: {directory}', flush=True)
            continue
        command = [sys.executable, '-m', 'src.training.train', '--suite', 'resnet50',
                   '--coral_layers', *(layers or ('logits',)), '--lambda_coral', str(weight),
                   '--results_dir', str(args.results_dir)]
        for key in RESNET_DEFAULTS:
            command.extend([f'--{key}', str(getattr(args, key))])
        if args.disable_cuda:
            command.append('--disable_cuda')
        if args.overwrite:
            command.append('--overwrite')
        print(f'Running: {directory}', flush=True)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / 'failure.json', dict(returncode=error.returncode, command=command,
                       training_complete=False, automatic_configuration_change=False))
            raise
    print(f'Validated comparison: {plot_experiments(args.results_dir)}', flush=True)


if __name__ == '__main__':
    main()
