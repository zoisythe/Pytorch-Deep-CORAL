"""Train one reproducible layer-placement experiment."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

# Required by deterministic CUDA matrix multiplication; set before CUDA starts.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.nn.functional as F

from src.common.experiment_protocol import SUITES, feature_protocol
from src.common.paths import (
    CONTROL_SETTINGS,
    ORIGINAL_DIR,
    ROOT,
    SEED,
    experiment_dir,
    suite_results_dir,
)
from src.core.coral import coral
from src.core.model import CORAL_LAYERS, Net
from src.data.data_loader import get_loader


def make_parser():
    parser = argparse.ArgumentParser(description='Train one Deep CORAL layer setting')
    parser.add_argument('--suite', choices=SUITES, default='original')
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay', type=float, default=5e-4, help='SGD weight decay')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lambda_coral', type=float, default=0.5)
    parser.add_argument('--source', choices=('amazon', 'dslr', 'webcam'), default='amazon')
    parser.add_argument('--target', choices=('amazon', 'dslr', 'webcam'), default='webcam')
    parser.add_argument('--coral_layers', nargs='+', choices=CORAL_LAYERS, default=['fc8'])
    parser.add_argument('--seed', type=int, choices=(SEED,), default=SEED)
    parser.add_argument('--results_dir', type=Path, help='Defaults to a separate directory for each suite')
    parser.add_argument('--num_workers', type=int, default=0 if os.name == 'nt' else 4,
                        help='DataLoader workers (Windows defaults to 0 to avoid CUDA DLL commit-memory exhaustion)')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace a run in the new experiment tree')
    return parser


def validate_args(args):
    if args.epochs < 1 or args.batch_size < 2 or args.num_workers < 0:
        raise ValueError('epochs >= 1, batch_size >= 2 and num_workers >= 0 are required')
    values = (args.lr, args.decay, args.momentum, args.lambda_coral)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Hyperparameters must be finite')
    if args.lr <= 0 or args.decay < 0 or not 0 <= args.momentum < 1 or args.lambda_coral < 0:
        raise ValueError('Invalid learning rate, weight decay, momentum or CORAL weight')
    if args.seed != SEED:
        raise ValueError('All experiments require seed=42')
    args.monitor_layers = SUITES[args.suite]['monitor_layers']
    if (args.lambda_coral > 0 and not args.coral_layers) or not set(args.coral_layers).issubset(args.monitor_layers):
        raise ValueError(f'{args.suite} supports these CORAL layers: {args.monitor_layers}')
    args.coral_layers = tuple(layer for layer in args.monitor_layers if layer in args.coral_layers)
    if args.lambda_coral == 0:
        # Separate optimization (none) from the suite's shared monitoring layers.
        args.coral_layers = ()
    if args.results_dir is None:
        args.results_dir = suite_results_dir(args.suite)
    if args.suite == 'original' and args.results_dir.resolve() != ORIGINAL_DIR.resolve():
        raise ValueError('Original controls must be archived in results/original')
    if args.suite != 'original' and args.results_dir.resolve().is_relative_to(ORIGINAL_DIR.resolve()):
        raise ValueError('Advanced results must be outside results/original')
    return args


def seed_everything(seed):
    if seed != SEED:
        raise ValueError('All RNG seeds must be 42')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def adaptation_loss(source_features, target_features, layers, weight, monitor_layers=None):
    monitor_layers = tuple(layers if monitor_layers is None else monitor_layers)
    if not monitor_layers or not set(layers).issubset(monitor_layers):
        raise ValueError('Monitoring layers must be nonempty and include optimization layers')
    if not layers and weight != 0:
        raise ValueError('A positive CORAL weight requires optimization layers')
    per_layer = {}
    for layer in monitor_layers:
        if weight > 0 and layer in layers:
            per_layer[layer] = coral(source_features[layer], target_features[layer])
        else:
            with torch.no_grad():
                per_layer[layer] = coral(source_features[layer].detach(), target_features[layer].detach())
    # The baseline logs the monitor mean for reference; it has no active layers.
    reduction_layers = layers or monitor_layers
    mean_loss = torch.stack([per_layer[layer] for layer in reduction_layers]).mean()
    return per_layer, mean_loss, weight * mean_loss


def train(model, optimizer, source_loader, target_loader, args, epoch=0):
    model.train()
    totals = {}
    steps = 0
    # zip creates each iterator once, and stops when the smaller domain ends.
    for (source_data, source_label), (target_data, _) in zip(source_loader, target_loader):
        source_data = source_data.to(args.device, non_blocking=True)
        source_label = source_label.to(args.device, non_blocking=True)
        target_data = target_data.to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out_source, source_features = model(source_data, return_features=True)
        if args.lambda_coral == 0:
            # Still forward the target to keep the Dropout RNG schedule identical.
            with torch.no_grad():
                _, target_features = model(target_data, return_features=True)
                per_layer, coral_mean, weighted = adaptation_loss(
                    source_features, target_features, args.coral_layers, 0.0, args.monitor_layers)
        else:
            _, target_features = model(target_data, return_features=True)
            per_layer, coral_mean, weighted = adaptation_loss(
                source_features, target_features, args.coral_layers, args.lambda_coral, args.monitor_layers)
        classification = F.cross_entropy(out_source, source_label)
        total = classification + weighted
        values = {
            'classification_loss': classification.item(),
            'CORAL_loss': coral_mean.item(),
            'weighted_CORAL_loss': weighted.item(),
            'total_loss': total.item(),
            **{f'CORAL_{layer}': loss.item() for layer, loss in per_layer.items()},
        }
        if not all(math.isfinite(v) for v in values.values()):
            raise FloatingPointError(f'Non-finite loss in epoch {epoch + 1}, step {steps + 1}')
        total.backward()
        optimizer.step()
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value
        steps += 1
    if not steps:
        raise ValueError('Both training domains need at least one complete batch')
    return {'train_steps': steps, **{k: v / steps for k, v in totals.items()}}


@torch.no_grad()
def evaluate(model, data_loader, dataset_name, args, epoch=0):
    model.eval()
    correct = 0
    count = 0
    for data, label in data_loader:
        data = data.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)
        output = model(data)
        correct += (output.argmax(dim=1) == label).sum().item()
        count += label.numel()
    if not count:
        raise ValueError(f'{dataset_name} evaluation domain is empty')
    return {f'{dataset_name}_accuracy': correct / count,
            f'{dataset_name}_correct': correct, f'{dataset_name}_samples': count}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def configure_device(args):
    args.device = torch.device('cuda' if not args.disable_cuda and torch.cuda.is_available() else 'cpu')
    if args.device.type == 'cuda' and args.num_workers == 0:
        # Small image transforms otherwise pay for CPU thread-pool synchronization.
        torch.set_num_threads(1)


def collect_environment(args):
    """Also used by the suite to reject stale completed runs before resuming."""
    return {'python': __import__('sys').version, 'torch': torch.__version__,
            'torchvision': __import__('torchvision').__version__, 'numpy': np.__version__,
            'cuda': torch.version.cuda, 'device': str(args.device),
            'gpu': torch.cuda.get_device_name(args.device) if args.device.type == 'cuda' else None,
            'deterministic_algorithms': True, 'tf32': False,
            'cpu_threads': torch.get_num_threads(),
            'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for folder in ('core', 'data', 'common', 'training')
                              for p in sorted((ROOT / 'src' / folder).glob('*.py'))}}


def run(args):
    args = validate_args(args)
    run_dir = experiment_dir(args.results_dir, args.coral_layers, args.seed, args.lambda_coral)
    if args.suite != 'original':
        from src.plotting.plot_accuracies_losses import validate_frozen_controls
        validate_frozen_controls(args)
        if run_dir.name in CONTROL_SETTINGS:
            print(f'Using frozen original control: {ORIGINAL_DIR / run_dir.name}', flush=True)
            return ORIGINAL_DIR / run_dir.name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{run_dir} already contains a run; choose a new --results_dir or --overwrite')
    seed_everything(args.seed)
    configure_device(args)
    loaders = [get_loader(domain, args.batch_size, train=is_train,
                          seed=SEED, num_workers=args.num_workers)
               for domain, is_train in [
                   (args.source, True), (args.target, True),
                   (args.source, False), (args.target, False)]]
    source_train, target_train, source_eval, target_eval = loaders
    if source_train.dataset.class_to_idx != target_train.dataset.class_to_idx:
        raise ValueError('Source and target class-to-index mappings differ')
    if min(len(source_train), len(target_train)) < 1:
        raise ValueError('Both training domains need at least one complete batch')

    model = Net(len(source_train.dataset.classes), pretrained=True,
                coral_layers=args.monitor_layers).to(args.device)
    optimizer = torch.optim.SGD([
        {'params': model.model.features.parameters()},
        {'params': model.model.classifier[:6].parameters()},
        {'params': model.model.classifier[6].parameters(), 'lr': 10 * args.lr},
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    config = {k: str(v) if isinstance(v, (Path, torch.device)) else v
              for k, v in vars(args).items() if k != 'overwrite'}
    config.update(schema_version=3, setting=run_dir.name,
                  feature_protocol=feature_protocol(args.monitor_layers), coral_reduction='mean',
                  effective_coral_weights={layer: args.lambda_coral / len(args.coral_layers)
                                           if layer in args.coral_layers else 0.0
                                           for layer in args.monitor_layers},
                  pretrained='IMAGENET1K_V1',
                  fc8_weight_std=0.005, fc8_bias=0.01, fc8_lr_multiplier=10,
                  train_steps_per_epoch=min(len(source_train), len(target_train)),
                  train_drop_last=True, class_to_idx=source_train.dataset.class_to_idx,
                  source_samples=len(source_train.dataset), target_samples=len(target_train.dataset),
                  source_initial_shuffle_seed=SEED, target_initial_shuffle_seed=SEED,
                  evaluation_loader_seeds=[SEED, SEED], worker_seed=SEED,
                  seed_policy='all_rngs_42',
                  original_controls=str(ORIGINAL_DIR) if args.suite != 'original' else None)
    environment = collect_environment(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Remove the completion marker first when an explicit rerun replaces a run.
    (run_dir / 'summary.json').unlink(missing_ok=True)
    write_json(run_dir / 'config.json', config)
    write_json(run_dir / 'environment.json', environment)
    if args.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
    rows = []
    started = time.monotonic()
    with (run_dir / 'train.log').open('w', encoding='utf-8') as log_file, \
            (run_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as metrics_file:
        writer = None
        try:
            for epoch in range(args.epochs):
                row = {'epoch': epoch + 1, **train(model, optimizer, source_train, target_train, args, epoch),
                       **evaluate(model, source_eval, 'source', args, epoch),
                       **evaluate(model, target_eval, 'target', args, epoch)}
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(metrics_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                metrics_file.flush()
                line = json.dumps(row)
                log_file.write(line + '\n')
                log_file.flush()
                print(f'{config["setting"]}: {line}', flush=True)
            summary = {'setting': config['setting'], 'epochs_completed': len(rows),
                       'final_source_accuracy': rows[-1]['source_accuracy'],
                       'final_target_accuracy': rows[-1]['target_accuracy'],
                       'best_target_accuracy': max(row['target_accuracy'] for row in rows),
                       'best_target_epoch': max(rows, key=lambda row: row['target_accuracy'])['epoch'],
                       'elapsed_seconds': time.monotonic() - started,
                       'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(args.device)
                       if args.device.type == 'cuda' else 0}
            write_json(run_dir / 'summary.json', summary)
        except BaseException:
            log_file.write(traceback.format_exc())
            raise
    return run_dir


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
