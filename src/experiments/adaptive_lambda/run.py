"""fc8 or conv5+fc8 CORAL with a delayed, gradient-balanced coefficient."""

import csv
import json
import math
import time
import traceback
from pathlib import Path

# Import the existing entry point first, including its deterministic CUDA setup.
from src.training.train import (collect_environment, configure_device,
                                make_parser, seed_everything, validate_args, write_json)

import torch
import torch.nn.functional as F

from src.common.experiment_protocol import feature_protocol
from src.common.paths import CONV_EXPERIMENTS_DIR, ORIGINAL_DIR, RESULTS_DIR, SEED
from src.core.coral import coral
from src.core.model import Net
from src.data.data_loader import get_loader
from src.experiments.adaptive_lambda.weight import GradientBalancedWeight, gradient_statistics
from src.training.evaluation import evaluate_with_predictions, save_best_epoch
from src.plotting.plot_accuracies_losses import assert_comparable, read_run, validate_frozen_controls


def learning_schedule(epochs, fraction, start_epoch=None):
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('adaptive_fraction must be in (0, 1]')
    if start_epoch is not None:
        if not 1 <= start_epoch <= epochs:
            raise ValueError('lambda_start_epoch must be within the training epochs')
        return start_epoch, epochs - start_epoch + 1
    adaptive_epochs = math.ceil(epochs * fraction)
    return epochs - adaptive_epochs + 1, adaptive_epochs


def train_epoch(model, optimizer, controller, loaders, args, epoch, step_writer, step_stream):
    model.train()
    totals, weights, step_rows = {}, [], []
    learn = epoch >= args.lambda_learning_start_epoch
    probe = (model.model.features[10].weight, model.model.features[10].bias)
    for step, ((source_data, labels), (target_data, _)) in enumerate(zip(*loaders), start=1):
        source_data = source_data.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        target_data = target_data.to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, source_features = model(source_data, return_features=True)
        _, target_features = model(target_data, return_features=True)
        classification = F.cross_entropy(logits, labels)
        per_layer = {layer: coral(source_features[layer], target_features[layer]) for layer in args.coral_layers}
        alignment = next(iter(per_layer.values())) if len(per_layer) == 1 else torch.stack(list(per_layer.values())).mean()
        weight = controller.value()
        weighted = weight * alignment
        total = classification + weighted
        ce_norm, coral_norm, cosine = gradient_statistics(classification, alignment, probe)
        diagnostics = controller.update(ce_norm, coral_norm, learn=learn)
        values = {
            'classification_loss': classification.item(), 'CORAL_loss': alignment.item(),
            **{f'CORAL_{layer}': loss.item() for layer, loss in per_layer.items()},
            'weighted_CORAL_loss': weighted.item(),
            'total_loss': total.item(), 'lambda_used': weight.item(), 'lambda_learning': int(learn),
            'gradient_CE': ce_norm.item(), 'gradient_CORAL': coral_norm.item(),
            'gradient_cosine': cosine.item(), **diagnostics,
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError(f'Non-finite metric in epoch {epoch}, step {step}')
        # The detached pre-update weight preserves the independent model objective.
        total.backward()
        optimizer.step()
        step_row = {'global_step': (epoch - 1) * args.steps_per_epoch + step,
                    'epoch': epoch, 'step_in_epoch': step, **values}
        step_rows.append(step_row)
        step_writer.writerow(step_row)
        step_stream.flush()
        weights.append(values['lambda_used'])
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
    if len(weights) != args.steps_per_epoch:
        raise ValueError('Unexpected number of paired training batches')
    return {'train_steps': len(weights), **{key: value / len(weights) for key, value in totals.items()},
            'lambda_min': min(weights), 'lambda_max': max(weights), 'lambda_end': controller.value().item()}, step_rows


def run(args):
    if args.stop_on_loss_balance:
        raise ValueError('Use src.experiments.loss_balance.run for fixed-weight stopping')
    args.results_dir = args.results_dir or RESULTS_DIR / ('fixed_lambda' if args.fixed_lambda else 'adaptive_lambda')
    args = validate_args(args)
    supported = {'fc_layers': ('fc8',), 'conv5_fc8': ('conv5', 'fc8')}
    if supported.get(args.suite) != args.coral_layers or args.lambda_coral <= 0:
        raise ValueError('Choose fc_layers with fc8, or conv5_fc8 with conv5 and fc8')
    if args.fixed_lambda:
        args.lambda_learning_start_epoch, args.adaptive_epochs = args.epochs + 1, 0
    else:
        args.lambda_learning_start_epoch, args.adaptive_epochs = learning_schedule(
            args.epochs, args.adaptive_fraction, args.lambda_start_epoch)
    setting = '_'.join(args.coral_layers)
    preparation_reference = ORIGINAL_DIR / 'fc8' if setting == 'fc8' else CONV_EXPERIMENTS_DIR / setting
    fixed_config, _, _ = read_run(preparation_reference)
    fixed_reference = (args.fixed_reference or preparation_reference).resolve()
    run_dir = (args.results_dir / setting).resolve()
    if run_dir.is_relative_to(ORIGINAL_DIR.resolve()):
        raise ValueError('Adaptive results cannot replace the original archive')
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{run_dir} already contains results; use --overwrite explicitly')
    validate_frozen_controls(args, allow_epoch_difference=True)
    seed_everything(SEED)
    configure_device(args)
    if run_dir in (preparation_reference.resolve(), fixed_reference.resolve()):
        raise ValueError('Adaptive results cannot replace the fixed-weight reference')
    args.monitor_layers = args.coral_layers
    loaders = [get_loader(domain, args.batch_size, train=training, seed=SEED,
                          num_workers=args.num_workers)
               for domain, training in ((args.source, True), (args.target, True),
                                        (args.source, False), (args.target, False))]
    source_train, target_train, source_eval, target_eval = loaders
    if source_train.dataset.class_to_idx != target_train.dataset.class_to_idx:
        raise ValueError('Source and target class mappings differ')
    args.steps_per_epoch = min(len(source_train), len(target_train))
    if args.steps_per_epoch < 1:
        raise ValueError('Both domains need a complete training batch')
    model = Net(len(source_train.dataset.classes), pretrained=True, coral_layers=args.monitor_layers).to(args.device)
    optimizer = torch.optim.SGD([
        {'params': model.model.features.parameters()},
        {'params': model.model.classifier[:6].parameters()},
        {'params': model.model.classifier[6].parameters(), 'lr': 10 * args.lr},
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    controller = GradientBalancedWeight(args.lambda_coral, args.lambda_lr, args.ema_decay,
                                        args.lambda_min, args.lambda_max, args.device)
    config = {key: str(value) if isinstance(value, (Path, torch.device)) else value
              for key, value in vars(args).items() if key not in ('overwrite', 'steps_per_epoch')}
    config.update(
        schema_version=5, suite='adaptive_lambda', setting=f'{setting}_{"fixed" if args.fixed_lambda else "adaptive"}_lambda',
        feature_protocol=feature_protocol(args.monitor_layers), coral_reduction='mean',
        effective_coral_weights={layer: f'lambda_used_per_step / {len(args.coral_layers)}' for layer in args.coral_layers},
        fixed_reference=str(fixed_reference),
        preparation_reference=str(preparation_reference),
        fixed_reference_epochs=args.epochs if args.fixed_reference else fixed_config['epochs'],
        comparison_common_epochs=args.epochs if args.fixed_reference else min(args.epochs, fixed_config['epochs']),
        lambda_schedule_mode='fixed' if args.fixed_lambda else 'explicit_start_epoch' if args.lambda_start_epoch is not None else 'final_fraction',
        lambda_mode='fixed' if args.fixed_lambda else 'gradient_norm_balance', lambda_parameterization='exp(log_weight)',
        lambda_update_objective='(lambda * EMA(g_CORAL) - EMA(g_CE))^2',
        lambda_optimizer={'name': 'Adam', 'lr': args.lambda_lr, 'betas': [0.9, 0.999],
                          'eps': 1e-8, 'weight_decay': 0},
        gradient_probe=['features[10].weight', 'features[10].bias'],
        gradient_statistics_detached=True, model_weight_detached=True,
        model_weight_timing='before_current_step_weight_update', ema_initialization='first_batch',
        fixed_epochs=args.lambda_learning_start_epoch - 1, ema_warmup_during_fixed_phase=True,
        selection_metric='best_target_accuracy', selection_scope='all_epochs', tie_break='earliest_epoch',
        pretrained='IMAGENET1K_V1', fc8_weight_std=0.005, fc8_bias=0.01, fc8_lr_multiplier=10,
        train_steps_per_epoch=args.steps_per_epoch, train_drop_last=True,
        class_to_idx=source_train.dataset.class_to_idx,
        source_samples=len(source_train.dataset), target_samples=len(target_train.dataset),
        source_initial_shuffle_seed=SEED, target_initial_shuffle_seed=SEED,
        evaluation_loader_seeds=[SEED, SEED], worker_seed=SEED, seed_policy='all_rngs_42',
        original_controls=str(ORIGINAL_DIR))
    assert_comparable(config, fixed_config, allow_epoch_difference=True)
    if fixed_config['lambda_coral'] != args.lambda_coral or fixed_config['coral_layers'] != list(args.coral_layers):
        raise ValueError('Fixed reference must use the same initial lambda and CORAL layers')
    environment = collect_environment(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'summary.json').unlink(missing_ok=True)
    (run_dir / 'best_epoch.json').unlink(missing_ok=True)
    write_json(run_dir / 'config.json', config)
    write_json(run_dir / 'environment.json', environment)
    if args.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
    fields = ['global_step', 'epoch', 'step_in_epoch', 'classification_loss', 'CORAL_loss',
              *(f'CORAL_{layer}' for layer in args.coral_layers),
              'weighted_CORAL_loss', 'total_loss', 'lambda_used', 'lambda_learning',
              'gradient_CE', 'gradient_CORAL', 'gradient_cosine', 'lambda_next',
              'gradient_CE_ema', 'gradient_CORAL_ema', 'weighted_gradient_CORAL_ema', 'weight_objective']
    rows, best = [], None
    started = time.monotonic()
    with (run_dir / 'train.log').open('w', encoding='utf-8') as log, \
            (run_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as metrics, \
            (run_dir / 'step_metrics.csv').open('w', newline='', encoding='utf-8') as steps:
        step_writer = csv.DictWriter(steps, fieldnames=fields)
        step_writer.writeheader()
        writer = None
        try:
            for epoch in range(1, args.epochs + 1):
                training, epoch_steps = train_epoch(model, optimizer, controller, loaders[:2],
                                                    args, epoch, step_writer, steps)
                source_metrics, source_predictions = evaluate_with_predictions(model, source_eval, 'source', args)
                target_metrics, target_predictions = evaluate_with_predictions(model, target_eval, 'target', args)
                row = {'epoch': epoch, **training, **source_metrics, **target_metrics}
                rows.append(row)
                if best is None or row['target_accuracy'] > best['target_accuracy']:
                    best = row
                    save_best_epoch(run_dir, row, epoch_steps, (source_predictions, target_predictions), loaders[2:])
                if writer is None:
                    writer = csv.DictWriter(metrics, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                metrics.flush()
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(f'Epoch {epoch:02d}/{args.epochs}: target={100 * row["target_accuracy"]:.2f}% '
                      f'source={100 * row["source_accuracy"]:.2f}% '
                      f'lambda_mean={row["lambda_used"]:.4f} lambda_next={row["lambda_end"]:.4f} '
                      f'phase={"adaptive" if row["lambda_learning"] else "fixed"} '
                      f'best={100 * best["target_accuracy"]:.2f}%@{best["epoch"]}', flush=True)
            adaptive_rows = rows[args.lambda_learning_start_epoch - 1:]
            best_adaptive = max(adaptive_rows, key=lambda row: row['target_accuracy']) if adaptive_rows else None
            summary = {
                'setting': config['setting'], 'epochs_completed': len(rows),
                'steps_completed': len(rows) * args.steps_per_epoch,
                'final_source_accuracy': rows[-1]['source_accuracy'],
                'final_target_accuracy': rows[-1]['target_accuracy'],
                'best_target_accuracy': best['target_accuracy'], 'best_target_epoch': best['epoch'],
                'selected_epoch': best['epoch'], 'selected_target_accuracy': best['target_accuracy'],
                'selected_source_accuracy': best['source_accuracy'], 'selected_lambda': best['lambda_used'],
                'selected_phase': 'adaptive' if best['lambda_learning'] else 'fixed',
                'best_epoch_data': 'best_epoch.json', 'selection_metric': 'best_target_accuracy',
                'best_adaptive_target_accuracy': best_adaptive['target_accuracy'] if best_adaptive else None,
                'best_adaptive_target_epoch': best_adaptive['epoch'] if best_adaptive else None,
                'lambda_learning_start_epoch': args.lambda_learning_start_epoch,
                'lambda_updates': args.adaptive_epochs * args.steps_per_epoch,
                'initial_lambda': args.lambda_coral, 'final_lambda': controller.value().item(),
                'minimum_lambda_used': min(row['lambda_min'] for row in rows),
                'maximum_lambda_used': max(row['lambda_max'] for row in rows),
                'elapsed_seconds': time.monotonic() - started,
                'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(args.device)
                if args.device.type == 'cuda' else 0, 'original_controls_retrained': False,
                'fixed_reference_retrained': False,
            }
            write_json(run_dir / 'summary.json', summary)
        except BaseException:
            log.write(traceback.format_exc())
            raise
    return run_dir


def main():
    parser = make_parser()
    parser.description = 'Train fc8 or conv5+fc8 CORAL with delayed gradient-balanced lambda'
    parser.set_defaults(suite='fc_layers')
    parser.add_argument('--lambda_lr', type=float, default=0.01)
    parser.add_argument('--ema_decay', type=float, default=0.9)
    parser.add_argument('--lambda_min', type=float, default=1e-4)
    parser.add_argument('--lambda_max', type=float, default=100.0)
    parser.add_argument('--adaptive_fraction', type=float, default=0.2,
                        help='Fraction of final epochs in which lambda is learned')
    parser.add_argument('--lambda_start_epoch', type=int,
                        help='Inclusive first epoch for learning lambda; overrides adaptive_fraction')
    parser.add_argument('--fixed_lambda', action='store_true', help='Keep lambda fixed throughout the run')
    parser.add_argument('--fixed_reference', type=Path, help='Fixed run for post-training comparison')
    parser.add_argument('--skip_plots', action='store_true', help='Generate comparison later when both runs are complete')
    args = parser.parse_args()
    try:
        directory = run(args)
        if args.fixed_lambda or args.skip_plots:
            print(f'Training data saved to {directory}', flush=True)
            return
        from src.plotting.plot_adaptive_lambda import plot_experiment
        plot_experiment(directory)
        print(f'Comparison saved to {directory / "comparison"}', flush=True)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
