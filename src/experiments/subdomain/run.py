"""Run five locked AlexNet fc8 global/class-conditional CORAL experiments."""

import argparse
import contextlib
import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn.functional as F

from src.core.coral import coral
from src.core.model import Net
from src.core.subdomain_coral import class_conditional_coral
from src.data.data_loader import get_loader
from src.experiments.subdomain.protocol import (
    BATCH_SIZE, CLASS_COUNT, EPOCHS, REPORTS_DIR, RUNS_DIR, SETTINGS, STEPS_PER_EPOCH,
    WARMUP_EPOCHS, make_config, weights_for_epoch,
)
from src.training.evaluation import evaluate_with_predictions, save_epoch
from src.training.train import collect_environment, configure_device, seed_everything, write_json


def make_loaders():
    loaders = [get_loader(domain, BATCH_SIZE, train=train, seed=42, num_workers=0)
               for domain, train in [('amazon', True), ('webcam', True),
                                     ('amazon', False), ('webcam', False)]]
    source, target = loaders[:2]
    if (len(source.dataset), len(target.dataset), len(source.dataset.classes),
            min(len(source), len(target))) != (2817, 795, CLASS_COUNT, STEPS_PER_EPOCH):
        raise ValueError('This protocol requires Office-31 A→W, batch=128 and six paired steps')
    if any(loader.dataset.class_to_idx != source.dataset.class_to_idx for loader in loaders):
        raise ValueError('The four data loaders must share the class mapping')
    return loaders


def macro_accuracy(logits, labels):
    totals = torch.bincount(labels, minlength=CLASS_COUNT)
    correct = torch.bincount(labels[logits.argmax(1) == labels], minlength=CLASS_COUNT)
    return (correct[totals > 0] / totals[totals > 0]).mean().item()


def train_epoch(model, optimizer, source_loader, target_loader, device, setting, epoch):
    """Target labels are discarded at the loader boundary and never inspected."""
    model.train()
    global_weight, local_weight = weights_for_epoch(setting, epoch)
    steps, class_rows = [], []
    for step, ((source_images, source_labels), (target_images, _)) in enumerate(
            zip(source_loader, target_loader), 1):
        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        source = model(source_images)
        # Keep the target forward and its Dropout RNG consumption in all settings.
        with contextlib.nullcontext() if global_weight or local_weight else torch.no_grad():
            target = model(target_images)
        probabilities = target.detach().softmax(1)
        with contextlib.nullcontext() if global_weight else torch.no_grad():
            global_loss = coral(source, target)
        with contextlib.nullcontext() if local_weight else torch.no_grad():
            local_loss, records = class_conditional_coral(source, target, source_labels, probabilities)
        classification = F.cross_entropy(source, source_labels)
        weighted_global = global_weight * global_loss
        weighted_local = local_weight * local_loss
        adaptation = weighted_global + weighted_local
        total = classification + adaptation
        counts = torch.bincount(probabilities.argmax(1), minlength=CLASS_COUNT).cpu().tolist()
        valid_classes = sum(record['valid'] for record in records)
        row = dict(epoch=epoch, step=step, global_weight=global_weight, local_weight=local_weight,
                   classification_loss=classification.item(), global_loss=global_loss.item(),
                   local_loss=local_loss.item(), weighted_global_loss=weighted_global.item(),
                   weighted_local_loss=weighted_local.item(), adaptation_loss=adaptation.item(),
                   total_loss=total.item(), valid_classes=valid_classes,
                   valid_class_fraction=valid_classes / CLASS_COUNT, no_valid_classes=int(not valid_classes),
                   target_confidence=probabilities.max(1).values.mean().item(),
                   target_entropy=(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)).mean().item(),
                   target_prediction_classes=sum(count > 0 for count in counts),
                   **{f'skip_{reason}': sum(reason in record['skip_reason'].split('|') for record in records)
                      for reason in ('source_count', 'target_top1_count', 'target_mass', 'target_ess')},
                   **{f'target_prediction_count_{index:02d}': count for index, count in enumerate(counts)})
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError(f'Non-finite training statistic: {setting}, epoch {epoch}, step {step}')
        total.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise FloatingPointError(f'Non-finite model gradient: {setting}, epoch {epoch}, step {step}')
        optimizer.step()
        steps.append(row)
        class_rows.extend(dict(epoch=epoch, step=step, **record) for record in records)
    if len(steps) != STEPS_PER_EPOCH:
        raise ValueError(f'Expected six training steps, got {len(steps)}')
    means = {key: sum(row[key] for row in steps) / len(steps) for key in steps[0]
             if key not in ('epoch', 'step')}
    return dict(train_steps=len(steps), **means), steps, class_rows


def run_setting(setting, args, environment):
    directory = RUNS_DIR / setting
    if directory.exists() and any(directory.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{directory} already contains output; use --resume or --overwrite')
    seed_everything(42)
    loaders = make_loaders()
    config = make_config(setting, loaders[0].dataset.class_to_idx, str(args.device))
    model = Net(CLASS_COUNT, pretrained=True, coral_layers=('fc8',)).to(args.device)
    groups = [{'params': model.model.features.parameters()},
              {'params': model.model.classifier[:6].parameters()},
              {'params': model.model.classifier[6].parameters(), 'lr': 0.01}]
    optimizer = torch.optim.SGD(groups, lr=0.001, momentum=0.9, weight_decay=0.0005)
    parameter_ids = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != {id(p) for p in model.parameters()}:
        raise RuntimeError('Optimizer must cover each model parameter exactly once')
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'summary.json').unlink(missing_ok=True)
    (directory / 'failure.json').unlink(missing_ok=True)
    write_json(directory / 'config.json', config)
    write_json(directory / 'environment.json', environment)
    if args.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
    rows = []
    started = time.monotonic()
    with contextlib.ExitStack() as stack:
        streams = {name: stack.enter_context((directory / f'{name}.csv').open('w', newline='', encoding='utf-8'))
                   for name in ('metrics', 'step_metrics', 'class_metrics')}
        log_file = stack.enter_context((directory / 'train.log').open('w', encoding='utf-8'))
        writers = {}
        try:
            for epoch in range(1, EPOCHS + 1):
                training, steps, class_rows = train_epoch(model, optimizer, *loaders[:2], args.device, setting, epoch)
                source_metrics, source_predictions = evaluate_with_predictions(model, loaders[2], 'source', args)
                target_metrics, target_predictions = evaluate_with_predictions(model, loaders[3], 'target', args)
                row = dict(epoch=epoch, **training, **source_metrics, **target_metrics,
                           source_macro_accuracy=macro_accuracy(*source_predictions),
                           target_macro_accuracy=macro_accuracy(*target_predictions))
                for name, records in [('metrics', [row]), ('step_metrics', steps), ('class_metrics', class_rows)]:
                    if name not in writers:
                        writers[name] = csv.DictWriter(streams[name], fieldnames=list(records[0]))
                        writers[name].writeheader()
                    writers[name].writerows(records)
                    streams[name].flush()
                rows.append(row)
                line = json.dumps(row)
                log_file.write(line + '\n')
                log_file.flush()
                print(f'{setting}: epoch {epoch:02d}/{EPOCHS}, target={row["target_accuracy"]:.4%}, '
                      f'CE={row["classification_loss"]:.5f}, G={row["global_loss"]:.5f}, '
                      f'L={row["local_loss"]:.5f}, valid={row["valid_classes"]:.2f}, '
                      f'weights=({row["global_weight"]:.2f},{row["local_weight"]:.2f})', flush=True)
            save_epoch(directory, rows[-1], (source_predictions, target_predictions), loaders[2:], name='final_epoch')
            final_metadata = json.loads((directory / 'final_epoch.json').read_text(encoding='utf-8'))
            final_metadata.update(selection_metric='epoch_50_target_accuracy', phase='post_warmup',
                                  target_accuracy_used_for_selection=False)
            write_json(directory / 'final_epoch.json', final_metadata)
            best = max(rows, key=lambda row: row['target_accuracy'])
            summary = dict(setting=setting, training_complete=True, epochs_completed=len(rows),
                           steps_completed=sum(row['train_steps'] for row in rows), selected_epoch=EPOCHS,
                           selection_metric='epoch_50_target_accuracy',
                           final_source_accuracy=rows[-1]['source_accuracy'],
                           final_target_accuracy=rows[-1]['target_accuracy'],
                           final_target_macro_accuracy=rows[-1]['target_macro_accuracy'],
                           best_target_accuracy=best['target_accuracy'], best_target_epoch=best['epoch'],
                           best_metric_descriptive_only=True, optimizer_coverage_verified=True,
                           target_training_labels_used=False, elapsed_seconds=time.monotonic() - started,
                           peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(args.device)
                           if args.device.type == 'cuda' else 0)
            write_json(directory / 'summary.json', summary)
        except BaseException as error:
            log_file.write(traceback.format_exc())
            write_json(directory / 'failure.json', dict(training_complete=False, error=repr(error),
                                                       epochs_completed=len(rows), automatic_configuration_change=False))
            raise
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=SETTINGS, help='Run one setting; default runs all five serially')
    parser.add_argument('--resume', action='store_true', help='Skip compatible fully validated completed runs')
    parser.add_argument('--overwrite', action='store_true', help='Rerun only the selected subdomain destinations')
    parser.add_argument('--skip_plots', action='store_true', help='Generate the report separately after training')
    parser.add_argument('--disable_cuda', action='store_true', help='Explicit CPU execution, recorded in metadata')
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error('--resume and --overwrite are mutually exclusive')
    args.num_workers = 0
    configure_device(args)
    if not args.disable_cuda and args.device.type != 'cuda':
        raise RuntimeError('CUDA is unavailable; the locked CUDA protocol cannot silently fall back to CPU')
    environment = collect_environment(args)
    class_to_idx = make_loaders()[0].dataset.class_to_idx
    settings = (args.setting,) if args.setting else SETTINGS
    completed = set()
    # Preflight all destinations before launching any expensive run.
    for setting in settings:
        directory = RUNS_DIR / setting
        if args.resume and (directory / 'summary.json').exists():
            from src.experiments.subdomain.verify import read_run
            config, _, _ = read_run(directory)
            recorded = json.loads((directory / 'environment.json').read_text(encoding='utf-8'))
            if config != make_config(setting, class_to_idx, str(args.device)) or recorded != environment:
                raise ValueError(f'Cannot resume incompatible protocol/environment: {directory}')
            completed.add(setting)
        elif directory.exists() and any(directory.iterdir()) and not args.overwrite:
            raise FileExistsError(f'Existing incomplete output at {directory}; use --overwrite to rerun')
    for setting in settings:
        if setting in completed:
            print(f'Skipping validated completed run: {RUNS_DIR / setting}', flush=True)
            continue
        if args.setting:
            run_setting(setting, args, environment)
        else:
            command = [sys.executable, '-m', 'src.experiments.subdomain.run', '--setting', setting, '--skip_plots']
            if args.overwrite:
                command.append('--overwrite')
            if args.disable_cuda:
                command.append('--disable_cuda')
            print(f'Running: {RUNS_DIR / setting}', flush=True)
            subprocess.run(command, check=True)
    if not args.skip_plots and all((RUNS_DIR / setting / 'summary.json').exists() for setting in SETTINGS):
        from src.plotting.plot_subdomain import plot_experiments
        print(f'Validated comparison: {plot_experiments()}', flush=True)
    else:
        print(f'Training output: {RUNS_DIR}; report destination: {REPORTS_DIR}', flush=True)


if __name__ == '__main__':
    main()
