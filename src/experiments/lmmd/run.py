"""Run one matched AlexNet fc8 LMMD control, preserving the five CORAL runs."""

import argparse
import contextlib
import csv
import json
import math
import os
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn.functional as F

from src.core.coral import coral
from src.core.lmmd import lmmd
from src.core.model import Net
from src.core.subdomain_coral import class_conditional_coral
from src.experiments.lmmd.protocol import REPORTS_DIR, RUN_DIR, make_config, weights_for_epoch
from src.experiments.subdomain.run import macro_accuracy, make_loaders
from src.training.evaluation import evaluate_with_predictions, save_epoch
from src.training.train import collect_environment, configure_device, seed_everything, write_json


def train_epoch(model, optimizer, source_loader, target_loader, device, epoch):
    model.train()
    global_weight, lmmd_weight = weights_for_epoch(epoch)
    steps, class_rows, lmmd_class_rows = [], [], []
    for step, ((source_images, source_labels), (target_images, _)) in enumerate(zip(source_loader, target_loader), 1):
        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        source, target = model(source_images), model(target_images)
        probabilities = target.detach().softmax(1)
        with contextlib.nullcontext() if global_weight else torch.no_grad():
            global_loss = coral(source, target)
        with torch.no_grad():
            local_loss, records = class_conditional_coral(source, target, source_labels, probabilities)
        with contextlib.nullcontext() if lmmd_weight else torch.no_grad():
            lmmd_loss, lmmd_records, bandwidth = lmmd(source, target, source_labels, probabilities)
        classification = F.cross_entropy(source, source_labels)
        weighted_global, weighted_local = global_weight * global_loss, 0.0 * local_loss
        weighted_lmmd = lmmd_weight * lmmd_loss
        # Preserve the original warmup computation graph exactly; M is monitored only.
        adaptation = weighted_global + weighted_local if epoch <= 10 else weighted_lmmd
        total = classification + adaptation
        counts = torch.bincount(probabilities.argmax(1), minlength=31).cpu().tolist()
        valid_classes = sum(record['valid'] for record in records)
        lmmd_valid_classes = sum(record['valid'] for record in lmmd_records)
        row = dict(epoch=epoch, step=step, global_weight=global_weight, local_weight=0.0,
                   classification_loss=classification.item(), global_loss=global_loss.item(),
                   local_loss=local_loss.item(), weighted_global_loss=weighted_global.item(),
                   weighted_local_loss=weighted_local.item(), adaptation_loss=adaptation.item(),
                   total_loss=total.item(), valid_classes=valid_classes, valid_class_fraction=valid_classes / 31,
                   no_valid_classes=int(not valid_classes),
                   target_confidence=probabilities.max(1).values.mean().item(),
                   target_entropy=(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)).mean().item(),
                   target_prediction_classes=sum(count > 0 for count in counts),
                   **{f'skip_{reason}': sum(reason in record['skip_reason'].split('|') for record in records)
                      for reason in ('source_count', 'target_top1_count', 'target_mass', 'target_ess')},
                   **{f'target_prediction_count_{index:02d}': count for index, count in enumerate(counts)},
                   lmmd_weight=lmmd_weight, lmmd_loss=lmmd_loss.item(), weighted_lmmd_loss=weighted_lmmd.item(),
                   lmmd_valid_classes=lmmd_valid_classes, lmmd_no_shared_classes=int(not lmmd_valid_classes), **bandwidth)
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError(f'Non-finite LMMD training statistic: {epoch}/{step}')
        total.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError(f'Non-finite LMMD model gradient: {epoch}/{step}')
        optimizer.step()
        steps.append(row)
        class_rows.extend(dict(epoch=epoch, step=step, **record) for record in records)
        lmmd_class_rows.extend(dict(epoch=epoch, step=step, **record) for record in lmmd_records)
    if len(steps) != 6:
        raise ValueError(f'Expected six training steps, got {len(steps)}')
    means = {key: sum(row[key] for row in steps) / len(steps) for key in steps[0] if key not in ('epoch', 'step')}
    return dict(train_steps=len(steps), **means), steps, class_rows, lmmd_class_rows


def run(args, environment):
    seed_everything(42)
    loaders = make_loaders()
    config = make_config(loaders[0].dataset.class_to_idx, str(args.device))
    model = Net(31, pretrained=True, coral_layers=('fc8',)).to(args.device)
    groups = [{'params': model.model.features.parameters()}, {'params': model.model.classifier[:6].parameters()},
              {'params': model.model.classifier[6].parameters(), 'lr': 0.01}]
    optimizer = torch.optim.SGD(groups, lr=0.001, momentum=0.9, weight_decay=0.0005)
    parameter_ids = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != {id(p) for p in model.parameters()}:
        raise RuntimeError('Optimizer must cover each model parameter exactly once')
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ('summary.json', 'failure.json'):
        (RUN_DIR / filename).unlink(missing_ok=True)
    write_json(RUN_DIR / 'config.json', config)
    write_json(RUN_DIR / 'environment.json', environment)
    if args.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
    rows = []
    started = time.monotonic()
    with contextlib.ExitStack() as stack:
        streams = {name: stack.enter_context((RUN_DIR / f'{name}.csv').open('w', newline='', encoding='utf-8'))
                   for name in ('metrics', 'step_metrics', 'class_metrics', 'lmmd_class_metrics')}
        log_file = stack.enter_context((RUN_DIR / 'train.log').open('w', encoding='utf-8'))
        writers = {}
        try:
            for epoch in range(1, 51):
                training, steps, classes, lmmd_classes = train_epoch(model, optimizer, *loaders[:2], args.device, epoch)
                sm, sp = evaluate_with_predictions(model, loaders[2], 'source', args)
                tm, tp = evaluate_with_predictions(model, loaders[3], 'target', args)
                row = dict(epoch=epoch, **training, **sm, **tm,
                           source_macro_accuracy=macro_accuracy(*sp), target_macro_accuracy=macro_accuracy(*tp))
                for name, records in [('metrics', [row]), ('step_metrics', steps),
                                      ('class_metrics', classes), ('lmmd_class_metrics', lmmd_classes)]:
                    if name not in writers:
                        writers[name] = csv.DictWriter(streams[name], fieldnames=list(records[0]))
                        writers[name].writeheader()
                    writers[name].writerows(records)
                    streams[name].flush()
                rows.append(row)
                log_file.write(json.dumps(row) + '\n')
                log_file.flush()
                print(f'lmmd: epoch {epoch:02d}/50, target={row["target_accuracy"]:.4%}, '
                      f'CE={row["classification_loss"]:.5f}, G={row["global_loss"]:.5f}, '
                      f'M={row["lmmd_loss"]:.5f}, shared={row["lmmd_valid_classes"]:.2f}, '
                      f'weights=({row["global_weight"]:.2f},{row["lmmd_weight"]:.2f})', flush=True)
                if epoch == 10:
                    from src.experiments.lmmd.verify import verify_warmup
                    verify_warmup()
                    print('LMMD warmup matches global control (epoch/step/class records).', flush=True)
            save_epoch(RUN_DIR, rows[-1], (sp, tp), loaders[2:], name='final_epoch')
            metadata = json.loads((RUN_DIR / 'final_epoch.json').read_text(encoding='utf-8'))
            metadata.update(selection_metric='epoch_50_target_accuracy', phase='post_warmup',
                            target_accuracy_used_for_selection=False)
            write_json(RUN_DIR / 'final_epoch.json', metadata)
            best = max(rows, key=lambda row: row['target_accuracy'])
            write_json(RUN_DIR / 'summary.json', dict(setting='lmmd', training_complete=True, epochs_completed=50,
                       steps_completed=300, selected_epoch=50, selection_metric='epoch_50_target_accuracy',
                       final_source_accuracy=rows[-1]['source_accuracy'], final_target_accuracy=rows[-1]['target_accuracy'],
                       final_target_macro_accuracy=rows[-1]['target_macro_accuracy'],
                       best_target_accuracy=best['target_accuracy'], best_target_epoch=best['epoch'],
                       best_metric_descriptive_only=True, optimizer_coverage_verified=True, target_training_labels_used=False,
                       elapsed_seconds=time.monotonic() - started,
                       peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.type == 'cuda' else 0))
        except BaseException as error:
            log_file.write(traceback.format_exc())
            write_json(RUN_DIR / 'failure.json', dict(training_complete=False, error=repr(error), epochs_completed=len(rows),
                                                     automatic_configuration_change=False))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip_plots', action='store_true')
    parser.add_argument('--disable_cuda', action='store_true')
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error('--resume and --overwrite are mutually exclusive')
    args.num_workers = 0
    configure_device(args)
    torch.set_num_threads(1)
    if not args.disable_cuda and args.device.type != 'cuda':
        raise RuntimeError('CUDA is unavailable; cannot silently change the fixed protocol')
    environment = collect_environment(args)
    from src.experiments.subdomain.verify import read_run
    from src.experiments.subdomain.protocol import RUNS_DIR
    control_config, _, _ = read_run(RUNS_DIR / 'global')
    if json.loads((RUNS_DIR / 'global' / 'environment.json').read_text(encoding='utf-8')) != environment:
        raise ValueError('LMMD must use the same environment as the frozen global control')
    completed = False
    if args.resume and (RUN_DIR / 'summary.json').exists():
        from src.experiments.lmmd.verify import read_lmmd_run
        config, _, _ = read_lmmd_run()
        if config != make_config(control_config['class_to_idx'], str(args.device)):
            raise ValueError('Incompatible completed LMMD config')
        if json.loads((RUN_DIR / 'environment.json').read_text(encoding='utf-8')) != environment:
            raise ValueError('Incompatible completed LMMD environment')
        completed = True
        print(f'Skipping validated completed run: {RUN_DIR}', flush=True)
    elif RUN_DIR.exists() and any(RUN_DIR.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{RUN_DIR} already contains output; use --resume or --overwrite')
    if not completed:
        run(args, environment)
    if not args.skip_plots:
        from src.plotting.plot_lmmd import plot_experiments
        print(f'Validated LMMD comparison: {plot_experiments()}', flush=True)
    else:
        print(f'LMMD training output: {RUN_DIR}; report destination: {REPORTS_DIR}', flush=True)


if __name__ == '__main__':
    main()
