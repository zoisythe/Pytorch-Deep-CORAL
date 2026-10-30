"""Train the one authorized DSAN-style ResNet-50 V2 LMMD experiment."""

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
from torchvision import datasets, transforms

from src.common.paths import DATASETS_DIR
from src.core.lmmd import lmmd
from src.experiments.resnet50_lmmd.model import DSANV2
from src.experiments.resnet50_lmmd.protocol import (
    EPOCHS,
    PATIENCE,
    RUN_DIR,
    STEPS,
    make_config,
    schedule,
)
from src.training.evaluation import evaluate_with_predictions, save_epoch
from src.training.train import (
    collect_environment,
    configure_device,
    seed_everything,
    write_json,
)


def make_loaders():
    train_transform = transforms.Compose([transforms.Resize([256, 256]), transforms.RandomCrop(224),
                                         transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    eval_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    loaders, image_datasets = [], []
    for domain, training in [('amazon', True), ('webcam', True), ('amazon', False), ('webcam', False)]:
        dataset = datasets.ImageFolder(DATASETS_DIR / domain / 'images',
                                       transform=train_transform if training else eval_transform)
        image_datasets.append(dataset)
        loaders.append(torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=training,
                       drop_last=training, num_workers=0, pin_memory=True,
                       generator=torch.Generator().manual_seed(42)))
    if len({tuple(dataset.class_to_idx.items()) for dataset in image_datasets}) != 1:
        raise ValueError('Domain class mappings differ')
    if tuple(len(dataset) for dataset in image_datasets) != (2817, 795, 2817, 795):
        raise ValueError('This experiment requires Office-31 Amazon→Webcam')
    if (len(loaders[0]), len(loaders[1])) != (88, 24):
        raise ValueError('Unexpected batch counts')
    return loaders


def train_epoch(model, optimizer, source_loader, target_loader, device, epoch):
    model.train()
    backbone_lr, head_lr, weight = schedule(epoch)
    for group, lr in zip(optimizer.param_groups, (backbone_lr, head_lr, head_lr)):
        group['lr'] = lr
    source_iterator, target_iterator = iter(source_loader), iter(target_loader)
    before = model.batchnorm_counts()
    steps, classes = [], []
    for step in range(1, len(source_loader)):
        source_images, source_labels = next(source_iterator)
        target_images, _ = next(target_iterator)
        if step % len(target_loader) == 0:
            target_iterator = iter(target_loader)
        source_images, source_labels = source_images.to(device), source_labels.to(device)
        target_images = target_images.to(device)
        optimizer.zero_grad(set_to_none=True)
        source_logits, source = model(source_images, return_features=True)
        target_logits, target = model(target_images, return_features=True)
        probabilities = target_logits.detach().softmax(1)
        local, records, bandwidth = lmmd(source, target, source_labels, probabilities)
        classification = F.cross_entropy(source_logits, source_labels)
        total = classification + weight * local
        if epoch == 1 and step == 1:
            if source.shape != (32, 256) or target.shape != (32, 256):
                raise ValueError('LMMD must align 256-dimensional bottleneck features')
            gradients = torch.autograd.grad(local, (source, target), retain_graph=True)
            if any(not torch.isfinite(g).all() or g.norm() == 0 for g in gradients):
                raise RuntimeError('Invalid bottleneck LMMD gradient')
        total.backward()
        # Check finiteness without clipping or changing the SGD update.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        optimizer.step()
        row = dict(epoch=epoch, step=step, backbone_lr=backbone_lr, head_lr=head_lr, lmmd_weight=weight,
                   classification_loss=classification.item(), lmmd_loss=local.item(),
                   weighted_lmmd_loss=(weight * local).item(), total_loss=total.item(),
                   shared_classes=sum(r['valid'] for r in records), gradient_norm=norm.item(),
                   target_confidence=probabilities.max(1).values.mean().item(),
                   target_entropy=(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)).mean().item(),
                   **bandwidth)
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError(f'Non-finite statistic at {epoch}/{step}')
        steps.append(row)
        classes.extend(dict(epoch=epoch, step=step, **record) for record in records)
    after = model.batchnorm_counts()
    if any(after[name] != count + 2 * STEPS for name, count in before.items()):
        raise RuntimeError('BatchNorm did not update twice per step')
    if len(steps) != STEPS:
        raise ValueError('Expected 87 source-driven updates')
    means = {key: sum(row[key] for row in steps) / STEPS for key in steps[0] if key not in ('epoch', 'step')}
    return dict(train_steps=STEPS, **means), steps, classes


def macro_accuracy(logits, labels):
    predictions = logits.argmax(1)
    return sum((predictions[labels == c] == c).float().mean().item() for c in range(31)) / 31


def save_checkpoint(model, optimizer, epoch, name):
    destination = RUN_DIR / name
    temporary = destination.with_suffix('.tmp')
    torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'pretrained_weights': 'ResNet50_Weights.IMAGENET1K_V2'}, temporary)
    temporary.replace(destination)


def run(args):
    seed_everything(42)
    loaders = make_loaders()
    model = DSANV2().to(args.device)
    optimizer = torch.optim.SGD([{'params': model.backbone.parameters(), 'lr': 0.001},
                                {'params': model.bottleneck.parameters(), 'lr': 0.01},
                                {'params': model.classifier.parameters(), 'lr': 0.01}],
                               momentum=0.9, weight_decay=0.0005)
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()}:
        raise RuntimeError('Optimizer must cover every parameter exactly once')
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RUN_DIR / 'config.json', make_config(loaders[0].dataset.class_to_idx, str(args.device)))
    write_json(RUN_DIR / 'environment.json', collect_environment(args))
    torch.cuda.reset_peak_memory_stats(args.device)
    rows, best_correct, stale = [], -1, 0
    started = time.monotonic()
    with contextlib.ExitStack() as stack:
        streams = {name: stack.enter_context((RUN_DIR / f'{name}.csv').open('w', newline='', encoding='utf-8'))
                   for name in ('metrics', 'step_metrics', 'class_metrics')}
        log_file = stack.enter_context((RUN_DIR / 'train.log').open('w', encoding='utf-8'))
        writers = {}
        try:
            for epoch in range(1, EPOCHS + 1):
                epoch_started = time.monotonic()
                training, steps, classes = train_epoch(model, optimizer, loaders[0], loaders[1], args.device, epoch)
                bn = model.batchnorm_snapshot()
                sm, sp = evaluate_with_predictions(model, loaders[2], 'source', args)
                tm, tp = evaluate_with_predictions(model, loaders[3], 'target', args)
                model.assert_batchnorm_unchanged(bn)
                improved = tm['target_correct'] > best_correct
                stale = 0 if improved else stale + 1
                row = dict(epoch=epoch, **training, **sm, **tm,
                           source_macro_accuracy=macro_accuracy(*sp), target_macro_accuracy=macro_accuracy(*tp),
                           improved=int(improved), stale_epochs=stale,
                           epoch_seconds=time.monotonic() - epoch_started, elapsed_seconds=time.monotonic() - started)
                if improved:
                    best_correct = tm['target_correct']
                    best_epoch = epoch
                    save_epoch(RUN_DIR, row, (sp, tp), loaders[2:], name='best_epoch')
                    save_checkpoint(model, optimizer, epoch, 'best_model.pth')
                for name, records in [('metrics', [row]), ('step_metrics', steps), ('class_metrics', classes)]:
                    if name not in writers:
                        writers[name] = csv.DictWriter(streams[name], fieldnames=list(records[0]))
                        writers[name].writeheader()
                    writers[name].writerows(records)
                    streams[name].flush()
                rows.append(row)
                log_file.write(json.dumps(row) + '\n')
                log_file.flush()
                write_json(RUN_DIR / 'progress.json', {'epoch': epoch, 'best_epoch': best_epoch,
                           'best_target_accuracy': best_correct / 795, 'latest_target_accuracy': tm['target_accuracy'],
                           'stale_epochs': stale, 'elapsed_seconds': time.monotonic() - started})
                print(f'epoch {epoch:03d}/{EPOCHS}, target={tm["target_accuracy"]:.4%}, '
                      f'best={best_correct / 795:.4%}@{best_epoch}, stale={stale}/{PATIENCE}, '
                      f'CE={row["classification_loss"]:.5f}, M={row["lmmd_loss"]:.5f}, '
                      f'lambda={row["lmmd_weight"]:.5f}, time={row["epoch_seconds"]:.1f}s', flush=True)
                if stale >= PATIENCE:
                    break
            save_epoch(RUN_DIR, rows[-1], (sp, tp), loaders[2:], name='stop_epoch')
            save_checkpoint(model, optimizer, epoch, 'stop_model.pth')
            write_json(RUN_DIR / 'summary.json', {'training_complete': True, 'setting': 'resnet50_v2_lmmd_author_code',
                       'epochs_completed': epoch, 'steps_completed': epoch * STEPS,
                       'stop_reason': 'target_accuracy_patience' if stale >= PATIENCE else 'max_epochs',
                       'selected_epoch': best_epoch, 'selected_target_accuracy': best_correct / 795,
                       'selection_metric': 'best_target_accuracy_earliest_tie', 'stop_epoch': epoch,
                       'stop_target_accuracy': rows[-1]['target_accuracy'],
                       'stop_source_accuracy': rows[-1]['source_accuracy'], 'target_labels_in_loss': False,
                       'target_labels_in_selection_and_stopping': True, 'optimizer_coverage_verified': True,
                       'bottleneck_gradient_verified': True, 'batchnorm_training_verified': True,
                       'batchnorm_evaluation_verified': True, 'elapsed_seconds': time.monotonic() - started,
                       'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(args.device)})
        except BaseException as error:
            log_file.write(traceback.format_exc())
            write_json(RUN_DIR / 'failure.json', {'training_complete': False, 'error': repr(error),
                       'epochs_completed': len(rows), 'automatic_configuration_change': False})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report_only', action='store_true', help='Validate/report the completed run without training')
    args = parser.parse_args()
    args.disable_cuda, args.num_workers = False, 0
    configure_device(args)
    if args.device.type != 'cuda':
        raise RuntimeError('This locked run requires CUDA')
    if not args.report_only:
        if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
            raise FileExistsError(f'Existing output at {RUN_DIR}; incomplete experiments are not silently overwritten')
        run(args)
    from src.experiments.resnet50_lmmd.report import build_report
    print(f'Validated report: {build_report()}', flush=True)


if __name__ == '__main__':
    main()
