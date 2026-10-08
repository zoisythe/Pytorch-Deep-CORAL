"""Full-domain evaluation and prediction artifacts without model checkpoints."""

import csv
from pathlib import Path

import numpy as np
import torch

from src.common.paths import ROOT


@torch.no_grad()
def evaluate_with_predictions(model, loader, domain, args):
    model.eval()
    outputs, labels = [], []
    for images, targets in loader:
        outputs.append(model(images.to(args.device, non_blocking=True)).cpu())
        labels.append(targets.cpu())
    logits, labels = torch.cat(outputs), torch.cat(labels)
    predictions = logits.argmax(dim=1)
    count = labels.numel()
    correct = (predictions == labels).sum().item()
    return {f'{domain}_accuracy': correct / count, f'{domain}_correct': correct,
            f'{domain}_samples': count}, (logits, labels)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_epoch(directory, row, evaluations, loaders, *, name='best_epoch', steps=None):
    from src.training.train import write_json

    output = directory / name
    output.mkdir(exist_ok=True)
    write_csv(output / 'metrics.csv', [row])
    files = [f'{name}/metrics.csv']
    if steps is not None:
        write_csv(output / 'step_metrics.csv', steps)
        files.append(f'{name}/step_metrics.csv')
    arrays = {'classes': np.array(loaders[0].dataset.classes)}
    for domain, (logits, labels), loader in zip(('source', 'target'), evaluations, loaders):
        confidences, predictions = logits.softmax(dim=1).max(dim=1)
        rows = [{'sample_index': index,
                 'image': Path(loader.dataset.samples[index][0]).relative_to(ROOT).as_posix(),
                 'label_index': label.item(), 'label': loader.dataset.classes[label.item()],
                 'prediction_index': prediction.item(), 'prediction': loader.dataset.classes[prediction.item()],
                 'confidence': confidence.item(), 'correct': int(label.item() == prediction.item())}
                for index, (label, prediction, confidence) in enumerate(zip(labels, predictions, confidences))]
        write_csv(output / f'{domain}_predictions.csv', rows)
        arrays.update({f'{domain}_logits': logits.numpy(), f'{domain}_labels': labels.numpy()})
        files.append(f'{name}/{domain}_predictions.csv')
    np.savez_compressed(output / 'evaluation.npz', **arrays)
    files.append(f'{name}/evaluation.npz')
    write_json(directory / f'{name}.json', {
        'epoch': row['epoch'],
        'selection_metric': 'target_accuracy' if name == 'best_epoch' else 'training_stop',
        'tie_break': 'earliest_epoch' if name == 'best_epoch' else None,
        'phase': 'adaptive' if row.get('lambda_learning') else 'fixed',
        'metrics': row, 'files': files,
    })


def save_best_epoch(directory, row, steps, evaluations, loaders):
    save_epoch(directory, row, evaluations, loaders, steps=steps)
