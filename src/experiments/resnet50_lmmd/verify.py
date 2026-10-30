"""Independent checks of the completed protocol, records and saved predictions."""

import csv
import importlib.util
import json
import math
import os
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from src.core.lmmd import lmmd
from src.experiments.resnet50_lmmd.protocol import (
    EPOCHS,
    PATIENCE,
    REPORT_DIR,
    RUN_DIR,
    STEPS,
    make_config,
    schedule,
)
from src.training.train import seed_everything, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message, tolerance=1e-6):
    require(math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance), message)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def validate_author_formula():
    """Compare 256-dimensional loss and gradients with unmodified author methods."""
    reference = REPORT_DIR / 'reference' / 'lmmd.py'
    spec = importlib.util.spec_from_file_location('dsan_author_lmmd_reference', reference)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load archived author LMMD reference')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    author = module.LMMD_loss(class_num=31)
    generator = torch.Generator().manual_seed(42)
    source = torch.randn(32, 256, generator=generator, requires_grad=True)
    target = torch.randn(32, 256, generator=generator, requires_grad=True)
    labels = torch.arange(32) % 31
    probabilities = torch.randn(32, 31, generator=generator).softmax(1).requires_grad_()
    actual, _, _ = lmmd(source, target, labels, probabilities)
    author_weights = [torch.from_numpy(value) for value in author.cal_weight(labels, probabilities)]
    kernel = author.guassian_kernel(source, target)
    expected = (author_weights[0] * kernel[:32, :32] + author_weights[1] * kernel[32:, 32:]
                - 2 * author_weights[2] * kernel[:32, 32:]).sum()
    actual_grad = torch.autograd.grad(actual, (source, target), retain_graph=True)
    reference_grad = torch.autograd.grad(expected, (source, target))
    error = max((left - right).abs().max().item() for left, right in zip(actual_grad, reference_grad))
    close(actual.item(), expected.item(), 'Author LMMD loss differs', 1e-6)
    require(error <= 1e-6 and probabilities.grad is None, 'Author LMMD gradient/detach differs')
    result = {'passed': True, 'dimensions': [32, 256], 'dtype': 'float32', 'loss': actual.item(),
                  'reference_loss': expected.item(), 'absolute_loss_error': abs(actual.item() - expected.item()),
                  'maximum_feature_gradient_error': error, 'target_probability_gradient': probabilities.grad,
                  'reference': 'unmodified_author_cal_weight_and_guassian_kernel_on_cpu',
                  'source_url': 'https://raw.githubusercontent.com/easezyc/deep-transfer-learning/master/UDA/pytorch1.0/DSAN/lmmd.py'}
    write_json(REPORT_DIR / 'author_validation.json', result)
    return result


def validate_predictions(name, row):
    directory = RUN_DIR / name
    arrays = np.load(directory / 'evaluation.npz', allow_pickle=False)
    for domain, count in [('source', 2817), ('target', 795)]:
        logits, labels = arrays[f'{domain}_logits'], arrays[f'{domain}_labels']
        require(logits.shape == (count, 31) and labels.shape == (count,), f'{name}/{domain} shape')
        require(np.isfinite(logits).all(), f'{name}/{domain} non-finite predictions')
        predictions = logits.argmax(1)
        correct = int((predictions == labels).sum())
        close(correct / count, row[f'{domain}_accuracy'], f'{name}/{domain} accuracy')
        records = read_csv(directory / f'{domain}_predictions.csv')
        require(len(records) == count, f'{name}/{domain} missing predictions')
        for index, record in enumerate(records):
            require(int(record['sample_index']) == index and int(record['label_index']) == labels[index]
                    and int(record['prediction_index']) == predictions[index]
                    and int(record['correct']) == int(labels[index] == predictions[index]),
                    f'{name}/{domain} prediction {index} differs')
    saved_row = read_csv(directory / 'metrics.csv')[0]
    for key in row:
        close(saved_row[key], row[key], f'{name} metric {key}')
    require(int(json.loads((RUN_DIR / f'{name}.json').read_text())['epoch']) == int(row['epoch']), 'Epoch metadata')


def validate_checkpoint_predictions():
    """Reload the selected model and independently evaluate all target images."""
    from src.experiments.resnet50_lmmd.model import DSANV2
    from src.experiments.resnet50_lmmd.run import make_loaders
    from src.training.evaluation import evaluate_with_predictions

    seed_everything(42)
    torch.set_num_threads(1)
    state = torch.load(RUN_DIR / 'best_model.pth', map_location='cpu', weights_only=True)
    model = DSANV2()
    model.load_state_dict(state['model'], strict=True)
    del state
    device = torch.device('cuda')
    model.to(device)
    before = model.batchnorm_snapshot()
    metrics, (logits, labels) = evaluate_with_predictions(model, make_loaders()[3], 'target', SimpleNamespace(device=device))
    model.assert_batchnorm_unchanged(before)
    recorded = np.load(RUN_DIR / 'best_epoch' / 'evaluation.npz', allow_pickle=False)
    require(np.array_equal(labels.numpy(), recorded['target_labels']), 'Reloaded model label order differs')
    error = float(np.abs(logits.numpy() - recorded['target_logits']).max())
    require(error <= 1e-6, 'Reloaded model logits differ')
    return {'passed': True, 'target_samples': metrics['target_samples'], 'maximum_logit_error': error,
            'target_correct': metrics['target_correct']}


def validate_run():
    summary = json.loads((RUN_DIR / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((RUN_DIR / 'config.json').read_text(encoding='utf-8'))
    require(config == make_config(config['class_to_idx'], config['device']), 'Locked protocol configuration differs')
    rows = read_csv(RUN_DIR / 'metrics.csv')
    steps = read_csv(RUN_DIR / 'step_metrics.csv')
    require(summary['training_complete'] and len(rows) == summary['epochs_completed'], 'Incomplete run')
    require(config['pretrained_weights'] == 'ResNet50_Weights.IMAGENET1K_V2' and config['bottleneck'] == 256,
            'Wrong model configuration')
    require(config['target_labels_in_selection_and_stopping'] and not config['target_labels_in_loss'],
            'Wrong target-label disclosure')
    require(len(steps) == len(rows) * STEPS == summary['steps_completed'], 'Step counts differ')
    best_correct, best_epoch, stale = -1, 0, 0
    for epoch, row in enumerate(rows, 1):
        require(int(row['epoch']) == epoch and int(row['train_steps']) == STEPS, 'Epoch order/steps')
        blr, hlr, weight = schedule(epoch)
        epoch_steps = steps[(epoch - 1) * STEPS:epoch * STEPS]
        for step, item in enumerate(epoch_steps, 1):
            require(int(item['epoch']) == epoch and int(item['step']) == step, 'Step order')
            require(all(math.isfinite(float(value)) for value in item.values()), 'Non-finite step record')
            for key, expected in [('backbone_lr', blr), ('head_lr', hlr), ('lmmd_weight', weight)]:
                close(item[key], expected, f'Schedule {key}', 1e-12)
            close(item['weighted_lmmd_loss'], weight * float(item['lmmd_loss']), 'Weighted loss', 1e-5)
            close(item['total_loss'], float(item['classification_loss']) + float(item['weighted_lmmd_loss']),
                  'Total loss', 1e-5)
        for key in steps[0]:
            if key not in ('epoch', 'step'):
                close(row[key], sum(float(item[key]) for item in epoch_steps) / STEPS, f'Epoch mean {key}', 1e-10)
        for domain, count in [('source', 2817), ('target', 795)]:
            require(int(row[f'{domain}_samples']) == count, 'Evaluation coverage')
            close(row[f'{domain}_accuracy'], int(row[f'{domain}_correct']) / count, 'Epoch accuracy', 1e-12)
        correct = int(row['target_correct'])
        improved = correct > best_correct
        stale = 0 if improved else stale + 1
        if improved:
            best_correct, best_epoch = correct, epoch
        require(int(row['improved']) == int(improved) and int(row['stale_epochs']) == stale, 'Patience replay')
        require(epoch == len(rows) or stale < PATIENCE, 'Training continued past earliest stop')
    require((stale >= PATIENCE and summary['stop_reason'] == 'target_accuracy_patience')
            or (len(rows) == EPOCHS and summary['stop_reason'] == 'max_epochs'), 'Wrong stopping point')
    require(summary['selected_epoch'] == best_epoch and summary['stop_epoch'] == len(rows), 'Selection differs')
    close(summary['selected_target_accuracy'], best_correct / 795, 'Selected accuracy')
    close(summary['stop_target_accuracy'], rows[-1]['target_accuracy'], 'Stop accuracy')
    with (RUN_DIR / 'train.log').open(encoding='utf-8') as stream:
        logged = [json.loads(line) for line in stream]
    require(len(logged) == len(rows), 'Log length')
    for row, log_row in zip(rows, logged):
        require(set(row) == set(log_row), 'Log fields')
        for key, value in row.items():
            close(value, log_row[key], f'Log/CSV {key}', 1e-12)
    class_count = 0
    with (RUN_DIR / 'class_metrics.csv').open(encoding='utf-8', newline='') as stream:
        iterator = iter(csv.DictReader(stream))
        for item in steps:
            records = [next(iterator) for _ in range(31)]
            class_count += 31
            for index, record in enumerate(records):
                require(int(record['class_index']) == index and record['epoch'] == item['epoch']
                        and record['step'] == item['step'], 'Class record order')
                valid = int(record['source_count']) > 0 and int(record['target_top1_count']) > 0
                require(int(record['valid']) == int(valid), 'Shared-class rule differs')
                require(bool(record['lmmd_loss']) == valid, 'Invalid class must have an empty loss')
            for key in ('source_count', 'target_top1_count'):
                require(sum(int(record[key]) for record in records) == 32, 'Class counts')
            close(sum(float(record['target_mass']) for record in records), 32, 'Soft probability mass', 1e-5)
            valid = [record for record in records if int(record['valid'])]
            require(len(valid) == int(item['shared_classes']), 'Shared-class count')
            close(sum(float(record['lmmd_loss']) for record in valid) / len(valid) if valid else 0,
                  item['lmmd_loss'], 'Class-average LMMD', 1e-5)
        require(next(iterator, None) is None, 'Unexpected extra class records')
    validate_predictions('best_epoch', rows[best_epoch - 1])
    validate_predictions('stop_epoch', rows[-1])
    for name, epoch in [('best_model.pth', best_epoch), ('stop_model.pth', len(rows))]:
        state = torch.load(RUN_DIR / name, map_location='cpu', weights_only=True)
        require(state['epoch'] == epoch and state['model']['bottleneck.weight'].shape == (256, 2048)
                and state['model']['classifier.weight'].shape == (31, 256), 'Checkpoint epoch/topology')
        require(all(torch.isfinite(value).all() for value in state['model'].values()), 'Non-finite checkpoint')
        del state
    result = {'passed': True, 'runs': 1, 'epochs': len(rows), 'steps': len(steps), 'class_rows': class_count,
                  'source_samples': 2817, 'target_samples': 795, 'patience_replayed': True,
                  'prediction_artifacts_verified': ['best_epoch', 'stop_epoch'], 'checkpoint_metadata_verified': True,
                  'target_labels_in_loss': False, 'target_labels_in_selection_and_stopping': True,
                  'author_formula': validate_author_formula(),
                  'selected_checkpoint_predictions': validate_checkpoint_predictions()}
    write_json(RUN_DIR / 'verification.json', result)
    return config, summary, rows, result


if __name__ == '__main__':
    print(json.dumps(validate_run()[3], indent=2))
