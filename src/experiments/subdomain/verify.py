"""Necessary mathematical checks and cross-artifact verification for the suite."""

import argparse
import csv
import json
import math
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from src.core.coral import coral
from src.core.subdomain_coral import class_conditional_coral
from src.experiments.subdomain.protocol import (
    ADAPTATION_SETTINGS, BATCH_SIZE, CLASS_COUNT, EPOCHS, REPORTS_DIR, RUNS_DIR,
    SETTINGS, STEPS_PER_EPOCH, WARMUP_EPOCHS, make_config, weights_for_epoch,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context, *, atol=2e-6, rtol=2e-5):
    require(math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
            f'{context}: {actual} != {expected}')


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value == '':
                row[key] = None
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                row[key] = value
            else:
                require(math.isfinite(number), f'Non-finite value in {path}: {key}')
                row[key] = number
    return rows


def mathematical_checks():
    generator = torch.Generator().manual_seed(42)
    source = torch.randn(7, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    target = torch.randn(9, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    probabilities = torch.ones(9, 1, dtype=torch.float64, requires_grad=True)
    loss, records = class_conditional_coral(source, target, torch.zeros(7, dtype=torch.long), probabilities)
    torch.testing.assert_close(loss, coral(source, target), atol=1e-12, rtol=1e-10)
    gradients = torch.autograd.grad(loss, (source, target, probabilities), allow_unused=True)
    require(records[0]['valid'] == 1 and gradients[2] is None, 'Probability weights must be detached')
    require(all(gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
                for gradient in gradients[:2]), 'Both feature gradients must be finite and connected')

    source = torch.randn(3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    target = torch.randn(3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    loss, records = class_conditional_coral(source, target, torch.arange(3), torch.eye(3, dtype=torch.float64))
    gradients = torch.autograd.grad(loss, (source, target))
    require(loss.item() == 0 and not any(record['valid'] for record in records), 'Unsupported batch must be zero')
    require(all(torch.equal(gradient, torch.zeros_like(gradient)) for gradient in gradients), 'Zero must stay connected')
    require(all(record['local_loss'] is None for record in records), 'Invalid distances must be missing, not zero')

    loss, records = class_conditional_coral(source[:1], target[:1], torch.tensor([0]),
                                           torch.tensor([[1., 0., 0.]], dtype=torch.float64))
    require(loss.item() == 0 and records[1]['target_mass'] == 0 and records[1]['target_ess'] == 0,
            'Single samples and absent classes must remain finite')

    probabilities = torch.tensor([[.99, .005, .005], [.34, .33, .33], [.01, .49, .50]], dtype=torch.float64)
    _, records = class_conditional_coral(source, target, torch.tensor([0, 0, 1]), probabilities)
    require(records[0]['target_top1_count'] == 2 and records[0]['target_ess'] < 2
            and records[0]['valid'] == 0 and 'target_ess' in records[0]['skip_reason'], 'ESS gate failed')

    probabilities = torch.tensor([[.9, .1], [.8, .2], [.1, .9]], dtype=torch.float64, requires_grad=True)
    loss, records = class_conditional_coral(source[:2], target, torch.zeros(2, dtype=torch.long), probabilities)
    gradient = torch.autograd.grad(loss, target)[0]
    require(records[0]['valid'] == 1 and gradient[2].norm() > 0,
            'Images with another top-1 class must still contribute their soft weight')
    result = dict(passed=True, dtype='float64', cases=5,
                  checks=['single_class_equals_global_coral', 'finite_feature_gradients',
                          'probability_weights_detached', 'connected_zero_and_missing_distances',
                          'single_samples_and_absent_classes', 'effective_sample_size_gate',
                          'all_target_images_contribute_soft_weights'])
    return result


def validate_predictions(directory, rows, config):
    with np.load(directory / 'final_epoch' / 'evaluation.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    classes = arrays['classes'].tolist()
    require(config['class_to_idx'] == {name: index for index, name in enumerate(classes)}, 'NPZ class mapping differs')
    for domain, sample_count in [('source', 2817), ('target', 795)]:
        logits, labels = arrays[f'{domain}_logits'], arrays[f'{domain}_labels']
        require(logits.shape == (sample_count, CLASS_COUNT) and labels.shape == (sample_count,), 'Wrong NPZ shapes')
        require(np.isfinite(logits).all() and labels.dtype.kind in 'iu'
                and set(labels.tolist()) == set(range(CLASS_COUNT)), 'Invalid final predictions or labels')
        predictions = logits.argmax(1)
        close(rows[-1][f'{domain}_accuracy'], float(np.mean(predictions == labels)), f'{domain} final accuracy')
        require(rows[-1][f'{domain}_correct'] == int(np.sum(predictions == labels)), 'Final correct count differs')
        macro = np.mean([np.mean(predictions[labels == index] == index) for index in range(CLASS_COUNT)])
        close(rows[-1][f'{domain}_macro_accuracy'], float(macro), f'{domain} final macro accuracy')
        probabilities = torch.from_numpy(logits).softmax(1).numpy()
        with (directory / 'final_epoch' / f'{domain}_predictions.csv').open(newline='', encoding='utf-8') as stream:
            prediction_rows = list(csv.DictReader(stream))
        require(len(prediction_rows) == sample_count, 'Prediction CSV does not cover the full domain')
        require(len({row['image'] for row in prediction_rows}) == sample_count, 'Repeated final evaluation image')
        for index, row in enumerate(prediction_rows):
            require(int(row['sample_index']) == index and int(row['label_index']) == labels[index]
                    and int(row['prediction_index']) == predictions[index]
                    and row['label'] == classes[labels[index]] and row['prediction'] == classes[predictions[index]]
                    and row['image'].split('/')[-2] == row['label']
                    and int(row['correct']) == int(predictions[index] == labels[index]), 'Prediction CSV/NPZ mismatch')
            close(float(row['confidence']), float(probabilities[index].max()), 'Final prediction confidence')
    require(read_csv(directory / 'final_epoch' / 'metrics.csv') == [rows[-1]], 'Final metrics snapshot differs')
    final_metadata = json.loads((directory / 'final_epoch.json').read_text(encoding='utf-8'))
    require(final_metadata['epoch'] == EPOCHS and final_metadata['selection_metric'] == 'epoch_50_target_accuracy'
            and final_metadata['target_accuracy_used_for_selection'] is False, 'Incorrect final selection metadata')
    return arrays


def read_run(directory):
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    require(config == make_config(directory.name, config['class_to_idx'], config['device']), 'Protocol mismatch')
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    require(not (directory / 'failure.json').exists() and summary.get('training_complete') is True,
            f'Run is not complete: {directory}')
    require(summary['setting'] == directory.name and summary['epochs_completed'] == EPOCHS
            and summary['steps_completed'] == EPOCHS * STEPS_PER_EPOCH and summary['selected_epoch'] == EPOCHS
            and summary['selection_metric'] == 'epoch_50_target_accuracy'
            and summary['target_training_labels_used'] is False, 'Wrong completion metadata')
    rows = read_csv(directory / 'metrics.csv')
    steps = read_csv(directory / 'step_metrics.csv')
    classes = read_csv(directory / 'class_metrics.csv')
    require(len(rows) == EPOCHS and len(steps) == EPOCHS * STEPS_PER_EPOCH
            and len(classes) == len(steps) * CLASS_COUNT, 'Wrong number of epoch/step/class rows')
    logs = [json.loads(line) for line in (directory / 'train.log').read_text(encoding='utf-8').splitlines()]
    require(logs == rows, 'Training log differs from metrics CSV')
    for index, step in enumerate(steps):
        epoch, number = divmod(index, STEPS_PER_EPOCH)
        require((step['epoch'], step['step']) == (epoch + 1, number + 1), 'Non-contiguous step sequence')
        weights = weights_for_epoch(directory.name, epoch + 1)
        require((step['global_weight'], step['local_weight']) == weights, 'Wrong warmup or post-warmup weights')
        class_rows = classes[index * CLASS_COUNT:(index + 1) * CLASS_COUNT]
        valid_losses = []
        for class_index, record in enumerate(class_rows):
            require((record['epoch'], record['step'], record['class_index']) == (epoch + 1, number + 1, class_index),
                    'Non-contiguous class sequence')
            reasons = [reason for condition, reason in [
                (record['source_count'] < 2, 'source_count'),
                (record['target_top1_count'] < 2, 'target_top1_count'),
                (record['target_mass'] <= 1e-12, 'target_mass'),
                (record['target_ess'] < 2, 'target_ess')] if condition]
            require(record['valid'] == int(not reasons) and (record['skip_reason'] or '') == '|'.join(reasons),
                    'Invalid class support decision')
            if reasons:
                require(record['local_loss'] is None, 'Invalid class distance was filled with a value')
            else:
                require(record['local_loss'] >= 0, 'Negative local distance')
                valid_losses.append(record['local_loss'])
            require(record['target_mass'] >= 0 and 0 <= record['target_ess'] <= BATCH_SIZE + 1e-3,
                    'Invalid target mass/ESS')
            require(step[f'target_prediction_count_{class_index:02d}'] == record['target_top1_count'],
                    'Target histogram differs from class rows')
        require(sum(record['source_count'] for record in class_rows) == BATCH_SIZE
                and sum(record['target_top1_count'] for record in class_rows) == BATCH_SIZE,
                'Class counts do not sum to the batch size')
        close(sum(record['target_mass'] for record in class_rows), BATCH_SIZE, 'Target probability mass')
        require(step['valid_classes'] == len(valid_losses) and step['no_valid_classes'] == int(not valid_losses),
                'Valid class count differs')
        close(step['valid_class_fraction'], len(valid_losses) / CLASS_COUNT, 'Valid class fraction')
        for reason in ('source_count', 'target_top1_count', 'target_mass', 'target_ess'):
            require(step[f'skip_{reason}'] == sum(reason in (record['skip_reason'] or '').split('|')
                                                 for record in class_rows), 'Skip reason aggregate differs')
        close(step['local_loss'], sum(valid_losses) / len(valid_losses) if valid_losses else 0, 'Mean valid local loss')
        close(step['weighted_global_loss'], weights[0] * step['global_loss'], 'Weighted global loss')
        close(step['weighted_local_loss'], weights[1] * step['local_loss'], 'Weighted local loss')
        close(step['adaptation_loss'], step['weighted_global_loss'] + step['weighted_local_loss'], 'Adaptation loss')
        close(step['total_loss'], step['classification_loss'] + step['adaptation_loss'], 'Total loss')
    for epoch, row in enumerate(rows, 1):
        require(row['epoch'] == epoch and row['train_steps'] == STEPS_PER_EPOCH
                and row['source_samples'] == 2817 and row['target_samples'] == 795, 'Wrong epoch/sample counts')
        for key in steps[0]:
            if key not in ('epoch', 'step'):
                values = steps[(epoch - 1) * STEPS_PER_EPOCH:epoch * STEPS_PER_EPOCH]
                close(row[key], sum(value[key] for value in values) / STEPS_PER_EPOCH,
                      f'Epoch aggregate {key}', atol=1e-12, rtol=1e-10)
        for domain in ('source', 'target'):
            close(row[f'{domain}_accuracy'], row[f'{domain}_correct'] / row[f'{domain}_samples'], 'Epoch accuracy')
    best = max(rows, key=lambda row: row['target_accuracy'])
    require(summary['best_target_epoch'] == best['epoch'], 'Best descriptive epoch is incorrect')
    close(summary['best_target_accuracy'], best['target_accuracy'], 'Best descriptive accuracy')
    close(summary['final_target_accuracy'], rows[-1]['target_accuracy'], 'Summary final accuracy')
    close(summary['final_target_macro_accuracy'], rows[-1]['target_macro_accuracy'], 'Summary final macro accuracy')
    validate_predictions(directory, rows, config)
    return config, rows, summary


def verify_suite(*, return_runs=False):
    runs = {setting: read_run(RUNS_DIR / setting) for setting in SETTINGS}
    reference_environment = json.loads((RUNS_DIR / 'global' / 'environment.json').read_text(encoding='utf-8'))
    reference_steps = read_csv(RUNS_DIR / 'global' / 'step_metrics.csv')[:WARMUP_EPOCHS * STEPS_PER_EPOCH]
    warmup_class_count = WARMUP_EPOCHS * STEPS_PER_EPOCH * CLASS_COUNT
    reference_classes = read_csv(RUNS_DIR / 'global' / 'class_metrics.csv')[:warmup_class_count]
    reference_images = {}
    for domain in ('source', 'target'):
        with (RUNS_DIR / 'global' / 'final_epoch' / f'{domain}_predictions.csv').open(
                newline='', encoding='utf-8') as stream:
            reference_images[domain] = [row['image'] for row in csv.DictReader(stream)]
    for setting in SETTINGS:
        recorded = json.loads((RUNS_DIR / setting / 'environment.json').read_text(encoding='utf-8'))
        require(recorded == reference_environment, 'Suite environments differ')
        for domain in ('source', 'target'):
            with (RUNS_DIR / setting / 'final_epoch' / f'{domain}_predictions.csv').open(
                    newline='', encoding='utf-8') as stream:
                images = [row['image'] for row in csv.DictReader(stream)]
            require(images == reference_images[domain], f'{setting} final image order differs: {domain}')
        if setting in ADAPTATION_SETTINGS:
            steps = read_csv(RUNS_DIR / setting / 'step_metrics.csv')[:len(reference_steps)]
            for actual, reference in zip(steps, reference_steps):
                for key in reference:
                    close(actual[key], reference[key], f'{setting} warmup step {key}', atol=1e-6, rtol=1e-6)
            for actual, reference in zip(runs[setting][1][:WARMUP_EPOCHS], runs['global'][1][:WARMUP_EPOCHS]):
                for key in reference:
                    close(actual[key], reference[key], f'{setting} warmup epoch {key}', atol=1e-6, rtol=1e-6)
            classes = read_csv(RUNS_DIR / setting / 'class_metrics.csv')[:warmup_class_count]
            for actual, reference in zip(classes, reference_classes):
                for key, expected in reference.items():
                    if isinstance(expected, (int, float)):
                        close(actual[key], expected, f'{setting} warmup class {key}', atol=1e-6, rtol=1e-6)
                    else:
                        require(actual[key] == expected, f'{setting} warmup class {key} differs')
    result = dict(passed=True, runs=len(runs), epochs=EPOCHS * len(runs),
                  steps=EPOCHS * STEPS_PER_EPOCH * len(runs),
                  class_rows=EPOCHS * STEPS_PER_EPOCH * CLASS_COUNT * len(runs),
                  warmup_epochs_verified=WARMUP_EPOCHS, target_samples_per_epoch=795,
                  warmup_class_rows_verified_per_run=warmup_class_count,
                  checks=['locked_protocol', 'warmup_equality', 'weight_schedule', 'class_support',
                          'loss_reductions', 'csv_log_npz_predictions', 'full_domain_accuracy',
                          'epoch_50_selection', 'matching_environments', 'paired_image_order'])
    return (result, runs) if return_runs else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--math-only', action='store_true')
    args = parser.parse_args()
    result = {'mathematics': mathematical_checks()}
    if not args.math_only:
        result['suite'] = verify_suite()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = 'math_validation.json' if args.math_only else 'verification.json'
    (REPORTS_DIR / name).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
