"""LMMD formula/boundary checks and validation of the one additional run."""

import argparse
import csv
import json
import math
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from src.core.lmmd import BANDWIDTH_FLOOR, lmmd
from src.experiments.lmmd.protocol import REPORTS_DIR, RUN_DIR, make_config, weights_for_epoch
from src.experiments.subdomain.protocol import RUNS_DIR
from src.experiments.subdomain.verify import close, read_csv, require, validate_predictions
from src.training.train import write_json


def numpy_reference(source, target, labels, probabilities):
    """Independent direct class-sum expression of the author's default LMMD."""
    combined = np.concatenate((source, target))
    distances = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=2)
    bandwidth = max(float(distances.sum() / (len(combined) ** 2 - len(combined))), BANDWIDTH_FLOOR)
    kernel = sum(np.exp(-distances / (bandwidth / 4 * 2 ** index)) for index in range(5))
    shared = sorted(set(labels.tolist()) & set(probabilities.argmax(1).tolist()))
    if not shared:
        return 0.0
    result = 0.0
    for label in shared:
        sw = (labels == label).astype(np.float64)
        sw /= sw.sum()
        tw = probabilities[:, label] / probabilities[:, label].sum()
        ss = float(sw @ kernel[:len(source), :len(source)] @ sw)
        tt = float(tw @ kernel[len(source):, len(source):] @ tw)
        st = float(sw @ kernel[:len(source), len(source):] @ tw)
        result += ss + tt - 2 * st
    return result / len(shared)


def mathematical_checks():
    generator = torch.Generator().manual_seed(42)
    source = torch.randn(4, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    target = torch.randn(4, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 2])
    probabilities = torch.tensor([[.8, .1, .1, 0], [.1, .8, .1, 0], [.2, .7, .1, 0], [.1, .7, .2, 0]],
                                 dtype=torch.float64, requires_grad=True)
    loss, records, _ = lmmd(source, target, labels, probabilities)
    reference = numpy_reference(source.detach().numpy(), target.detach().numpy(), labels.numpy(), probabilities.detach().numpy())
    torch.testing.assert_close(loss, torch.tensor(reference, dtype=torch.float64), atol=1e-12, rtol=1e-10)
    gradients = torch.autograd.grad(loss, (source, target, probabilities), allow_unused=True)
    require(gradients[2] is None and all(gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
                                       for gradient in gradients[:2]), 'LMMD feature/probability gradient check failed')
    require([record['valid'] for record in records] == [1, 1, 0, 0], 'LMMD shared-class mask differs')

    uniform = torch.ones(4, 1, dtype=torch.float64)
    single, _, _ = lmmd(source, target, torch.zeros(4, dtype=torch.long), uniform)
    reference = numpy_reference(source.detach().numpy(), target.detach().numpy(), np.zeros(4, dtype=np.int64), uniform.numpy())
    close(single.item(), reference, 'Single-class multi-RBF MMD', atol=1e-12, rtol=1e-10)
    singleton, singleton_records, _ = lmmd(source[:1], target[:1], torch.zeros(1, dtype=torch.long), uniform[:1])
    require(singleton_records[0]['valid'] == 1 and torch.isfinite(singleton), 'Shared singleton classes must be valid')

    absent = torch.tensor([[0., 1.]] * 4, dtype=torch.float64, requires_grad=True)
    zero, records, _ = lmmd(source, target, torch.zeros(4, dtype=torch.long), absent)
    gradients = torch.autograd.grad(zero, (source, target, absent), allow_unused=True)
    require(zero.item() == 0 and not any(record['valid'] for record in records)
            and all(record['lmmd_loss'] is None for record in records), 'Empty shared classes must be missing with zero loss')
    require(gradients[2] is None and all(torch.equal(gradient, torch.zeros_like(gradient)) for gradient in gradients[:2]),
            'No-shared-class zero must stay connected to features')

    identical_source = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    identical_target = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    identical, _, bandwidth = lmmd(identical_source, identical_target, torch.zeros(2, dtype=torch.long), torch.ones(3, 1))
    gradients = torch.autograd.grad(identical, (identical_source, identical_target))
    require(abs(identical.item()) < 1e-12 and bandwidth['lmmd_bandwidth_clamped'] == 1
            and all(torch.isfinite(gradient).all() for gradient in gradients), 'Identical features need finite bandwidth protection')

    soft = torch.tensor([[.9, .1], [.8, .2], [.1, .9]], dtype=torch.float64, requires_grad=True)
    all_images, _, _ = lmmd(source[:2], target[:3], torch.zeros(2, dtype=torch.long), soft)
    gradient = torch.autograd.grad(all_images, target)[0]
    require(gradient[2].norm() > 0, 'Other argmax classes must still contribute soft weight')
    return dict(passed=True, dtype='float64', cases=6, checks=['independent_author_formula', 'single_class_multi_rbf_mmd',
                'shared_singleton_class', 'finite_feature_gradients', 'detached_probabilities',
                'connected_zero_no_shared_class', 'absent_classes_missing', 'identical_feature_bandwidth_floor',
                'all_target_images_soft_weighted'])


def verify_warmup():
    for filename, count in [('metrics.csv', 10), ('step_metrics.csv', 60), ('class_metrics.csv', 1860)]:
        reference = read_csv(RUNS_DIR / 'global' / filename)[:count]
        actual = read_csv(RUN_DIR / filename)[:count]
        require(len(actual) == count, f'Incomplete LMMD warmup: {filename}')
        for observed, expected in zip(actual, reference):
            for key, value in expected.items():
                if isinstance(value, (int, float)):
                    close(observed[key], value, f'LMMD warmup {filename}/{key}', atol=1e-6, rtol=1e-6)
                else:
                    require(observed[key] == value, f'LMMD warmup {filename}/{key} differs')
    return dict(passed=True, epochs=10, steps=60, coral_class_rows=1860, tolerance=1e-6)


def read_lmmd_run():
    config = json.loads((RUN_DIR / 'config.json').read_text(encoding='utf-8'))
    require(config == make_config(config['class_to_idx'], config['device']), 'LMMD config mismatch')
    summary = json.loads((RUN_DIR / 'summary.json').read_text(encoding='utf-8'))
    require(not (RUN_DIR / 'failure.json').exists() and summary['training_complete'] is True,
            'LMMD run did not complete')
    require(summary['setting'] == 'lmmd' and summary['epochs_completed'] == 50 and summary['steps_completed'] == 300
            and summary['selected_epoch'] == 50 and summary['selection_metric'] == 'epoch_50_target_accuracy'
            and summary['best_metric_descriptive_only'] is True and summary['target_training_labels_used'] is False,
            'LMMD completion metadata differs')
    rows, steps = read_csv(RUN_DIR / 'metrics.csv'), read_csv(RUN_DIR / 'step_metrics.csv')
    classes, lmmd_classes = read_csv(RUN_DIR / 'class_metrics.csv'), read_csv(RUN_DIR / 'lmmd_class_metrics.csv')
    require((len(rows), len(steps), len(classes), len(lmmd_classes)) == (50, 300, 9300, 9300), 'LMMD row counts differ')
    logs = [json.loads(line) for line in (RUN_DIR / 'train.log').read_text(encoding='utf-8').splitlines()]
    require(logs == rows, 'LMMD logs/epoch CSV mismatch')
    for index, step in enumerate(steps):
        epoch, number = divmod(index, 6)
        require((step['epoch'], step['step']) == (epoch + 1, number + 1), 'LMMD step order differs')
        gw, mw = weights_for_epoch(epoch + 1)
        require((step['global_weight'], step['lmmd_weight'], step['local_weight']) == (gw, mw, 0.0), 'LMMD schedule differs')
        coral_records, lmmd_records = classes[index * 31:(index + 1) * 31], lmmd_classes[index * 31:(index + 1) * 31]
        for records, loss_key, is_lmmd in [(coral_records, 'local_loss', False), (lmmd_records, 'lmmd_loss', True)]:
            valid_losses = []
            for class_index, record in enumerate(records):
                require((record['epoch'], record['step'], record['class_index']) == (epoch + 1, number + 1, class_index),
                        'LMMD class row order differs')
                gates = [(record['source_count'] < (1 if is_lmmd else 2), 'source_count'),
                         (record['target_top1_count'] < (1 if is_lmmd else 2), 'target_top1_count')]
                if not is_lmmd:
                    gates += [(record['target_mass'] <= 1e-12, 'target_mass'), (record['target_ess'] < 2, 'target_ess')]
                reasons = [reason for condition, reason in gates if condition]
                require(record['valid'] == int(not reasons) and (record['skip_reason'] or '') == '|'.join(reasons),
                        'LMMD/CORAL support gate differs')
                if reasons:
                    require(record[loss_key] is None, 'Unsupported class distance must remain missing')
                else:
                    valid_losses.append(record[loss_key])
                require(0 <= record['target_ess'] <= 128.001 and record['target_mass'] >= 0, 'Invalid LMMD mass/ESS')
                require(step[f'target_prediction_count_{class_index:02d}'] == record['target_top1_count'], 'Histogram differs')
            require(sum(record['source_count'] for record in records) == 128
                    and sum(record['target_top1_count'] for record in records) == 128, 'Batch class counts differ')
            close(sum(record['target_mass'] for record in records), 128, 'Batch target mass')
            valid_key = 'lmmd_valid_classes' if is_lmmd else 'valid_classes'
            zero_key = 'lmmd_no_shared_classes' if is_lmmd else 'no_valid_classes'
            require(step[valid_key] == len(valid_losses) and step[zero_key] == int(not valid_losses), 'Coverage count differs')
            close(step[loss_key], sum(valid_losses) / len(valid_losses) if valid_losses else 0,
                  f'Mean {loss_key}', atol=5e-6, rtol=2e-5)
        for coral_record, lmmd_record in zip(coral_records, lmmd_records):
            for key in ('source_count', 'target_top1_count', 'target_mass'):
                close(coral_record[key], lmmd_record[key], f'CORAL/LMMD common support {key}')
        close(step['valid_class_fraction'], step['valid_classes'] / 31, 'CORAL valid fraction')
        for reason in ('source_count', 'target_top1_count', 'target_mass', 'target_ess'):
            require(step[f'skip_{reason}'] == sum(reason in (record['skip_reason'] or '').split('|')
                                                for record in coral_records), 'CORAL skip counts differ')
        close(step['weighted_global_loss'], gw * step['global_loss'], 'Weighted global')
        require(step['weighted_local_loss'] == 0, 'CORAL local must be monitored only')
        close(step['weighted_lmmd_loss'], mw * step['lmmd_loss'], 'Weighted LMMD')
        close(step['adaptation_loss'], step['weighted_global_loss'] + step['weighted_lmmd_loss'], 'LMMD adaptation')
        close(step['total_loss'], step['classification_loss'] + step['adaptation_loss'], 'LMMD total')
        close(step['lmmd_bandwidth'], max(step['lmmd_bandwidth_raw'], BANDWIDTH_FLOOR), 'LMMD bandwidth floor', atol=1e-15)
        require(step['lmmd_bandwidth_clamped'] == int(step['lmmd_bandwidth_raw'] < BANDWIDTH_FLOOR), 'Bandwidth flag differs')
    for epoch, row in enumerate(rows, 1):
        require(row['epoch'] == epoch and row['train_steps'] == 6 and row['source_samples'] == 2817
                and row['target_samples'] == 795, 'LMMD epoch/sample counts differ')
        for key in steps[0]:
            if key not in ('epoch', 'step'):
                close(row[key], sum(item[key] for item in steps[(epoch - 1) * 6:epoch * 6]) / 6,
                      f'LMMD epoch aggregate {key}', atol=1e-12, rtol=1e-10)
        for domain in ('source', 'target'):
            close(row[f'{domain}_accuracy'], row[f'{domain}_correct'] / row[f'{domain}_samples'], 'LMMD epoch accuracy')
    for key, field in [('final_source_accuracy', 'source_accuracy'), ('final_target_accuracy', 'target_accuracy'),
                       ('final_target_macro_accuracy', 'target_macro_accuracy')]:
        close(summary[key], rows[-1][field], f'LMMD summary {key}')
    best = max(rows, key=lambda row: row['target_accuracy'])
    require(summary['best_target_epoch'] == best['epoch'], 'LMMD descriptive best epoch differs')
    close(summary['best_target_accuracy'], best['target_accuracy'], 'LMMD descriptive best accuracy')
    validate_predictions(RUN_DIR, rows, config)
    verify_warmup()
    recorded = json.loads((RUN_DIR / 'environment.json').read_text(encoding='utf-8'))
    require(recorded == json.loads((RUNS_DIR / 'global' / 'environment.json').read_text(encoding='utf-8')),
            'LMMD/global environments differ')
    for domain in ('source', 'target'):
        images = []
        for directory in (RUN_DIR, RUNS_DIR / 'global'):
            with (directory / 'final_epoch' / f'{domain}_predictions.csv').open(newline='', encoding='utf-8') as stream:
                images.append([row['image'] for row in csv.DictReader(stream)])
        require(images[0] == images[1], 'LMMD/global image order differs')
    return config, rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--math-only', action='store_true')
    args = parser.parse_args()
    result = {'mathematics': mathematical_checks()}
    if not args.math_only:
        read_lmmd_run()
        result['run'] = dict(passed=True, epochs=50, steps=300, lmmd_class_rows=9300, coral_monitor_class_rows=9300,
                             target_samples=795, warmup=verify_warmup(), target_training_labels_used=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_DIR / ('math_validation.json' if args.math_only else 'verification.json'), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
