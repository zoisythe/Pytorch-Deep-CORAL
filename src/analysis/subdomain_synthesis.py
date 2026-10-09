"""Recompute post-hoc scale/shape diagnostics from the six archived runs.

Reads only tracked report snapshots. Does not train, change existing archives,
or overwrite the authored mathematical report. See analysis/protocol.md.
"""

import csv
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.common.paths import ROOT

BASE = ROOT / 'reports' / 'subdomain_coral'
DESTINATION = BASE / 'analysis'
SETTINGS = ('no_adaptation', 'global', 'global_quarter', 'local', 'mixed', 'lmmd')
NAMES = ('CE only', 'Global 0.5', 'Global 0.25', 'Local 0.5', 'Global + local', 'LMMD 0.5')
COLORS = ('#64748b', '#2563eb', '#0891b2', '#c2410c', '#7c3aed', '#059669')
DIMENSION = 31


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def covariance(features):
    centered = features - features.mean(0)
    return centered.T @ centered / (len(features) - 1)


def cosine_distance(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1e-12:
        return None
    return float(1 - np.sum(left * right) / denominator)


def available_mean(values):
    supported = [value for value in values if value is not None]
    return float(np.mean(supported)) if supported else None


def effective_rank(matrix):
    values = np.linalg.eigvalsh(matrix).clip(min=0)
    if values.sum() == 0:
        return 0.0
    probabilities = values / values.sum()
    positive = probabilities > 0
    return float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))


def covariance_components(features, labels):
    """Exact finite-sample within/between decomposition, with N-1 divisor."""
    mean = features.mean(0)
    within = np.zeros((DIMENSION, DIMENSION))
    between = np.zeros_like(within)
    for index in range(DIMENSION):
        selected = features[labels == index]
        within += (len(selected) - 1) * covariance(selected)
        offset = selected.mean(0) - mean
        between += len(selected) * np.outer(offset, offset)
    return within / (len(features) - 1), between / (len(features) - 1)


def diagnostics(setting, representation, source, target, source_labels, target_labels, classes, checks):
    source_cov, target_cov = covariance(source), covariance(target)
    shifted = target - target.max(1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(1, keepdims=True)
    target_weights = probabilities / probabilities.sum(0, keepdims=True)
    per_class = []
    for index, name in enumerate(classes):
        sc = source[source_labels == index]
        tc = target[target_labels == index]
        if min(len(sc), len(tc)) < 2:
            raise ValueError('The archived full domains must support all 31 true classes')
        cs, ct = covariance(sc), covariance(tc)
        ns, nt = np.linalg.norm(cs), np.linalg.norm(ct)
        distance = float(np.square(cs - ct).sum() / (4 * DIMENSION ** 2))
        scale_part = float((ns - nt) ** 2 / (4 * DIMENSION ** 2))
        orientation_part = float(2 * (ns * nt - np.sum(cs * ct)) / (4 * DIMENSION ** 2))
        checks['covariance_distance_decomposition_error'] = max(
            checks['covariance_distance_decomposition_error'], abs(distance - scale_part - orientation_part))
        per_class.append(dict(
            setting=setting, representation=representation, class_index=index, class_name=name,
            source_samples=len(sc), target_samples=len(tc),
            target_correct=int((tc.argmax(1) == index).sum()),
            target_accuracy=float((tc.argmax(1) == index).mean()),
            true_covariance_distance=distance, covariance_scale_part=scale_part,
            covariance_orientation_part=orientation_part, covariance_cosine_distance=cosine_distance(cs, ct),
            centroid_mse=float(np.square(sc.mean(0) - tc.mean(0)).mean()),
            centroid_cosine_distance=cosine_distance(sc.mean(0), tc.mean(0)),
            source_covariance_trace=float(np.trace(cs)), target_covariance_trace=float(np.trace(ct)),
            target_soft_class_purity=float(target_weights[target_labels == index, index].sum()),
            target_soft_class_ess=float(1 / np.square(target_weights[:, index]).sum())))
    result = dict(
        setting=setting, representation=representation,
        target_correct=int((target.argmax(1) == target_labels).sum()), target_samples=len(target),
        target_accuracy=float((target.argmax(1) == target_labels).mean()),
        target_macro_accuracy=float(np.mean([row['target_accuracy'] for row in per_class])),
        global_covariance_distance=float(np.square(source_cov - target_cov).sum() / (4 * DIMENSION ** 2)),
        true_local_covariance_distance=float(np.mean([row['true_covariance_distance'] for row in per_class])),
        covariance_scale_part=float(np.mean([row['covariance_scale_part'] for row in per_class])),
        covariance_orientation_part=float(np.mean([row['covariance_orientation_part'] for row in per_class])),
        covariance_cosine_distance=available_mean([row['covariance_cosine_distance'] for row in per_class]),
        centroid_mse=float(np.mean([row['centroid_mse'] for row in per_class])),
        centroid_cosine_distance=available_mean([row['centroid_cosine_distance'] for row in per_class]),
        covariance_cosine_missing=sum(row['covariance_cosine_distance'] is None for row in per_class),
        centroid_cosine_missing=sum(row['centroid_cosine_distance'] is None for row in per_class),
        source_mean_logit_norm=float(np.linalg.norm(source, axis=1).mean()),
        target_mean_logit_norm=float(np.linalg.norm(target, axis=1).mean()),
        source_covariance_trace=float(np.trace(source_cov)), target_covariance_trace=float(np.trace(target_cov)),
        source_effective_rank=effective_rank(source_cov), target_effective_rank=effective_rank(target_cov),
        target_confidence=float(probabilities.max(1).mean()),
        target_soft_class_purity=float(np.mean([row['target_soft_class_purity'] for row in per_class])),
        target_soft_class_ess=float(np.mean([row['target_soft_class_ess'] for row in per_class])))
    for domain, features, labels in [('source', source, source_labels), ('target', target, target_labels)]:
        within, between = covariance_components(features, labels)
        result[f'{domain}_within_trace'] = float(np.trace(within))
        result[f'{domain}_between_trace'] = float(np.trace(between))
        checks['within_between_covariance_error'] = max(
            checks['within_between_covariance_error'], float(np.max(np.abs(covariance(features) - within - between))))
    centered_target = target - target.mean(1, keepdims=True)
    alternative = target.copy()
    alternative[np.arange(len(target)), target_labels] = -np.inf
    margin = target[np.arange(len(target)), target_labels] - alternative.max(1)
    norms = np.linalg.norm(centered_target, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError('Normalized target margins require nonzero centered logits')
    result['target_normalized_margin_median'] = float(np.median(margin / norms))
    return result, per_class


def mathematical_identities():
    """Independent numerical checks of identities used in the authored report."""
    rng = np.random.default_rng(42)
    source, target = rng.normal(size=(8, 4)), rng.normal(size=(9, 4))
    a, b = rng.uniform(size=8), rng.uniform(size=9)
    a, b = a / a.sum(), b / b.sum()

    def weighted_cov(z, weights):
        mean = weights @ z
        centered = z - mean
        return centered.T @ (centered * weights[:, None]) / (1 - np.square(weights).sum())

    def local(zs, zt):
        return np.square(weighted_cov(zs, a) - weighted_cov(zt, b)).sum()

    scale = 0.37
    uniform = np.full(len(source), 1 / len(source))
    combined = np.concatenate((source, target))
    distances = np.square(combined[:, None] - combined[None, :]).sum(2)
    bandwidth = distances.sum() / (len(combined) * (len(combined) - 1))
    kernels = sum(np.exp(-distances / (bandwidth * 2 ** (q - 2))) for q in range(5))
    scaled_distances = np.square(scale * combined[:, None] - scale * combined[None, :]).sum(2)
    scaled_bandwidth = scaled_distances.sum() / (len(combined) * (len(combined) - 1))
    scaled_kernels = sum(np.exp(-scaled_distances / (scaled_bandwidth * 2 ** (q - 2))) for q in range(5))
    # A linear kernel has an explicit feature map, so the kernel expansion is independently checkable.
    linear = combined @ combined.T
    weights = np.concatenate((a, -b))
    direct_mean = np.square(a @ source - b @ target).sum()
    errors = dict(
        uniform_weight_covariance=float(np.max(np.abs(weighted_cov(source, uniform) - covariance(source)))),
        frozen_weight_covariance_scaling=float(abs(local(scale * source, scale * target) - scale ** 4 * local(source, target))),
        frozen_weight_domain_translation=float(abs(local(source + 7, target - 3) - local(source, target))),
        bandwidth_twice_trace=float(abs(bandwidth - 2 * np.trace(covariance(combined)))),
        frozen_weight_adaptive_rbf_scaling=float(np.max(np.abs(kernels - scaled_kernels))),
        linear_kernel_mean_expansion=float(abs(weights @ linear @ weights - direct_mean)))
    if max(errors.values()) > 1e-11:
        raise AssertionError(errors)
    dimension = source.shape[1]
    delta = weighted_cov(source, a) - weighted_cov(target, b)
    analytic_gradient = (np.diag(a) - np.outer(a, a)) @ source @ delta
    analytic_gradient /= dimension ** 2 * (1 - np.square(a).sum())
    difference_gradient = np.empty_like(source)
    epsilon = 1e-5
    for index in np.ndindex(source.shape):
        left, right = source.copy(), source.copy()
        left[index] += epsilon
        right[index] -= epsilon
        difference_gradient[index] = (local(left, target) - local(right, target)) / (8 * dimension ** 2 * epsilon)
    errors['weighted_coral_gradient_finite_difference'] = float(np.max(np.abs(analytic_gradient - difference_gradient)))
    if errors['weighted_coral_gradient_finite_difference'] > 1e-8:
        raise AssertionError(errors)
    return errors


def plot(rows):
    selected = [row for row in rows if row['representation'] == 'contrast']
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), layout='constrained')
    definitions = [
        ('target_covariance_trace', 'Target logit contrast variance', 'Covariance trace'),
        ('covariance_cosine_distance', 'True-class covariance shape mismatch', 'Mean Frobenius cosine distance'),
        ('centroid_cosine_distance', 'True-class centroid direction mismatch', 'Mean cosine distance'),
        ('target_soft_class_purity', 'Target soft class-weight purity', 'Mean true-class weight mass')]
    for axis, (field, title, ylabel) in zip(axes.flat, definitions):
        values = [row[field] for row in selected]
        bars = axis.bar(range(6), values, color=COLORS, width=0.7)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.3f}',
                      ha='center', va='bottom', fontsize=9)
        axis.set_xticks(range(6), NAMES, rotation=20, ha='right', fontsize=9)
        axis.set(title=title, ylabel=ylabel, ylim=(0, max(values) * 1.18))
        axis.grid(axis='y', alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Epoch 50 | fc8 with per-image common offset removed | full-domain post-hoc diagnostics', fontsize=12)
    destination = DESTINATION / 'assets' / 'scale_shape.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    summaries = {row['setting']: row for row in read_csv(BASE / 'lmmd' / 'summary.csv')}
    previous = {row['setting']: row for row in read_csv(BASE / 'lmmd' / 'diagnostics.csv')}
    checks = dict(covariance_distance_decomposition_error=0.0, within_between_covariance_error=0.0,
                  existing_diagnostic_error=0.0, common_offset_probability_error=0.0)
    aggregate, class_rows, phases, predictions, paths = [], [], [], {}, []
    class_names, domain_labels = None, None
    image_orders = {}
    fields = ('classification_loss', 'global_loss', 'local_loss', 'weighted_global_loss',
              'weighted_local_loss', 'adaptation_loss', 'valid_classes', 'target_confidence',
              'target_entropy', 'no_valid_classes', 'lmmd_loss', 'weighted_lmmd_loss',
              'lmmd_valid_classes', 'lmmd_no_shared_classes', 'lmmd_bandwidth_raw', 'lmmd_bandwidth_clamped')
    for setting in SETTINGS:
        run = BASE / ('lmmd/data/lmmd' if setting == 'lmmd' else f'data/{setting}')
        archive_path = run / 'final_epoch' / 'evaluation.npz'
        paths.extend([archive_path, run / 'metrics.csv'])
        with np.load(archive_path, allow_pickle=False) as archive:
            source, target = archive['source_logits'].astype(np.float64), archive['target_logits'].astype(np.float64)
            sl, tl, classes = archive['source_labels'], archive['target_labels'], archive['classes']
        if source.shape != (2817, 31) or target.shape != (795, 31):
            raise AssertionError(f'Unexpected whole-domain shapes: {setting}')
        for domain, features, labels in [('source', source, sl), ('target', target, tl)]:
            prediction_path = run / 'final_epoch' / f'{domain}_predictions.csv'
            paths.append(prediction_path)
            records = read_csv(prediction_path)
            order = [row['image'] for row in records]
            if domain not in image_orders:
                image_orders[domain] = order
            if order != image_orders[domain] or len(set(order)) != len(features):
                raise AssertionError(f'Archived {domain} image order differs or has duplicates')
            np.testing.assert_array_equal([int(row['sample_index']) for row in records], np.arange(len(features)))
            np.testing.assert_array_equal([int(row['label_index']) for row in records], labels)
            np.testing.assert_array_equal([int(row['prediction_index']) for row in records], features.argmax(1))
        if class_names is None:
            class_names, domain_labels = classes.copy(), (sl.copy(), tl.copy())
        if not (np.array_equal(classes, class_names) and np.array_equal(sl, domain_labels[0])
                and np.array_equal(tl, domain_labels[1])):
            raise AssertionError('Archived sample labels or class order differ')
        for representation in ('raw', 'contrast'):
            zs, zt = (source, target) if representation == 'raw' else (
                source - source.mean(1, keepdims=True), target - target.mean(1, keepdims=True))
            row, per_class = diagnostics(setting, representation, zs, zt, sl, tl, classes, checks)
            np.testing.assert_allclose(row['target_accuracy'], float(summaries[setting]['target_accuracy']), atol=1e-15, rtol=0)
            np.testing.assert_allclose(row['target_macro_accuracy'], float(summaries[setting]['target_macro_accuracy']), atol=1e-7, rtol=0)
            aggregate.append(row)
            class_rows.extend(per_class)
            if representation == 'raw':
                for key in ('global_covariance_distance', 'true_local_covariance_distance', 'centroid_mse',
                            'source_mean_logit_norm', 'target_mean_logit_norm', 'source_covariance_trace',
                            'target_covariance_trace', 'source_effective_rank', 'target_effective_rank', 'target_confidence'):
                    error = abs(row[key] - float(previous[setting][key]))
                    checks['existing_diagnostic_error'] = max(checks['existing_diagnostic_error'], error)
                    np.testing.assert_allclose(row[key], float(previous[setting][key]), atol=1e-10, rtol=1e-10)
                predictions[setting] = target.argmax(1)
            else:
                if not np.array_equal(zt.argmax(1), predictions[setting]):
                    raise AssertionError('Removing common logit offsets changed predictions')
                shifted = target - target.max(1, keepdims=True)
                raw_prob = np.exp(shifted) / np.exp(shifted).sum(1, keepdims=True)
                contrast_shifted = zt - zt.max(1, keepdims=True)
                contrast_prob = np.exp(contrast_shifted) / np.exp(contrast_shifted).sum(1, keepdims=True)
                checks['common_offset_probability_error'] = max(
                    checks['common_offset_probability_error'], float(np.max(np.abs(raw_prob - contrast_prob))))
        epochs = read_csv(run / 'metrics.csv')
        if [int(row['epoch']) for row in epochs] != list(range(1, 51)) or any(int(row['train_steps']) != 6 for row in epochs):
            raise AssertionError('Expected 50 complete six-step epochs')
        for name, selected in [('warmup_1_10', epochs[:10]), ('adaptation_11_50', epochs[10:]), ('epoch_50', epochs[-1:])]:
            row = dict(setting=setting, phase=name, epochs=len(selected), steps=6 * len(selected))
            row.update({field: float(np.mean([float(item[field]) for item in selected])) if field in selected[0] else None
                        for field in fields})
            phases.append(row)
    comparisons = []
    for setting, reference in [('global', 'no_adaptation'), ('global_quarter', 'global'), ('local', 'global'),
                               ('mixed', 'global_quarter'), ('mixed', 'global'), ('lmmd', 'global'),
                               ('lmmd', 'global_quarter'), ('lmmd', 'local'), ('lmmd', 'mixed'), ('lmmd', 'no_adaptation')]:
        current = predictions[setting] == domain_labels[1]
        control = predictions[reference] == domain_labels[1]
        corrected, regressed = int((current & ~control).sum()), int((~current & control).sum())
        comparisons.append(dict(setting=setting, reference=reference, target_samples=795, corrected=corrected,
                                regressed=regressed, net_correct=corrected - regressed,
                                gain_pp=100 * (corrected - regressed) / 795))
    for row in aggregate + class_rows + phases + comparisons:
        if any(isinstance(value, (float, np.floating)) and not np.isfinite(value) for value in row.values()):
            raise FloatingPointError('Non-finite offline diagnostic')
    if max(checks.values()) > 1e-9:
        raise AssertionError(checks)
    verification = dict(passed=True, training_performed=False, runs=6, domain_samples={'source': 2817, 'target': 795},
                        representations=2, diagnostic_rows=len(aggregate), class_rows=len(class_rows),
                        numerical_errors=checks, mathematical_identities=mathematical_identities(),
                        paired_image_order_verified=True, prediction_csvs_verified=12,
                        normalized_diagnostic_missing=sum(row['covariance_cosine_missing'] + row['centroid_cosine_missing']
                                                          for row in aggregate),
                        target_labels_used='evaluation_and_posthoc_diagnostics_only')
    write_csv(DESTINATION / 'diagnostics.csv', aggregate)
    write_csv(DESTINATION / 'class_diagnostics.csv', class_rows)
    write_csv(DESTINATION / 'training_phases.csv', phases)
    write_csv(DESTINATION / 'comparisons.csv', comparisons)
    write_json(DESTINATION / 'verification.json', verification)
    write_json(DESTINATION / 'provenance.json', dict(
        analysis_type='posthoc_saved_epoch_50_logits', as_of='2026-10-09', training_performed=False,
        dtype='float64', source_files=[path.relative_to(ROOT).as_posix() for path in paths]
        + ['reports/subdomain_coral/lmmd/summary.csv', 'reports/subdomain_coral/lmmd/diagnostics.csv'],
        generator='src/analysis/subdomain_synthesis.py', protocol='reports/subdomain_coral/analysis/protocol.md',
        missing_values='empty_csv_cells_are_undefined_not_zero',
        representations={'raw': 'unmodified_fc8_logits', 'contrast': 'subtract_per_image_mean_across_31_logits'},
        aggregation='equal_mean_of_31_true_classes_for_class_diagnostics; all_795_images_for_accuracy_and_margin',
        training_phases='unweighted_mean_of_epoch_means; six_updates_in_every_epoch',
        weight_purity='sum_normalized_soft_target_class_weights_over_that_true_class; not_calibration'))
    plot(aggregate)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
