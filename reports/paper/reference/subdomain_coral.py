"""Class-conditional covariance alignment with detached soft target weights."""

import torch
import torch.nn.functional as F

MIN_SOURCE = 2
MIN_TARGET_TOP1 = 2
MIN_TARGET_ESS = 2.0
MASS_EPS = 1e-12


def weighted_covariances(features, normalized_weights):
    """Return one finite-sample-corrected covariance per weight row [C, N]."""
    means = normalized_weights @ features
    centered = features.unsqueeze(0) - means.unsqueeze(1)
    numerator = centered.transpose(1, 2) @ (centered * normalized_weights.unsqueeze(2))
    correction = 1 - normalized_weights.square().sum(dim=1)
    # Unsupported classes are excluded by the caller; keep their arithmetic finite.
    return numerator / correction.clamp_min(MASS_EPS)[:, None, None]


def class_conditional_coral(source, target, source_labels, target_probabilities):
    """Return (loss, class records); target training labels are not an input.

    Inputs are [Ns, D], [Nt, D], [Ns], [Nt, C]. Top-1 counts gate classes,
    while every target image contributes its detached soft probability weight.
    The correction reduces to N-1 for uniform weights; feature-dependent soft
    weights do not imply an unbiased population-covariance estimator.
    """
    if (source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]
            or min(source.shape) < 1 or min(target.shape) < 1):
        raise ValueError('Class-conditional CORAL needs nonempty [N, D] feature tensors')
    if (source.device != target.device or source.dtype != target.dtype
            or not source.is_floating_point() or not target.is_floating_point()):
        raise ValueError('Features must share a floating dtype and device')
    if (source_labels.shape != (len(source),) or source_labels.device != source.device
            or source_labels.dtype not in (torch.int32, torch.int64)):
        raise ValueError('Source labels must be an integer [Ns] tensor on the feature device')
    if (target_probabilities.ndim != 2 or len(target_probabilities) != len(target)
            or target_probabilities.shape[1] < 1 or target_probabilities.device != target.device
            or not target_probabilities.is_floating_point()):
        raise ValueError('Target probabilities must be floating [Nt, C] on the feature device')
    probabilities = target_probabilities.detach().to(dtype=source.dtype)
    classes = probabilities.shape[1]
    if (not torch.isfinite(source).all() or not torch.isfinite(target).all()
            or not torch.isfinite(probabilities).all()):
        raise FloatingPointError('Non-finite class-conditional CORAL inputs')
    if (torch.any(source_labels < 0) or torch.any(source_labels >= classes)
            or torch.any(probabilities < 0)
            or not torch.allclose(probabilities.sum(1), torch.ones_like(probabilities[:, 0]),
                                  atol=1e-6, rtol=1e-5)):
        raise ValueError('Invalid source labels or target probability distributions')

    source_membership = F.one_hot(source_labels.long(), classes).T.to(source.dtype)
    source_counts = source_membership.sum(1)
    source_weights = source_membership / source_counts.clamp_min(1)[:, None]
    target_mass = probabilities.sum(0)
    target_weights = probabilities.T / target_mass.clamp_min(MASS_EPS)[:, None]
    target_squared_weight = target_weights.square().sum(1)
    target_ess = torch.where(target_mass > MASS_EPS,
                             target_squared_weight.clamp_min(MASS_EPS).reciprocal(), 0)
    target_counts = torch.bincount(probabilities.argmax(1), minlength=classes)
    valid = ((source_counts >= MIN_SOURCE) & (target_counts >= MIN_TARGET_TOP1)
             & (target_mass > MASS_EPS) & (target_ess >= MIN_TARGET_ESS))

    source_cov = weighted_covariances(source, source_weights)
    target_cov = weighted_covariances(target, target_weights)
    distances = (source_cov - target_cov).square().sum((1, 2)) / (4 * source.shape[1] ** 2)
    if not torch.isfinite(distances).all():
        raise FloatingPointError('Non-finite class-conditional covariance distance')
    loss = distances[valid].mean() if valid.any() else (source.sum() + target.sum()) * 0

    records = []
    for index, (ns, nt, mass, ess, supported, distance) in enumerate(zip(
            source_counts.detach().cpu().tolist(), target_counts.cpu().tolist(),
            target_mass.cpu().tolist(), target_ess.cpu().tolist(), valid.cpu().tolist(),
            distances.detach().cpu().tolist())):
        reasons = []
        if ns < MIN_SOURCE:
            reasons.append('source_count')
        if nt < MIN_TARGET_TOP1:
            reasons.append('target_top1_count')
        if mass <= MASS_EPS:
            reasons.append('target_mass')
        if ess < MIN_TARGET_ESS:
            reasons.append('target_ess')
        records.append(dict(class_index=index, source_count=int(ns), target_top1_count=nt,
                            target_mass=mass, target_ess=ess, valid=int(supported),
                            skip_reason='|'.join(reasons), local_loss=distance if supported else None))
    return loss, records
