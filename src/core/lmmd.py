"""DSAN-style LMMD: detached class weights and a sum of five RBF kernels."""

import torch
import torch.nn.functional as F

KERNEL_NUM = 5
KERNEL_MUL = 2.0
BANDWIDTH_FLOOR = 1e-12


def lmmd(source, target, source_labels, target_probabilities):
    """Return (scalar, class records, bandwidth diagnostics), without target truth.

    Shared source-label/target-argmax classes use all target soft probabilities.
    Includes within-domain diagonals, matching the author's biased V-statistic.
    """
    if (source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]
            or min(source.shape) < 1 or min(target.shape) < 1):
        raise ValueError('LMMD requires nonempty [N, D] features with the same D')
    if (source.dtype != target.dtype or source.device != target.device
            or not source.is_floating_point() or not target.is_floating_point()):
        raise ValueError('LMMD features must share a floating dtype and device')
    if (source_labels.shape != (len(source),) or source_labels.device != source.device
            or source_labels.dtype not in (torch.int32, torch.int64)):
        raise ValueError('Source labels must be integer [Ns] on the feature device')
    if (target_probabilities.ndim != 2 or len(target_probabilities) != len(target)
            or target_probabilities.shape[1] < 1 or target_probabilities.device != target.device
            or not target_probabilities.is_floating_point()):
        raise ValueError('Target probabilities must be floating [Nt, C] on the feature device')
    probabilities = target_probabilities.detach().to(source.dtype)
    classes = probabilities.shape[1]
    if any(not torch.isfinite(value).all() for value in (source, target, probabilities)):
        raise FloatingPointError('Non-finite LMMD inputs')
    if (torch.any(source_labels < 0) or torch.any(source_labels >= classes)
            or torch.any(probabilities < 0)
            or not torch.allclose(probabilities.sum(1), torch.ones_like(probabilities[:, 0]),
                                  atol=1e-6, rtol=1e-5)):
        raise ValueError('Invalid LMMD labels or probabilities')
    membership = F.one_hot(source_labels.long(), classes).T.to(source.dtype)
    source_counts = membership.sum(1)
    source_weights = membership / source_counts.clamp_min(1)[:, None]
    target_counts = torch.bincount(probabilities.argmax(1), minlength=classes)
    target_mass = probabilities.sum(0)
    denominator = torch.where(target_mass > 0, target_mass, torch.ones_like(target_mass))
    target_weights = probabilities.T / denominator[:, None]
    target_ess = torch.where(target_mass > 0, target_weights.square().sum(1).clamp_min(1e-30).reciprocal(), 0)
    valid = (source_counts > 0) & (target_counts > 0)

    combined = torch.cat((source, target))
    squared_distances = (combined.unsqueeze(0) - combined.unsqueeze(1)).square().sum(2)
    raw_bandwidth = squared_distances.detach().sum() / (len(combined) ** 2 - len(combined))
    bandwidth = raw_bandwidth.clamp_min(BANDWIDTH_FLOOR)
    base = bandwidth / KERNEL_MUL ** (KERNEL_NUM // 2)
    kernels = sum(torch.exp(-squared_distances / (base * KERNEL_MUL ** index)) for index in range(KERNEL_NUM))
    ss, tt, st = kernels[:len(source), :len(source)], kernels[len(source):, len(source):], kernels[:len(source), len(source):]
    count = int(valid.sum().item())
    if count:
        sw, tw = source_weights[valid], target_weights[valid]
        weight_ss, weight_tt, weight_st = sw.T @ sw / count, tw.T @ tw / count, sw.T @ tw / count
        if len(source) == len(target):
            loss = (weight_ss * ss + weight_tt * tt - 2 * weight_st * st).sum()
        else:
            loss = (weight_ss * ss).sum() + (weight_tt * tt).sum() - 2 * (weight_st * st).sum()
    else:
        loss = (source.sum() + target.sum()) * 0
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite LMMD loss')
    with torch.no_grad():
        distances = ((source_weights @ ss) * source_weights).sum(1)
        distances += ((target_weights @ tt) * target_weights).sum(1)
        distances -= 2 * ((source_weights @ st) * target_weights).sum(1)
    if not torch.isfinite(distances).all():
        raise FloatingPointError('Non-finite per-class LMMD')
    records = []
    for index in range(classes):
        ns, nt = int(source_counts[index].item()), int(target_counts[index].item())
        reasons = [name for condition, name in [(ns == 0, 'source_count'), (nt == 0, 'target_top1_count')] if condition]
        records.append(dict(class_index=index, source_count=ns, target_top1_count=nt,
                            target_mass=target_mass[index].item(), target_ess=target_ess[index].item(),
                            valid=int(not reasons), skip_reason='|'.join(reasons),
                            lmmd_loss=distances[index].item() if not reasons else None))
    diagnostics = dict(lmmd_bandwidth_raw=raw_bandwidth.item(), lmmd_bandwidth=bandwidth.item(),
                       lmmd_bandwidth_clamped=int(raw_bandwidth.item() < BANDWIDTH_FLOOR))
    return loss, records, diagnostics
