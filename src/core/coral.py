"""CORAL for image vectors (Gram) and local convolutional channels (covariance)."""


def _center(features):
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError('CORAL requires a (batch >= 2, features >= 1) tensor')
    return features - features.mean(dim=0, keepdim=True)


def coral(source, target):
    if source.ndim != target.ndim or source.ndim not in (2, 4):
        raise ValueError('CORAL requires two 2D vectors or two 4D feature maps')
    if source.shape[1] != target.shape[1]:
        raise ValueError('Source and target must have the same feature/channel dimension')
    if source.ndim == 4:
        source_covariance = compute_channel_covariance(source)
        target_covariance = compute_channel_covariance(target)
        channels = source.shape[1]
        return (source_covariance - target_covariance).square().sum() / (4 * channels * channels)
    source, target = _center(source), _center(target)
    ns, d = source.shape
    nt = target.shape[0]
    # ||Xs^T Xs/(ns-1) - Xt^T Xt/(nt-1)||_F^2, by the trace identity.
    source_term = (source @ source.T).square().sum() / (ns - 1) ** 2
    target_term = (target @ target.T).square().sum() / (nt - 1) ** 2
    cross_term = (source @ target.T).square().sum() / ((ns - 1) * (nt - 1))
    # Roundoff can make an exactly zero distance slightly negative.
    return (source_term + target_term - 2 * cross_term).clamp_min(0) / (4 * d * d)


def compute_covariance(input_data):
    """Unbiased sample covariance; also serves as the direct reference formula."""
    centered = _center(input_data)
    return centered.T @ centered / (input_data.shape[0] - 1)


def compute_channel_covariance(features):
    """Sample-covariance convention over B*H*W local observations, not images.

    Spatial observations can be correlated. This divisor does not imply an
    unbiased estimate of a population covariance under independent sampling.
    Only a C*C matrix is constructed, never a (B*H*W)*(B*H*W) Gram matrix.
    """
    if features.ndim != 4 or any(size < 1 for size in features.shape):
        raise ValueError('Channel covariance requires a nonempty (B, C, H, W) tensor')
    observations = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    return compute_covariance(observations)
