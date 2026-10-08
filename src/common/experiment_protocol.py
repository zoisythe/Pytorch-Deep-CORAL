"""Shared definitions for the existing FC suite and the conv5/fc8 suite."""

from src.core.model import CORAL_LAYERS, FC_LAYERS, LAYER_INDICES

RESNET_LAYERS = ('layer4', 'pooled', 'logits')
ALL_LAYERS = (*CORAL_LAYERS, *RESNET_LAYERS)
RESNET_DEFAULTS = dict(source='amazon', target='webcam', epochs=100, batch_size=32,
                       num_workers=0, seed=42, lr=0.001, decay=0.0005, momentum=0.9,
                       balance_tolerance=0.25, balance_patience=2,
                       balance_min_epochs=20, balance_ema_decay=0.5)

SUITES = {
    'original': {
        'monitor_layers': CORAL_LAYERS,
        'settings': (('fc8',),),
    },
    'fc_layers': {
        'monitor_layers': FC_LAYERS,
        'settings': (('fc6',), ('fc7',), ('fc8',), ('fc6', 'fc7'),
                     ('fc6', 'fc8'), ('fc7', 'fc8'), FC_LAYERS),
    },
    'conv5_fc8': {
        'monitor_layers': ('conv5', 'fc8'),
        'settings': (('fc8',), ('conv5',), ('conv5', 'fc8')),
    },
    'resnet50': {
        'monitor_layers': RESNET_LAYERS,
        'settings': (('logits',), ('pooled',), ('pooled', 'logits'),
                     ('layer4',), ('layer4', 'logits')),
    },
}


def validate_resnet_protocol(config):
    expected = dict(RESNET_DEFAULTS, schema_version=6, suite='resnet50', backbone='resnet50',
                    pretrained='IMAGENET1K_V2', bn_policy='shared_source_then_target', bn_momentum=0.1,
                    pooling='spatial_mean', precision='float32', activation_checkpointing=False,
                    classifier_weight_std=0.005, classifier_bias=0.01, classifier_lr_multiplier=10,
                    train_steps_per_epoch=24, train_drop_last=True, source_samples=2817, target_samples=795,
                    source_initial_shuffle_seed=42, target_initial_shuffle_seed=42, worker_seed=42,
                    evaluation_loader_seeds=[42, 42], seed_policy='all_rngs_42', original_controls=None,
                    monitor_layers=list(RESNET_LAYERS), feature_protocol=feature_protocol(RESNET_LAYERS),
                    coral_reduction='mean')
    differences = [key for key, value in expected.items() if config.get(key) != value]
    if differences:
        raise ValueError(f'Incorrect ResNet protocol: {differences}')
    active = tuple(config['coral_layers'])
    if config['lambda_coral'] == 0:
        if active or config['setting'] != 'no_coral' or config['stop_on_loss_balance']:
            raise ValueError('ResNet baseline must have no CORAL and no early stop')
    elif (config['lambda_coral'] != 0.5 or active not in SUITES['resnet50']['settings']
          or config['setting'] != '_'.join(active) or not config['stop_on_loss_balance']):
        raise ValueError('Incorrect ResNet CORAL placement/weight/stopping protocol')
    weights = {layer: config['lambda_coral'] / len(active) if layer in active else 0.0
               for layer in RESNET_LAYERS}
    if config.get('effective_coral_weights') != weights:
        raise ValueError('Incorrect ResNet effective CORAL weights')


def feature_protocol(layers):
    """Metadata describes every monitored layer, including inactive monitors."""
    if set(layers).issubset(RESNET_LAYERS):
        return {
            layer: {
                'location': {'layer4': 'layer4.output_after_residual_add_and_relu',
                             'pooled': 'layer4.spatial_mean', 'logits': 'fc.linear_output'}[layer],
                'layout': 'B,C,H,W' if layer == 'layer4' else 'B,D',
                'observation_layout': '(B*H*W),C' if layer == 'layer4' else 'B,D',
                'covariance': 'centered_channel_covariance' if layer == 'layer4' else 'centered_sample_gram',
                'covariance_divisor': 'B*H*W-1' if layer == 'layer4' else 'B-1',
                'loss_divisor': '4*C^2' if layer == 'layer4' else '4*D^2',
                'dimension': 'num_classes' if layer == 'logits' else 2048,
                **({'spatial_shape_at_224': [7, 7]} if layer == 'layer4' else {}),
            } for layer in layers
        }
    return {
        layer: {
            'location': 'features[10].output_before_relu_and_maxpool' if layer == 'conv5'
                        else f'classifier[{LAYER_INDICES[layer]}].linear_output',
            'layout': 'B,C,H,W' if layer == 'conv5' else 'B,D',
            'observation_layout': '(B*H*W),C' if layer == 'conv5' else 'B,D',
            'covariance': 'centered_channel_covariance' if layer == 'conv5' else 'centered_sample_gram',
            'covariance_divisor': 'B*H*W-1' if layer == 'conv5' else 'B-1',
            'loss_divisor': '4*C^2' if layer == 'conv5' else '4*D^2',
            'dimension': {'conv5': 256, 'fc6': 4096, 'fc7': 4096, 'fc8': 'num_classes'}[layer],
            **({'spatial_shape_at_224': [13, 13]} if layer == 'conv5' else {}),
        }
        for layer in layers
    }
