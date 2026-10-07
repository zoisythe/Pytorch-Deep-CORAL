"""Shared definitions for the existing FC suite and the conv5/fc8 suite."""

from src.core.model import CORAL_LAYERS, FC_LAYERS, LAYER_INDICES

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
}


def feature_protocol(layers):
    """Metadata describes every monitored layer, including inactive monitors."""
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
