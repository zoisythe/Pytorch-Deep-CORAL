"""Locked author-code protocol, distinct from the historical CORAL suite."""

import math

from src.common.paths import ROOT

RUN_DIR = ROOT / 'results' / 'resnet50_lmmd' / 'author_code_seed42'
REPORT_DIR = ROOT / 'reports' / 'resnet50_lmmd'
EPOCHS = 200
PATIENCE = 30
STEPS = 87


def schedule(epoch):
    if not 1 <= epoch <= EPOCHS:
        raise ValueError('Epoch must be in 1..200')
    decay = (1 + 10 * (epoch - 1) / EPOCHS) ** 0.75
    return 0.001 / decay, 0.01 / decay, 0.5 * (2 / (1 + math.exp(-10 * epoch / EPOCHS)) - 1)


def make_config(classes, device):
    return {'suite': 'resnet50_lmmd_author_code', 'seed': 42, 'source': 'amazon', 'target': 'webcam',
                'class_to_idx': classes, 'source_samples': 2817, 'target_samples': 795, 'num_classes': 31,
                'backbone': 'resnet50', 'pretrained_weights': 'ResNet50_Weights.IMAGENET1K_V2',
                'bottleneck': 256, 'alignment': 'bottleneck', 'new_linear_initialization': 'pytorch_default',
                'extra_activation': False, 'global_pool': 'spatial_mean', 'batch_size': 32, 'workers': 0,
                'source_loader_batches': 88, 'target_loader_batches': 24, 'steps_per_epoch': STEPS,
                'sampling': 'author_range_1_len_source_cycle_target', 'epochs': EPOCHS, 'patience': PATIENCE,
                'selection_metric': 'best_target_accuracy_earliest_tie', 'target_labels_in_loss': False,
                'target_labels_in_selection_and_stopping': True, 'momentum': 0.9, 'weight_decay': 0.0005,
                'base_lrs': [0.001, 0.01, 0.01], 'lr_schedule': 'base/(1+10*(epoch-1)/200)^0.75',
                'adaptation_schedule': '0.5*(2/(1+exp(-10*epoch/200))-1)', 'coral_warmup': False,
                'train_transform': ['Resize([256,256])', 'RandomCrop(224)', 'RandomHorizontalFlip()', 'ToTensor()'],
                'eval_transform': ['Resize([224,224])', 'ToTensor()'], 'normalization': None,
                'train_shuffle': True, 'train_drop_last': True, 'eval_shuffle': False,
                'independent_loader_generators': True, 'batchnorm': 'shared_source_then_target',
                'kernel': 'sum_5_rbf', 'bandwidth': 'detached_mean_offdiagonal_squared_distance',
                'bandwidth_floor': 1e-12, 'estimator': 'v_statistic_including_diagonal',
                'shared_classes': 'source_truth_intersect_target_argmax', 'target_weights': 'detached_softmax',
                'precision': 'float32', 'deterministic': True, 'tf32': False, 'device': device,
                'protocol': 'reports/resnet50_lmmd/protocol.md', 'original_controls_retrained': False}
