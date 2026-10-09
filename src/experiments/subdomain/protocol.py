"""Locked five-setting AlexNet fc8 protocol, independent of legacy controls."""

from src.common.experiment_protocol import feature_protocol
from src.common.paths import RESULTS_DIR, ROOT
from src.core.subdomain_coral import MASS_EPS, MIN_SOURCE, MIN_TARGET_ESS, MIN_TARGET_TOP1

RUNS_DIR = RESULTS_DIR / 'subdomain_coral'
REPORTS_DIR = ROOT / 'reports' / 'subdomain_coral'
SETTINGS = ('no_adaptation', 'global', 'global_quarter', 'local', 'mixed')
ADAPTATION_SETTINGS = SETTINGS[1:]
FINAL_WEIGHTS = {'no_adaptation': (0.0, 0.0), 'global': (0.5, 0.0),
                 'global_quarter': (0.25, 0.0), 'local': (0.0, 0.5), 'mixed': (0.25, 0.25)}
EPOCHS = 50
WARMUP_EPOCHS = 10
BATCH_SIZE = 128
STEPS_PER_EPOCH = 6
CLASS_COUNT = 31


def weights_for_epoch(setting, epoch):
    if setting not in SETTINGS or not 1 <= epoch <= EPOCHS:
        raise ValueError('Unknown subdomain setting or epoch outside 1..50')
    if setting != 'no_adaptation' and epoch <= WARMUP_EPOCHS:
        return 0.5, 0.0
    return FINAL_WEIGHTS[setting]


def make_config(setting, class_to_idx, device):
    global_weight, local_weight = FINAL_WEIGHTS[setting]
    return dict(schema_version=1, suite='subdomain_coral', setting=setting, backbone='alexnet',
                pretrained='IMAGENET1K_V1', source='amazon', target='webcam',
                source_samples=2817, target_samples=795, num_classes=CLASS_COUNT,
                class_to_idx=class_to_idx, seed=42, seed_policy='all_rngs_42',
                source_initial_shuffle_seed=42, target_initial_shuffle_seed=42,
                evaluation_loader_seeds=[42, 42], worker_seed=42,
                epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS, batch_size=BATCH_SIZE,
                num_workers=0, train_steps_per_epoch=STEPS_PER_EPOCH, train_drop_last=True,
                batch_pairing='shorter_loader_zip', lr=0.001, classifier_lr=0.01,
                momentum=0.9, decay=0.0005, classifier_weight_std=0.005, classifier_bias=0.01,
                precision='float32', tf32=False, deterministic_algorithms=True, device=device,
                monitor_layers=['fc8'], feature_protocol=feature_protocol(('fc8',)),
                preprocessing='existing_resize_224_per_domain_mean_std',
                local_covariance='normalized_weight_finite_sample_correction',
                local_reduction='mean_valid_classes', local_zero_policy='connected_zero_no_valid_classes',
                target_probability_temperature=1.0, target_probability_detached=True,
                target_probability_forward='same_training_forward', confidence_threshold=None,
                min_source_count=MIN_SOURCE, min_target_top1_count=MIN_TARGET_TOP1,
                min_target_ess=MIN_TARGET_ESS, target_mass_epsilon=MASS_EPS,
                warmup_global_weight=0.5 if setting != 'no_adaptation' else 0.0,
                final_global_weight=global_weight, final_local_weight=local_weight,
                target_training_labels_used=False, early_stopping=False,
                selection_metric='epoch_50_target_accuracy', original_controls_retrained=True,
                results_dir=str(RUNS_DIR), method='class_conditional_coral_inspired_by_dsan_weights')
