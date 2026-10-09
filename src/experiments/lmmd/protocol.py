"""One LMMD control; the original five subdomain controls remain frozen."""

from src.core.lmmd import BANDWIDTH_FLOOR, KERNEL_MUL, KERNEL_NUM
from src.experiments.subdomain.protocol import REPORTS_DIR as SUBDOMAIN_REPORTS
from src.experiments.subdomain.protocol import RUNS_DIR, make_config as coral_config

RUN_DIR = RUNS_DIR / 'lmmd'
REPORTS_DIR = SUBDOMAIN_REPORTS / 'lmmd'


def weights_for_epoch(epoch):
    if not 1 <= epoch <= 50:
        raise ValueError('LMMD epoch must be in 1..50')
    return (0.5, 0.0) if epoch <= 10 else (0.0, 0.5)


def make_config(class_to_idx, device):
    config = coral_config('local', class_to_idx, device)
    config.update(suite='lmmd_control', setting='lmmd', method='dsan_lmmd_on_alexnet_fc8',
                  final_local_weight=0.0, final_lmmd_weight=0.5, original_controls_retrained=False,
                  local_covariance_role='monitoring_only', lmmd_kernel='sum_rbf',
                  lmmd_kernel_num=KERNEL_NUM, lmmd_kernel_mul=KERNEL_MUL,
                  lmmd_bandwidth='detached_mean_offdiagonal_squared_distance',
                  lmmd_bandwidth_floor=BANDWIDTH_FLOOR, lmmd_fixed_sigma=None,
                  lmmd_estimator='biased_v_statistic_including_diagonal',
                  lmmd_min_source_count=1, lmmd_min_target_top1_count=1,
                  lmmd_min_target_ess=None, lmmd_reduction='mean_shared_classes',
                  lmmd_zero_policy='connected_zero_no_shared_classes',
                  controls_dir=str(RUNS_DIR), protocol='reports/subdomain_coral/lmmd/protocol.md')
    return config
