from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = ROOT / 'datasets'
RESULTS_DIR = ROOT / 'results'
ORIGINAL_DIR = RESULTS_DIR / 'original'
EXPERIMENTS_DIR = RESULTS_DIR / 'fc_layers'
CONV_EXPERIMENTS_DIR = RESULTS_DIR / 'conv5'
SEED = 42
CONTROL_SETTINGS = ('no_adaptation', 'fc8')


def suite_results_dir(suite):
    if suite == 'original':
        return ORIGINAL_DIR
    return CONV_EXPERIMENTS_DIR if suite == 'conv5_fc8' else EXPERIMENTS_DIR


def experiment_dir(results_dir, layers, seed, lambda_coral):
    if seed != SEED:
        raise ValueError('All experiments require seed=42')
    setting = '_'.join(layers) if lambda_coral > 0 else 'no_adaptation'
    return Path(results_dir) / setting


def setting_dir(results_dir, setting):
    return ORIGINAL_DIR / setting if setting in CONTROL_SETTINGS else Path(results_dir) / setting
