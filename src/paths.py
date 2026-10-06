from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = ROOT / 'datasets'
RESULTS_DIR = ROOT / 'results'
PLOTS_DIR = RESULTS_DIR / 'plots'
CHECKPOINTS_DIR = RESULTS_DIR / 'checkpoints'
LOGS_DIR = RESULTS_DIR / 'logs'
