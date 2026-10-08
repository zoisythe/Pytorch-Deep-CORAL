"""Fixed lambda=0.5 with a target-label-free loss-balance stopping signal."""

from src.training.train import make_parser, run
from src.common.paths import RESULTS_DIR


def main():
    parser = make_parser()
    parser.description = 'Train fixed CORAL until sustained training-loss balance'
    parser.set_defaults(suite='conv5_fc8', coral_layers=['conv5', 'fc8'], epochs=100,
                        lambda_coral=0.5, stop_on_loss_balance=True,
                        results_dir=RESULTS_DIR / 'loss_balance')
    args = parser.parse_args()
    if args.suite == 'original':
        parser.error('Loss-balance runs must preserve the frozen original controls')
    try:
        directory = run(args)
        print(f'Fixed-weight early-stop data saved to {directory}', flush=True)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
