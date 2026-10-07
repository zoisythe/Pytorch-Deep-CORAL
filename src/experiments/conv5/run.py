"""Run conv5 and conv5+fc8; controls are always read from results/original."""

from src.training.run_experiments import main

if __name__ == '__main__':
    main('conv5_fc8')
