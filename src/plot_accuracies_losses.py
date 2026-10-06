import matplotlib

matplotlib.use('Agg')  # figures are saved to disk, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch

from paths import LOGS_DIR, PLOTS_DIR


def main():
    path_adapt = LOGS_DIR / 'log_adaptation.pth'
    path_no_adapt = LOGS_DIR / 'log_no_adaptation.pth'

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load logs
    log_adapt = torch.load(path_adapt)
    log_no_adapt = torch.load(path_no_adapt)

    # compute mean for each epoch
    adapt = {
        'classification_loss': torch.FloatTensor(log_adapt['classification_loss']).mean(dim=1).numpy(),
        'coral_loss': torch.FloatTensor(log_adapt['CORAL_loss']).mean(dim=1).numpy(),
        'source_accuracy': torch.FloatTensor(log_adapt['source_accuracy']).mean(dim=1).numpy(),
        'target_accuracy': torch.FloatTensor(log_adapt['target_accuracy']).mean(dim=1).numpy()
    }

    no_adapt = {
        'classification_loss': torch.FloatTensor(log_no_adapt['classification_loss']).mean(dim=1).numpy(),
        'coral_loss': torch.FloatTensor(log_no_adapt['CORAL_loss']).mean(dim=1).numpy(),
        'source_accuracy': torch.FloatTensor(log_no_adapt['source_accuracy']).mean(dim=1).numpy(),
        'target_accuracy': torch.FloatTensor(log_no_adapt['target_accuracy']).mean(dim=1).numpy()
    }

    # Add the first 0 value
    adapt['target_accuracy'] = np.insert(adapt['target_accuracy'], 0, 0)
    no_adapt['target_accuracy'] = np.insert(no_adapt['target_accuracy'], 0, 0)
    adapt['source_accuracy'] = np.insert(adapt['source_accuracy'], 0, 0)
    no_adapt['source_accuracy'] = np.insert(no_adapt['source_accuracy'], 0, 0)

    plt.gca().set_prop_cycle(color=['blue', 'green', 'red', 'm'])

    axes = plt.gca()
    axes.set_ylim([0, 1.1])  # ty: ignore[invalid-argument-type]

    l1, = plt.plot(adapt['target_accuracy'], label="test acc. w/ coral loss", marker='*')
    l2, = plt.plot(no_adapt['target_accuracy'], label="test acc. w/o coral loss", marker='.')

    l3, = plt.plot(adapt['source_accuracy'], label="training acc. w/ coral loss", marker='^')
    l4, = plt.plot(no_adapt['source_accuracy'], label="training acc. w/o coral loss", marker='+')

    plt.legend(handles=[l1, l2, l3, l4], loc=4)

    fig_acc = plt.gcf()
    plt.figure()

    fig_acc.savefig(PLOTS_DIR / 'accuracies.pdf', dpi=1000)

    # Classification loss and CORAL loss for training w/ CORAL loss
    plt.gca().set_prop_cycle(color=['red', 'blue'])

    axes = plt.gca()
    axes.set_ylim([0, 0.5])  # ty: ignore[invalid-argument-type]

    l5, = plt.plot(adapt['classification_loss'], label="classification loss", marker='.')
    l6, = plt.plot(adapt['coral_loss'], label="coral loss", marker='*')

    plt.legend(handles=[l5, l6], loc=1)

    fig_acc = plt.gcf()
    plt.figure()
    fig_acc.savefig(PLOTS_DIR / 'losses_adapt.pdf', dpi=1000)

    # CORAL distance for training w/o CORAL loss (lambda = 0)
    plt.gca().set_prop_cycle(color=['blue'])
    l7, = plt.plot(no_adapt['coral_loss'], label="distance w/o coral loss", marker='.')

    plt.legend(handles=[l7], loc=2)

    fig_acc = plt.gcf()
    plt.figure()
    fig_acc.savefig(PLOTS_DIR / 'coral_loss_no_adapt.pdf', dpi=1000)

    # Classification loss and CORAL loss for training w/ CORAL loss
    plt.gca().set_prop_cycle(color=['red', 'blue'])

    axes = plt.gca()
    axes.set_ylim([0, 0.5])  # ty: ignore[invalid-argument-type]

    l8, = plt.plot(adapt['classification_loss'], label="classification loss w/ coral loss", marker='.')
    l9, = plt.plot(no_adapt['classification_loss'], label="classification loss w/o coral loss", marker='*')

    plt.legend(handles=[l8, l9], loc=1)

    fig_acc = plt.gcf()
    plt.figure()
    fig_acc.savefig(PLOTS_DIR / 'classification_loss.pdf', dpi=1000)


if __name__ == '__main__':
    main()
