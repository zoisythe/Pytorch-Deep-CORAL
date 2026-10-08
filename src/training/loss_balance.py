"""Stop a fixed-weight run after sustained balance of its training losses."""

import math

DEFAULT_TOLERANCE = 0.25
DEFAULT_PATIENCE = 2
DEFAULT_MIN_EPOCHS = 20
DEFAULT_EMA_DECAY = 0.5
MAX_TRAINING_TOLERANCE = 0.25


class LossBalanceStopper:
    def __init__(self, tolerance=DEFAULT_TOLERANCE, patience=DEFAULT_PATIENCE,
                 min_epochs=DEFAULT_MIN_EPOCHS, ema_decay=DEFAULT_EMA_DECAY):
        if not math.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError('balance_tolerance must be in (0, 1)')
        if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError('balance_ema_decay must be in [0, 1)')
        if patience < 1 or min_epochs < 1:
            raise ValueError('balance_patience and balance_min_epochs must be positive')
        self.tolerance, self.patience = tolerance, patience
        self.min_epochs, self.ema_decay = min_epochs, ema_decay
        self.ce_ema = self.coral_ema = None
        self.streak = self.epoch = 0

    def update(self, epoch, classification_loss, weighted_coral_loss):
        if epoch != self.epoch + 1:
            raise ValueError('Loss balance must observe consecutive epochs from 1')
        if any(not math.isfinite(value) or value < 0
               for value in (classification_loss, weighted_coral_loss)):
            raise ValueError('Loss balance requires finite, nonnegative training losses')
        if self.ce_ema is None:
            self.ce_ema, self.coral_ema = classification_loss, weighted_coral_loss
        else:
            self.ce_ema = self.ema_decay * self.ce_ema + (1 - self.ema_decay) * classification_loss
            self.coral_ema = self.ema_decay * self.coral_ema + (1 - self.ema_decay) * weighted_coral_loss
        gap = abs(self.ce_ema - self.coral_ema) / max(self.ce_ema, self.coral_ema, 1e-12)
        raw_gap = abs(classification_loss - weighted_coral_loss) / max(classification_loss, weighted_coral_loss, 1e-12)
        self.epoch = epoch
        balanced = epoch >= self.min_epochs and gap <= self.tolerance and raw_gap <= self.tolerance
        self.streak = self.streak + 1 if balanced else 0
        return {'classification_loss_ema': self.ce_ema, 'weighted_CORAL_loss_ema': self.coral_ema,
                'balance_gap': gap, 'balance_raw_gap': raw_gap, 'balance_streak': self.streak,
                'balance_stop': int(self.streak >= self.patience)}


def stopper_from_args(args):
    if args.balance_tolerance > MAX_TRAINING_TOLERANCE:
        raise ValueError('Training loss-balance tolerance cannot exceed 25%')
    return LossBalanceStopper(args.balance_tolerance, args.balance_patience,
                              args.balance_min_epochs, args.balance_ema_decay)
