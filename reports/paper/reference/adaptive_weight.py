"""Learn a positive CORAL weight without minimizing it through the model loss."""

import math

import torch


def gradient_statistics(classification, alignment, parameters, epsilon=1e-12):
    """Measure both gradients on the same shared parameters, without filling .grad."""
    parameters = tuple(parameters)
    ce_gradients = torch.autograd.grad(classification, parameters, retain_graph=True)
    coral_gradients = torch.autograd.grad(alignment, parameters, retain_graph=True)
    ce_squared = sum(gradient.detach().square().sum() for gradient in ce_gradients)
    coral_squared = sum(gradient.detach().square().sum() for gradient in coral_gradients)
    dot = sum((left.detach() * right.detach()).sum()
              for left, right in zip(ce_gradients, coral_gradients))
    ce_norm, coral_norm = ce_squared.sqrt(), coral_squared.sqrt()
    cosine = (dot / (ce_norm * coral_norm).clamp_min(epsilon)).clamp(-1, 1)
    return ce_norm.detach(), coral_norm.detach(), cosine.detach()


class GradientBalancedWeight:
    def __init__(self, initial=0.5, lr=0.01, ema_decay=0.9,
                 minimum=1e-4, maximum=100.0, device='cpu'):
        values = (initial, lr, ema_decay, minimum, maximum)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Adaptive weight hyperparameters must be finite')
        if not 0 < minimum <= initial <= maximum or lr <= 0 or not 0 <= ema_decay < 1:
            raise ValueError('Invalid adaptive weight hyperparameters')
        self.log_weight = torch.nn.Parameter(torch.tensor(math.log(initial), device=device))
        self.optimizer = torch.optim.Adam([self.log_weight], lr=lr, weight_decay=0)
        self.ema_decay = ema_decay
        self.log_bounds = (math.log(minimum), math.log(maximum))
        self.ce_ema = None
        self.coral_ema = None

    def value(self):
        # Model backpropagation must never optimize the weight using CE + lambda*C.
        return self.log_weight.exp().detach()

    def update(self, ce_norm, coral_norm, learn=True):
        ce_norm, coral_norm = ce_norm.detach(), coral_norm.detach()
        if self.ce_ema is None:
            self.ce_ema, self.coral_ema = ce_norm.clone(), coral_norm.clone()
        else:
            self.ce_ema = self.ema_decay * self.ce_ema + (1 - self.ema_decay) * ce_norm
            self.coral_ema = self.ema_decay * self.coral_ema + (1 - self.ema_decay) * coral_norm
        weighted_norm = self.log_weight.exp() * self.coral_ema
        objective = (weighted_norm - self.ce_ema).square()
        if not torch.isfinite(objective):
            raise FloatingPointError('Non-finite adaptive weight objective')
        if learn:
            self.optimizer.zero_grad(set_to_none=True)
            objective.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.log_weight.clamp_(*self.log_bounds)
        return {
            'lambda_next': self.value().item(),
            'gradient_CE_ema': self.ce_ema.item(),
            'gradient_CORAL_ema': self.coral_ema.item(),
            'weighted_gradient_CORAL_ema': weighted_norm.detach().item(),
            'weight_objective': objective.detach().item(),
        }
