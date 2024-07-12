"""
Loss functions.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight)
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)

        return err


class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))


class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight=None):
        return torch.mean(weight * torch.pow(pred - target, 2))


class BCELoss(BaseLoss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy(pred, target, weight=weight)


class KLDivLoss(BaseLoss):
    """
    KL divergence between a given Gaussian (with mean mu and variance var)
    and the standard Gaussian distributions
    """
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def _forward(self, mu, log_var, weight):
        # see Appendix B from VAE paper:
        # Kingma and Welling, Auto-Encoding Variational Bayes, ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return weight * (-0.5) * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
