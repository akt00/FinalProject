import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiceLoss(nn.Module):
    """ Dice Loss for binary segmentation """
    def __init__(self, smooth: float = 1.) -> None:
        super(DiceLoss, self).__init__()
        # avoid division by zero
        self.smooth = smooth

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """ forward pass """
        _preds = preds[:, 0].contiguous().view(-1)
        _targets = targets[:, 0].contiguous().view(-1)
        intersection = (_preds * _targets).sum()
        # approximation of union for computational efficiency
        appx_union = _preds.sum() + _targets.sum()
        dice_coeff = (2 * intersection + self.smooth) / (
            appx_union + self.smooth)
        # dice_loss = 1 - dice_coeff
        return 1 - dice_coeff


class FocalLoss(nn.Module):
    """ Focal Loss for imbanced classes """
    mode = 'binary'

    def __init__(self, alpha: float = .25, gamma: float = 2.) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """ forward pass """
        targets = targets.type(preds.type())
        logpt = F.binary_cross_entropy(
            input=preds, target=targets, reduction='none'
        )
        pt = torch.exp(-logpt)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_factor * (1 - pt) ** self.gamma * logpt
        return loss.mean()


class FocalDiceLoss(nn.Module):
    """ Combination of Focal Loss and Dice Loss"""
    def __init__(self, weight: float = 2) -> None:
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.weight = weight

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """ forward pass """
        return self.focal_loss(preds, targets) * self.weight + self.dice_loss(preds, targets)


class TverskyLoss(nn.Module):
    """ Tversky Loss for penalizing FP/FN """
    def __init__(self, alpha: float = .7, smooth: float = 1.) -> None:
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def tversky(self, preds: Tensor, targets: Tensor) -> Tensor:
        _preds = preds[:, 0].contiguous().view(-1)
        _targets = targets[:, 0].contiguous().view(-1)
        tp = (_preds * _targets).sum()
        fn = (_targets * (1 - _preds)).sum()
        fp = ((1 - _targets) * _preds).sum()
        return (tp + self.smooth) / (
            tp + self.alpha * fp + (1 - self.alpha) * fn + self.smooth)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """ forward pass """
        return 1 - self.tversky(preds, targets)


class FocalTverskyLoss(nn.Module):
    """ Focal Tversky Loss for small lesion segmentation """
    def __init__(self, alpha: float = .3, gamma: float = .75,
                 smooth: float = 1.) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.tversky = TverskyLoss(alpha=self.alpha, smooth=self.smooth)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """ forward pass """
        tversky_score = self.tversky(preds, targets)
        return (1 - tversky_score).pow(self.gamma)


class BCEDiceLoss(nn.Module):
    """ Binary cross entropy loss with dice loss """
    def __init__(self) -> None:
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """ forward pass """
        return self.bce_loss(preds, targets) + self.dice_loss(preds, targets)
