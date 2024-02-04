import torch
import torch.nn as nn
import torch.nn.functional as F


# the code in this section is originally written by Akihiro Tanimoto
class DiceLoss(nn.Module):
    """ The custom class for dice loss

    This loss can only be used for binary segmentation tasks

    Attributes:
        kwargs:
            smoothing (float): avoids division by zero. default is 1.0
    """
    def __init__(self, smooth=1.) -> None:
        super(DiceLoss, self).__init__()
        # smoothing value to avoid division by zero
        self.smooth = smooth

    def forward(self, preds, targets):
        """ returns the mean dice loss

        Attributes:
            preds: prediction results from the troch model. (b, ch, h, w)
            targets: ground truth labels. (b, ch, h, w)

        Returns:
            The computed dice loss = (1 - dice coefficient)
        """
        _preds = preds[:, 0].contiguous().view(-1)
        _targets = targets[:, 0].contiguous().view(-1)
        intersection = (_preds * _targets).sum()
        # approximation of union for computational efficiency
        appx_union = _preds.sum() + _targets.sum()
        dice_coeff = (intersection + self.smooth) / (appx_union + self.smooth)
        # dice_loss = 1 - dice_coeff
        return 1 - dice_coeff
# the cell ends here


class FocalLoss(nn.Module):
    mode = 'binary'

    def __init__(self, alpha=.25, gamma=2.) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        _preds = preds.view(-1)
        _targets = targets.view(-1)

        return self.focal_loss_logits(_preds, _targets, self.gamma, self.alpha)

    def focal_loss_logits(self, output: torch.Tensor, target: torch.Tensor,
                          gamma: float, alpha: float):
        target = target.type(output.type())
        logpt = F.binary_cross_entropy_with_logits(
            input=output, target=target, reduction='none'
        )
        pt = torch.exp(-logpt)
        focal_term = (1.0 - pt).pow(gamma)
        loss = focal_term * logpt
        loss *= alpha * target + (1 - alpha) * (1 - target)
        return loss.mean()


class FocalDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

    def forward(self, preds, targets):
        return self.focal_loss(preds, targets) + self.dice_loss(preds, targets)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, smooth=1.) -> None:
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

    def tversky(self, preds, targets):
        _preds = preds[:, 0].contiguous().view(-1)
        _targets = targets[:, 0].contiguous().view(-1)
        tp = (_preds * _targets).sum()
        fn = (_targets * (1 - _preds)).sum()
        fp = ((1 - _targets) * _preds).sum()
        return (tp + self.smooth) / (
            tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)

    def forward(self, preds, targets):
        return 1 - self.tversky(preds, targets)


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=.7, gamma=.75, smooth=1.) -> None:
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.tversky = TverskyLoss(alpha=self.alpha, smooth=self.smooth)
        self.gamma = gamma

    def forward(self, preds, targets):
        tversky_score = self.tversky(preds, targets)
        return (1 - tversky_score).pow(self.gamma)


# the code in this section is originally written by Akihiro Tanimoto
class BCEDiceLoss(nn.Module):
    """ Binary cross entropy loss with dice loss

    Not in use
    """
    def __init__(self) -> None:
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, preds, targets):
        """ forward pass for loss computation

        Attributes:
            preds: predictions from torch's model. (b, ch, h, w)
            targets: ground truth labels. (b, ch, h, w)

        Returns (torch.Tensor):
            The computated bce dice loss.
        """
        return self.bce_loss(preds, targets) + self.dice_loss(preds, targets)
# the cell ends here


# the code in this section is originally written by Akihiro Tanimoto
if __name__ == '__main__':
    import torch

    t1 = torch.randn(8, 1, 10, 10)
    t1 = (t1 > 0.5).float()
    t2 = torch.randn(8, 1, 10, 10)
    t2 = (t2 > 0.5).float()
    loss_fn = FocalTverskyLoss()
    print(loss_fn(t1, t2))
# the cell ends here
