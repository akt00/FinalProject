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
    def __init__(self, **kwargs) -> None:
        super(DiceLoss, self).__init__()
        # smoothing value to avoid division by zero
        self.smoothing = kwargs.get('smoothing', 1.0)

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
        dice_coeff = (intersection + self.smoothing) / (appx_union
                                                        + self.smoothing)
        # dice_loss = 1 - dice_coeff
        return 1 - dice_coeff
# the cell ends here


class FocalLoss(nn.Module):
    mode = 'binary'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gamma = kwargs.get('gamma', 2.0)
        self.alpha = kwargs.get('alpha', 0.25)

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
    # tests the losses
    loss_fn = BCEDiceLoss()
    in_shape = [32, 1, 256, 256]
    preds = torch.randn(*in_shape).clamp(min=0, max=1)
    gt = (torch.randn(*in_shape) > .5).float()
    print(loss_fn(preds, gt))
# the cell ends here