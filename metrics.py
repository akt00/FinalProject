import torch


# the code in this section is originally written by Akihiro Tanimoto
def iou(p: torch.Tensor, gt: torch.Tensor, smoothing=1.):
    """ computes IoU
    this function can compute IoU only for binary segmentation

    Args:
        p (torch.Tensor): the output from the model with threshold applied.
                            (b, ch, h, w)
        gt (torch.Tensor): the ground truth. (b, ch, h, w)
        smoothing (float): applies smoothing to avoid division by zero
    """
    # flattens (batch, ch, h, w) to (batch, -1)
    p = p[:, 0].view(p.shape[0], -1)
    gt = gt[:, 0].view(gt.shape[0], -1)
    intersec = (p * gt).sum(1)
    union = (p + gt).sum(1) - intersec
    return (intersec + smoothing) / (union + smoothing)
# the sectino ends here


# the code in this section is originally written by Akihiro Tanimoto
def sensitivity(p, gt, smoothing=1.):
    """ computes true positive rate
    this function can compute sensitivity only for binary segmentation

    Args:
        p (torch.Tensor): the output from the model with threshold applied.
                            (b, ch, h, w)
        gt (torch.Tensor): the ground truth. (b, ch, h, w)
        smoothing (float): applies smoothing to avoid division by zero
    """
    # flattens (batch, ch, h, w) to (batch, -1)
    p = p[:, 0].view(p.shape[0], -1)
    gt = gt[:, 0].view(gt.shape[0], -1)
    tp = (p * gt).sum(dim=1)
    fn = (gt - p).clamp(min=0).sum(dim=1)
    return (tp + smoothing) / (tp + fn + smoothing)
# the section ends here


# the code in this section is originally written by Akihiro Tanimoto
def specificity(p, gt, smoothing=1.):
    """ computes true negative rate
    this function can compute specificity only for binary segmentation

    Args:
        p (torch.Tensor): the output from the model with threshold applied.
                            (b, ch, h, w)
        gt (torch.Tensor): the ground truth. (b, ch, h, w)
        smoothing (float): applies smoothing to avoid division by zero
    """
    # flattens (batch, ch, h, w) to (batch, -1)
    p = p[:, 0].view(p.shape[0], -1)
    gt = gt[:, 0].view(gt.shape[0], -1)
    tn = ((gt + p) == 0).sum(dim=1)
    fp = (p - gt).clamp(min=0).sum(dim=1)
    return (tn + smoothing) / (tn + fp + smoothing)
# the section ends here


# the code in this section is originally written by Akihiro Tanimoto
if __name__ == '__main__':
    # print-debugs the metric functions
    in_shape = [3, 1, 5]
    pred = torch.randn(*in_shape)
    target = torch.randn(*in_shape)
    pred = (pred > .5).float()
    target = (target > .5).float()
    print(pred)
    print(target)
    print(iou(pred, target))
    print(sensitivity(pred, target))
    print(specificity(pred, target))
# the section ends here
