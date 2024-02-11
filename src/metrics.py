import torch
from torch import Tensor


def pixel_accuracy(p: Tensor, gt: Tensor) -> Tensor:
    p = p[:, 0].contiguous().view(p.shape[0], -1)
    gt = gt[:, 0].contiguous().view(gt.shape[0], -1)
    return (p == gt).float().sum(dim=1) / gt.shape[1]


# the code in this section is originally written by Akihiro Tanimoto
def iou(p: torch.Tensor, gt: torch.Tensor, smooth: float = 1.):
    """ Computes IoU

    This function can compute IoU only for binary segmentation

    Args:
        p (torch.Tensor): the output from the model with threshold applied.
                            (b, ch, h, w)
        gt (torch.Tensor): the ground truth. (b, ch, h, w)
        smooth (float): applies smooth to avoid division by zero
    """
    # flattens (batch, ch, h, w) to (batch, -1)
    p = p[:, 0].contiguous().view(p.shape[0], -1)
    gt = gt[:, 0].contiguous().view(gt.shape[0], -1)
    intersec = (p * gt).sum(1)
    union = (p + gt).sum(1) - intersec
    return (intersec + smooth) / (union + smooth)
# the section ends here


def precision(p: torch.Tensor, gt: torch.Tensor, smooth: float = 1.):
    p = p[:, 0].contiguous().view(p.shape[0], -1)
    gt = gt[:, 0].contiguous().view(gt.shape[0], -1)
    tp = (p * gt).sum(dim=1)
    fp = ((1 - gt) * p).sum(dim=1)
    return (tp + smooth) / (tp + fp + smooth)


# the code in this section is originally written by Akihiro Tanimoto
def sensitivity(p: torch.Tensor, gt: torch.Tensor, smooth: float = 1.):
    """ Computes true positive rate aka recall

    This function can compute sensitivity only for binary segmentation

    Args:
        p: the output from the model with threshold applied. (b, ch, h, w)
        gt: the ground truth. (b, ch, h, w)
        smooth: applies smooth to avoid division by zero
    """
    # flattens (batch, ch, h, w) to (batch, -1)
    p = p[:, 0].contiguous().view(p.shape[0], -1)
    gt = gt[:, 0].contiguous().view(gt.shape[0], -1)
    tp = (p * gt).sum(dim=1)
    fn = (gt * (1 - p)).sum(dim=1)
    return (tp + smooth) / (tp + fn + smooth)
# the section ends here


def f1_score(p: torch.Tensor, gt: torch.Tensor, smooth: float = 1.):
    """ Computes the F1 score
    F1 = 2 * (precision * recall) / (precision + recall)
    This is the same as TP / (TP + (FP + FN) / 2)

    """
    precision_score = precision(p, gt, smooth=smooth)
    recall_score = sensitivity(p, gt, smooth=smooth)
    return 2 * (precision_score * recall_score) / (
        precision_score + recall_score)


def mean_ap(p: torch.Tensor, gt: torch.Tensor, smooth: float = 1.):
    precisions, recalls = [], []
    for conf in range(11):
        _p = (p > (conf / 10.)).float()
        _gt = (gt > (conf / 10.)).float()
        # print(_p)
        precisions.append(precision(_p, _gt, smooth).mean())
        recalls.append(sensitivity(_p, _gt, smooth).mean())

    precisions = torch.Tensor(precisions)
    recalls = torch.Tensor(recalls)
    diffs = recalls[1:] - recalls[:-1]

    ap = torch.sum(precisions[:-1] * diffs)
    # print(f'precisions: {precisions}')
    # print(f'recalls: {recalls}')
    return ap


# the code in this section is originally written by Akihiro Tanimoto
def specificity(p, gt, smooth: float = 1.):
    """ computes true negative rate
    this function can compute specificity only for binary segmentation

    Args:
        p (torch.Tensor): the output from the model with threshold applied.
                            (b, ch, h, w)
        gt (torch.Tensor): the ground truth. (b, ch, h, w)
        smooth (float): applies smooth to avoid division by zero
    """
    # flattens (batch, ch, h, w) to (batch, -1)
    p = p[:, 0].contiguous().view(p.shape[0], -1)
    gt = gt[:, 0].contiguous().view(gt.shape[0], -1)
    tn = ((gt + p) == 0).sum(dim=1)
    fp = ((1 - gt) * p).sum(dim=1)
    return (tn + smooth) / (tn + fp + smooth)
# the section ends here


def yoden_j_index(p, gt, smooth: float = 1.):
    return sensitivity(p, gt, smooth) + specificity(p, gt, smooth) - 1
