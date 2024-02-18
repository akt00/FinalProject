import torch
from torch import Tensor


def pixel_accuracy(p: Tensor, gt: Tensor) -> Tensor:
    """ Computes Pixel Accurate

    Args:
        p: output from the model with threshold applied (b, 1, h, w)
        gt: ground truth (b, 1, h, w)
    """
    p = p[:, 0].contiguous().view(p.shape[0], -1)
    gt = gt[:, 0].contiguous().view(gt.shape[0], -1)
    return (p == gt).float().sum(dim=1) / gt.shape[1]


def iou(p: Tensor, gt: Tensor, smooth: float = 1.) -> Tensor:
    """ Computes IoU

    Args:
        p: output from the model with threshold applied (b, 1, h, w)
        gt: ground truth (b, 1, h, w)
        smooth: avoid division by zero
    """
    # flattens (batch, ch, h, w) to (batch, -1)
    p = p[:, 0].contiguous().view(p.shape[0], -1)
    gt = gt[:, 0].contiguous().view(gt.shape[0], -1)
    intersec = (p * gt).sum(1)
    union = (p + gt).sum(1) - intersec
    return (intersec + smooth) / (union + smooth)


def precision(p: Tensor, gt: Tensor, smooth: float = 1.) -> Tensor:
    """ Computes Precision

    Args:
        p: output from the model with threshold applied (b, 1, h, w)
        gt: ground truth (b, 1, h, w)
        smooth: avoid division by zero
    """
    p = p[:, 0].contiguous().view(p.shape[0], -1)
    gt = gt[:, 0].contiguous().view(gt.shape[0], -1)
    tp = (p * gt).sum(dim=1)
    fp = ((1 - gt) * p).sum(dim=1)
    return (tp + smooth) / (tp + fp + smooth)


def sensitivity(p: Tensor, gt: Tensor, smooth: float = 1.) -> Tensor:
    """ Computes true positive rate aka recall

    Args:
        p: output from the model with threshold applied (b, 1, h, w)
        gt: ground truth (b, 1, h, w)
        smooth: avoid division by zero
    """
    # flattens (batch, ch, h, w) to (batch, -1)
    p = p[:, 0].contiguous().view(p.shape[0], -1)
    gt = gt[:, 0].contiguous().view(gt.shape[0], -1)
    tp = (p * gt).sum(dim=1)
    fn = (gt * (1 - p)).sum(dim=1)
    return (tp + smooth) / (tp + fn + smooth)


def f1_score(p: Tensor, gt: Tensor, smooth: float = 1.) -> Tensor:
    """ Computes the F1 score

    Args:
        p: output from the model with threshold applied (b, 1, h, w)
        gt: ground truth (b, 1, h, w)
        smooth: avoid division by zero
    """
    precision_score = precision(p, gt, smooth=smooth)
    recall_score = sensitivity(p, gt, smooth=smooth)
    return 2 * (precision_score * recall_score) / (
        precision_score + recall_score)


def mean_ap(p: Tensor, gt: Tensor, smooth: float = 1.) -> Tensor:
    """ Computes mean Average Precision

    Args:
        p: output from the model with threshold applied (b, 1, h, w)
        gt: ground truth (b, 1, h, w)
        smooth: avoid division by zero
    """
    precisions, recalls = [], []
    # computes precision/recall for each confidence
    for conf in range(11):
        _p = (p > (conf / 10.)).float()
        _gt = (gt > (conf / 10.)).float()
        precisions.append(precision(_p, _gt, smooth).mean())
        recalls.append(sensitivity(_p, _gt, smooth).mean())
    # computes average AUC
    precisions = torch.Tensor(precisions)
    recalls = torch.Tensor(recalls)
    diffs = recalls[1:] - recalls[:-1]
    ap = torch.sum(precisions[:-1] * diffs)
    return ap


def specificity(p: Tensor, gt: Tensor, smooth: float = 1.) -> Tensor:
    """ Computes Recall

    Args:
        p: output from the model with threshold applied (b, 1, h, w)
        gt: ground truth (b, 1, h, w)
        smooth: avoid division by zero
    """
    # flattens (batch, ch, h, w) to (batch, -1)
    p = p[:, 0].contiguous().view(p.shape[0], -1)
    gt = gt[:, 0].contiguous().view(gt.shape[0], -1)
    tn = ((gt + p) == 0).sum(dim=1)
    fp = ((1 - gt) * p).sum(dim=1)
    return (tn + smooth) / (tn + fp + smooth)


def yoden_j_index(p: Tensor, gt: Tensor, smooth: float = 1.) -> Tensor:
    """ Computes Yoden's J Index

    Args:
        p: output from the model with threshold applied (b, 1, h, w)
        gt: ground truth (b, 1, h, w)
        smooth: avoid division by zero
    """
    return sensitivity(p, gt, smooth) + specificity(p, gt, smooth) - 1
