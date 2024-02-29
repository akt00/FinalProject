import pytest
import torch
# local module
from src.metrics import iou, precision, sensitivity


def test_iou():
    pred = torch.Tensor([[1, 1, 1, 0, 0]]).unsqueeze(dim=0)
    assert pred.shape == (1, 1, 5)
    target = torch.Tensor([[0, 0, 1, 1, 1]]).unsqueeze(dim=0)
    assert target.shape == (1, 1, 5)
    val = iou(pred, target, smooth=0)
    assert val.item() == pytest.approx(1 / 5)


def test_precision():
    pred = torch.Tensor([[0, 1, 1, 1, 0]]).unsqueeze(dim=0)
    assert pred.shape == (1, 1, 5)
    target = torch.Tensor([[0, 0, 1, 1, 1]]).unsqueeze(dim=0)
    assert target.shape == (1, 1, 5)
    val = precision(pred, target, smooth=0)
    assert val.item() == pytest.approx(2 / 3)


def test_recall():
    pred = torch.Tensor([[0, 1, 1, 1, 0]]).unsqueeze(dim=0)
    assert pred.shape == (1, 1, 5)
    target = torch.Tensor([[1, 0, 1, 1, 1]]).unsqueeze(dim=0)
    assert target.shape == (1, 1, 5)
    val = sensitivity(pred, target, smooth=0)
    assert val.item() == pytest.approx(2 / 4)
