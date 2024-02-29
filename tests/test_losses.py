import pytest
import torch
# local module
from src.losses import DiceLoss


def test_diceloss():
    pred = torch.Tensor([[1, 1, 1, 0, 0]]).unsqueeze(dim=0)
    assert pred.shape == (1, 1, 5)
    target = torch.Tensor([[0, 0, 1, 1, 1]]).unsqueeze(dim=0)
    assert target.shape == (1, 1, 5)
    loss = DiceLoss(smooth=0)
    val = loss(pred, target)
    assert val.item() == pytest.approx((1 - 2 / 6))
