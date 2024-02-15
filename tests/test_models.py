import torch
from src.models import (SEModule, SPPPooling, SPPModule)


def test_semodule():
    in_shape = (8, 96, 16, 15)
    x = torch.randn(*in_shape)
    model = SEModule(96)
    out = model(x)
    assert out.shape == in_shape


def test_spppooling():
    in_shape = (8, 512, 16, 16)
    x = torch.randn(*in_shape)
    model = SPPPooling(512, 256)
    out = model(x)
    assert out.shape == (8, 256, 16, 16)


def test_sppmodule():
    in_shape = (8, 512, 16, 16)
    x = torch.randn(*in_shape)
    model = SPPModule(512)
    out = model(x)
    assert out.shape == (8, 256, 16, 16)
