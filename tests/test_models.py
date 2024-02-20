import torch
from src.models import (
    AttentionGate, ConvBlock, DCNv2, DCNBlock,
    SEModule, SPPPooling, SPPModule, ResidualBlock
    )


def test_attentiongate():
    gate_shape = (8, 128, 64, 64)
    skip_shape = (8, 256, 128, 128)
    gate = torch.randn(*gate_shape)
    skip = torch.randn(*skip_shape)
    model = AttentionGate(128, 256, 128)
    out = model(gate, skip)
    assert out.shape == skip_shape


def test_dcnv2():
    in_shape = (8, 3, 256, 256)
    x = torch.randn(*in_shape)
    model = DCNv2(3, 64)
    out = model(x)
    assert out.shape == (8, 64, 256, 256)


def test_convblock():
    in_shape = (8, 64, 64, 64)
    x = torch.randn(*in_shape)
    model = ConvBlock(64, 128)
    out = model(x)
    assert out.shape == (8, 128, 64, 64)


def test_dcnblock():
    in_shape = (8, 64, 64, 64)
    x = torch.randn(*in_shape)
    model = DCNBlock(64, 128)
    out = model(x)
    assert out.shape == (8, 128, 64, 64)


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


def test_residualblock():
    in_shape = (8, 32, 64, 64)
    x = torch.randn(*in_shape)
    model_keep = ResidualBlock(32, True)
    out1 = model_keep(x)
    assert out1.shape == (8, 64, 64, 64)
    model_down = ResidualBlock(32, True, downsample=True)
    out2 = model_down(x)
    assert out2.shape == (8, 64, 32, 32)
