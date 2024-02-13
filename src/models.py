import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import deform_conv2d


class AttentionGate(nn.Module):
    """ Implementation of Attention Gate

    Attributes:
        gate_features: No. of features for gating signal
        skip_features: No. of features for skip signal
        attention_coeffs: No. of attention coefficients
    """
    def __init__(self, gate_features: int, skip_features: int,
                 attention_coeffs: int) -> None:
        super(AttentionGate, self).__init__()
        self.gating_sig = nn.Sequential(
            nn.Conv2d(gate_features, attention_coeffs, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(attention_coeffs),
            )
        self.skip_x = nn.Sequential(
            nn.Conv2d(skip_features, attention_coeffs, kernel_size=2,
                      stride=2, bias=False),
            nn.BatchNorm2d(attention_coeffs),
            )
        self.relu = nn.ReLU()
        self.psi = nn.Sequential(
            nn.Conv2d(attention_coeffs, 1, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
            )
        self.resampler = nn.Upsample(scale_factor=2)

    def forward(self, gate: Tensor, skip: Tensor) -> Tensor:
        """ forward pass

        Attirbutes:
            gate: torch tensor from the decoder (b, ch, h, w)
            skip: torch tensor from the encoder (b, ch, h, w)
        """
        g = self.gating_sig(gate)
        x = self.skip_x(skip)
        gx = self.relu(g + x)
        coeffs = self.psi(gx)
        coeffs = self.resampler(coeffs)
        out = skip * coeffs
        return out


class DCNv2(nn.Module):
    """ Implementatino of Deformable Convolution

    Attributes:
        in_channels: No. of input features
        out_channels: No. of output features
        kernel_size: the size of the convolution kernel
        stride: No. of stride for convolution widow
        padding: padding for convolution
        bias: bias for convolution kernels
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | tuple[int, int] = 3, stride: int = 1,
                 padding: int = 1, bias: bool = False) -> None:
        super(DCNv2, self).__init__()

        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size.shape[0], kernel_size.shape[1]
        else:
            kh, kw = kernel_size, kernel_size

        self.stride = stride
        self.padding = padding
        # base convolution with deformable offsets
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        # offset convolution
        self.offset = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * kh * kw,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        # weights are initialized to zero in the original paper
        nn.init.constant_(self.offset.weight, 0.)
        if bias:
            nn.init.constant_(self.offset.bias, 0.)
        # modulated mask convolution
        self.mask_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=kh * kw,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        # weights are initialized to zero in the original paper
        nn.init.constant_(self.mask_conv.weight, 0.)
        if bias:
            nn.init.constant_(self.mask_conv.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ forward pass (b, ch, h, w) """
        offset = self.offset(x)
        mask = 2 * torch.sigmoid(self.mask_conv(x))
        out = deform_conv2d(input=x, offset=offset, weight=self.conv.weight,
                            bias=self.conv.bias, stride=self.stride,
                            padding=self.padding, mask=mask)
        return out


class ConvBlock(nn.Module):
    """ Convolution block layer for U-Net

    Attributes:
        in_channels: No. of input channels
        features: No. of output channels
    """
    def __init__(self, in_channels: int, features: int) -> None:
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(in_channels=features, out_channels=features,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            )

    def forward(self, x: Tensor) -> Tensor:
        """ forward pass (b, ch, h, w) """
        return self.block(x)


class DCNBlock(nn.Module):
    """ Convolution block with Deformable Convolution for U-Net

    Attributes:
        in_channels: No. of input channels
        features: No. of output features
    """
    def __init__(self, in_channels: int, features: int) -> None:
        super(DCNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            DCNv2(features, features),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(in_channels=features, out_channels=features,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            )

    def forward(self, x: Tensor) -> Tensor:
        """ forward pass (b, ch, h, w) """
        return self.block(x)


class UNet(nn.Module):
    """ Base U-Net model

    The base model can be extended with Attention Gate and Deformable \
        Convolution

    Attributes:
        in_channels: No. of input channels
        out_channels: No. of classes
        features: No. of the base featuress
        logits: applies sigmoid on model predictions if false
        attention_gate: enable Attention Gate if true
        dcn: replaces CNNs with Deformable Convolutions
    """
    def __init__(self, in_channels: int, out_channels: int = 1,
                 features: int = 32, logits: bool = False,
                 attention_gate: bool = False, dcn: bool = False) -> None:
        super(UNet, self).__init__()
        self.logits = logits
        self.attention_gate = attention_gate
        self.dcn = dcn
        # encoder layers
        self.p = nn.MaxPool2d(kernel_size=2, stride=2)
        if self.dcn:
            self.e1 = DCNBlock(in_channels, features)
        else:
            self.e1 = ConvBlock(in_channels, features)
        self.e2 = ConvBlock(features, features*2)
        self.e3 = ConvBlock(features*2, features*4)
        self.e4 = ConvBlock(features*4, features*8)
        # bottom
        if self.dcn:
            self.neck = DCNBlock(features*8, features*16)
        else:
            self.neck = ConvBlock(features*8, features*16)
        # decoder layers
        self.ag4 = AttentionGate(
            gate_features=features*16, skip_features=features*8,
            attention_coeffs=features*8
            )
        self.up4 = nn.ConvTranspose2d(
            in_channels=features*16, out_channels=features*8,
            kernel_size=2, stride=2, bias=False
            )
        self.d4 = ConvBlock(features*16, features*8)
        self.ag3 = AttentionGate(
            gate_features=features*8, skip_features=features*4,
            attention_coeffs=features*4
            )
        self.up3 = nn.ConvTranspose2d(
            in_channels=features*8, out_channels=features*4,
            kernel_size=2, stride=2, bias=False
            )
        self.d3 = ConvBlock(features*8, features*4)
        self.ag2 = AttentionGate(
            gate_features=features*4, skip_features=features*2,
            attention_coeffs=features*2
            )
        self.up2 = nn.ConvTranspose2d(
            in_channels=features*4, out_channels=features*2,
            kernel_size=2, stride=2, bias=False
            )
        if self.dcn:
            self.d2 = DCNBlock(features*4, features*2)
        else:
            self.d2 = ConvBlock(features*4, features*2)
        self.ag1 = AttentionGate(
            gate_features=features*2, skip_features=features*1,
            attention_coeffs=features*1
            )
        self.up1 = nn.ConvTranspose2d(
            in_channels=features*2, out_channels=features,
            kernel_size=2, stride=2, bias=False
            )
        self.d1 = ConvBlock(features*2, features)
        self.out_conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1,
            bias=False
            )

    def forward(self, x: Tensor) -> Tensor:
        """ forward pass (b, ch, h, w) """
        # encoding layers
        e1 = self.e1(x)
        e2 = self.e2(self.p(e1))
        e3 = self.e3(self.p(e2))
        e4 = self.e4(self.p(e3))
        # bottleneck layer
        neck = self.neck(self.p(e4))
        # decoding layers
        if self.attention_gate:
            e4 = self.ag4(neck, e4)
        d4 = self.up4(neck)
        d4 = torch.cat(tensors=(d4, e4), dim=1)
        d4 = self.d4(d4)
        if self.attention_gate:
            e3 = self.ag3(d4, e3)
        d3 = self.up3(d4)
        d3 = torch.cat(tensors=(d3, e3), dim=1)
        d3 = self.d3(d3)
        if self.attention_gate:
            e2 = self.ag2(d3, e2)
        d2 = self.up2(d3)
        d2 = torch.cat(tensors=(d2, e2), dim=1)
        d2 = self.d2(d2)
        if self.attention_gate:
            e1 = self.ag1(d2, e1)
        d1 = self.up1(d2)
        d1 = torch.cat(tensors=(d1, e1), dim=1)
        d1 = self.d1(d1)
        if self.logits:
            return self.out_conv(d1)
        # output convolution with sigmoid
        return torch.sigmoid(self.out_conv(d1))
