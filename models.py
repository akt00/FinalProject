import torch
import torch.nn as nn


# the code in this section is originally written by Akihiro Tanimoto
class ConvBlock(nn.Module):
    """ CNN block layer for U-Net

    Attributes:
        in_channels (int): the number of input channels
        features: the number of output channels after the first ConvBlock
    """
    def __init__(self, in_channels: int, features: int) -> None:
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=features,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=features)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=features)
        self.r2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """ forward pass

        Args:
            x: input feature
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.r1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.r2(x)
        return x
# The section ends here


# the code in this section is originally written by Akihiro Tanimoto
class UNet(nn.Module):
    """ U-Net model

    Attributes:
        in_channels: the number of input channels
        out_channels: the number of output channels. this equals the \
            number of classes
        features: the number of output channels from the 1st ConvBlock
    """
    def __init__(self, in_channels: int, out_channels: int = 1,
                 features: int = 32) -> None:
        super(UNet, self).__init__()
        # encoder layers
        self.enc1 = ConvBlock(in_channels=in_channels, features=features)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(in_channels=features, features=features*2)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(in_channels=features*2, features=features*4)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = ConvBlock(in_channels=features*4, features=features*8)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # bottom
        self.bottolneck = ConvBlock(in_channels=features*8,
                                    features=features*16)
        # decoder layers
        self.up4 = nn.ConvTranspose2d(
            in_channels=features*16,
            out_channels=features*8,
            kernel_size=2,
            stride=2
            )
        self.dec4 = ConvBlock(in_channels=features*16, features=features*8)

        self.up3 = nn.ConvTranspose2d(
            in_channels=features*8,
            out_channels=features*4,
            kernel_size=2,
            stride=2
        )
        self.dec3 = ConvBlock(in_channels=features*8, features=features*4)

        self.up2 = nn.ConvTranspose2d(
            in_channels=features*4,
            out_channels=features*2,
            kernel_size=2,
            stride=2
        )
        self.dec2 = ConvBlock(in_channels=features*4, features=features*2)

        self.up1 = nn.ConvTranspose2d(
            in_channels=features*2,
            out_channels=features,
            kernel_size=2,
            stride=2
        )
        self.dec1 = ConvBlock(in_channels=features*2, features=features)

        self.out_conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        """ forward pass

        Args:
            x: input tensor
        """
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.p1(enc1))
        enc3 = self.enc3(self.p2(enc2))
        enc4 = self.enc4(self.p3(enc3))

        bottleneck = self.bottolneck(self.p4(enc4))

        dec4 = self.up4(bottleneck)
        dec4 = torch.cat(tensors=(dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.up3(dec4)
        dec3 = torch.cat(tensors=(dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat(tensors=(dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat(tensors=(dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.out_conv(dec1))
# the section ends here


class AttentionGate(nn.Module):
    def __init__(self, gate_features: int, skip_features: int,
                 attention_coeffs: int) -> None:
        super(AttentionGate, self).__init__()
        self.gating_sig = nn.Sequential(
            nn.Conv2d(gate_features, attention_coeffs, kernel_size=1,
                      padding=0, bias=True),
            nn.BatchNorm2d(attention_coeffs)
        )
        self.skip_x = nn.Sequential(
            nn.Conv2d(skip_features, attention_coeffs, kernel_size=2,
                      stride=2, padding=0, bias=True),
            nn.BatchNorm2d(attention_coeffs)
        )
        self.relu = nn.ReLU()
        self.psi = nn.Sequential(
            nn.Conv2d(attention_coeffs, 1, kernel_size=1, stride=1, padding=0,
                      bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.resampler = nn.Upsample(scale_factor=2)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor):
        g = self.gating_sig(gate)
        x = self.skip_x(skip)
        gx = self.relu(g + x)
        coeffs = self.psi(gx)
        coeffs = self.resampler(coeffs)
        out = skip * coeffs
        return out


class AttentionGatedUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1,
                 features: int = 32) -> None:
        super(AttentionGatedUNet, self).__init__()
        self.p = nn.MaxPool2d(kernel_size=2, stride=2)
        # encoder
        self.e1 = ConvBlock(in_channels=in_channels, features=features)
        self.e2 = ConvBlock(in_channels=features, features=features*2)
        self.e3 = ConvBlock(in_channels=features*2, features=features*4)
        self.e4 = ConvBlock(in_channels=features*4, features=features*8)
        # bottom
        self.bottolneck = ConvBlock(in_channels=features*8,
                                    features=features*16)
        # attention decoder
        self.ag4 = AttentionGate(
            gate_features=features*16, skip_features=features*8,
            attention_coeffs=features*8
        )
        self.up4 = nn.ConvTranspose2d(
            in_channels=features*16, out_channels=features*8,
            kernel_size=2, stride=2
            )
        self.d4 = ConvBlock(in_channels=features*16, features=features*8)
        self.ag3 = AttentionGate(
            gate_features=features*8, skip_features=features*4,
            attention_coeffs=features*4
        )
        self.up3 = nn.ConvTranspose2d(
            in_channels=features*8, out_channels=features*4,
            kernel_size=2, stride=2
            )
        self.d3 = ConvBlock(in_channels=features*8, features=features*4)
        self.ag2 = AttentionGate(
            gate_features=features*4, skip_features=features*2,
            attention_coeffs=features*2
        )
        self.up2 = nn.ConvTranspose2d(
            in_channels=features*4, out_channels=features*2,
            kernel_size=2, stride=2
            )
        self.d2 = ConvBlock(in_channels=features*4, features=features*2)
        self.ag1 = AttentionGate(
            gate_features=features*2, skip_features=features*1,
            attention_coeffs=features*1
        )
        self.up1 = nn.ConvTranspose2d(
            in_channels=features*2, out_channels=features*1,
            kernel_size=2, stride=2
            )
        self.d1 = ConvBlock(in_channels=features*2, features=features*1)
        self.out_conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoding
        e1 = self.e1(x)
        e2 = self.e2(self.p(e1))
        e3 = self.e3(self.p(e2))
        e4 = self.e4(self.p(e3))
        # bottlneck
        bottle = self.bottolneck(self.p(e4))
        # attention gated decoding
        # print(bottle.shape, e4.shape)
        s4 = self.ag4(bottle, e4)
        d4 = self.up4(bottle)
        d4 = torch.cat(tensors=(d4, s4), dim=1)
        d4 = self.d4(d4)
        # print(d4.shape, e3.shape)
        s3 = self.ag3(d4, e3)
        d3 = self.up3(d4)
        d3 = torch.cat(tensors=(d3, s3), dim=1)
        d3 = self.d3(d3)

        s2 = self.ag2(d3, e2)
        d2 = self.up2(d3)
        d2 = torch.cat(tensors=(d2, s2), dim=1)
        d2 = self.d2(d2)

        s1 = self.ag1(d2, e1)
        d1 = self.up1(d2)
        d1 = torch.cat(tensors=(d1, s1), dim=1)
        d1 = self.d1(d1)
        return torch.sigmoid(self.out_conv(d1))


class DCNv2(nn.Module):
    def __init__(self) -> None:
        super(DCNv2, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


t = torch.randn(8, 3, 256, 256)
model = AttentionGatedUNet(3)
print(model(t).shape)
