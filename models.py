import torch
import torch.nn as nn


# the code in this section is originally written by Akihiro Tanimoto
class ConvBlock(nn.Module):
    """ CNN block layer for U-Net

    Attributes:
        in_channels (int): the number of input channels
        features: the number of output channels after the first ConvBlock
        args: not in use
        kwargs: not in use
    """
    def __init__(self, in_channels, features, *args, **kwargs) -> None:
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=features,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=features)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=features)
        self.r2 = nn.ReLU()

    def forward(self, x):
        """ forward pass

        Args:
            x (torch.Tensor): input feature
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
    """ UNet

    Attributes:
        in_channels (int): the number of input channels
        out_channels (int): the number of output channels. this equals the \
            number of classes
        features (int): the number of output channels from the 1st ConvBlock
        args: currently not in use
        kwargs: currently not in use
    """
    def __init__(self, in_channels, out_channels=1, features=32,
                 *args, **kwargs) -> None:
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

    def forward(self, x):
        """ forward pass

        Args:
            x (torch.Tensor): input
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
