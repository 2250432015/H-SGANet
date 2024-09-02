import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from utils import SpatialTransformer
from nets.graphlayers import Vig
from nets.SSA import *

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None, up=True):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        if up is True:
            x = self.up(x)
        return x

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.c1 = nn.Sequential(nn.Conv3d(2, 16, 3, 1, 1, bias=False),
                                nn.LeakyReLU(inplace=True),
                                nn.InstanceNorm3d(16))
        self.c2 = nn.Sequential(nn.AvgPool3d(3, stride=2, padding=1),
                                nn.Conv3d(16, 32, 3, 1, 1, bias=False),
                                nn.LeakyReLU(inplace=True),
                                nn.InstanceNorm3d(32))
        self.c3 = nn.Sequential(nn.AvgPool3d(3, stride=2, padding=1),
                                nn.Conv3d(32, 32, 3, 1, 1, bias=False),
                                nn.LeakyReLU(inplace=True),
                                nn.InstanceNorm3d(32))
        self.c4 = nn.Sequential(nn.AvgPool3d(3, stride=2, padding=1),
                                nn.Conv3d(32, 32, 3, 1, 1, bias=False),
                                nn.LeakyReLU(inplace=True),
                                nn.InstanceNorm3d(32))

        self.up1 = DecoderBlock(32, 16, 16, use_batchnorm=False)
        self.up2 = DecoderBlock(32, 32, 32, use_batchnorm=False)
        self.up3 = DecoderBlock(32, 32, 32, use_batchnorm=False)
        self.up4 = DecoderBlock(32, 32, 32, use_batchnorm=False)

        self.vig1 = Vig(32)
        self.vig2 = Vig(32)
        self.vig3 = Vig(32)
        self.vig4 = Vig(32)

        self.ssa = MobileViTv2Attention(d_model=28)

        self.reg_head = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer([160, 192, 224])

    def forward(self, x):
        source = x[:, 0:1, :, :, :]
        cx = x.clone()
        
        f1 = self.c1(cx)
        fvig1 = self.vig1(f1)  
        x = f1 + fvig1 

        f2 = self.c2(x)
        fvig2 = self.vig2(f2) 
        x = f2 + fvig2  

        f3 = self.c3(x)
        fvig3 = self.vig3(f3)  
        x = f3 + fvig3  

        f4 = self.c4(f3)
        fvig4 = self.vig4(f4) 
        x = f4 + fvig4  

        fssa = self.ssa(x)
        x = x + fssa  

        x = self.up4(x, f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1, False)

        flow = self.reg_head(x)
        output = self.spatial_trans(source, flow)
        return output, flow
