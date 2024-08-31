import torch
import torch.nn.functional as F
# from timm.models.layers import DropPath
from torch import nn
import gc

class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    
    K is the number of superpatches, therefore hops equals res // K.
    """
    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv3d(in_channels * 2, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
            )
        self.K = K

    def forward(self, x):
        B, C, H, W, D = x.shape

        x_j = x - x
        for i in range(self.K, H, self.K):
            x_c = x.detach() - torch.cat([x[:, :, -i:, :, :], x[:, :, :-i, :, :]], dim=2).detach()
            x_j = torch.max(x_j, x_c).detach()
            del x_c
            gc.collect()
            torch.cuda.empty_cache()

        for i in range(self.K, W, self.K):
            x_r = x.detach() - torch.cat([x[:, :, :, -i:, :], x[:, :, :, :-i, :]], dim=3).detach()
            x_j = torch.max(x_j, x_r).detach()
            del x_r
            gc.collect()
            torch.cuda.empty_cache()

        for i in range(self.K, D, self.K):
            x_v = x.detach() - torch.cat([x[:, :, :, :, -i:], x[:, :, :, :, :-i]], dim=4).detach()
            x_j = torch.max(x_j, x_v).detach()
            del x_v
            gc.collect()
            torch.cuda.empty_cache()

        x = torch.cat([x, x_j], dim=1)
        del x_j
        return self.nn(x)


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, drop_path=0.0, K=20):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.fc1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm3d(in_channels),
        )
        self.graph_conv = MRConv4d(in_channels, in_channels, K=self.K)
        self.fc2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm3d(in_channels),
        )  # out_channels back to 1x}
        self.drop_path =  nn.Identity()

       
    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp

        return x

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features # same as input
        hidden_features = hidden_features or in_features # x4
        self.fc1 = nn.Sequential(
            nn.Conv3d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm3d(hidden_features),
        )
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Conv3d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm3d(out_features),
        )
        self.drop_path = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  
    
class Vig(nn.Module):
    def __init__(self, hidden, global_blocks=1):
        super().__init__()
        self.backbone = nn.ModuleList([])
        for _ in range(global_blocks):
            # self.backbone += [nn.Sequential(Grapher(hidden),
            #                 FFN(hidden, hidden * 4))]
            self.backbone += [nn.Sequential(Grapher(hidden))
                                            ]
        
    def forward(self, x):
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        return x

if __name__=='__main__':
    g = Vig(6)
    x = torch.rand((1, 6, 3, 2, 25))
    y = g(x)
    print(y.shape)