import torch
import torch.nn as nn
import torch.nn.functional as F
from pyheaven.torch_utils import *

def ConvMixer(in_channels, h, depth, kernel_size=9, patch_size=7, n_classes=1000):
    Seq, ActBn = nn.Sequential, lambda x: Seq(x, nn.GELU(), nn.BatchNorm2d(h))
    Residual = type('Residual', (Seq,), {'forward': lambda self, x: self[0](x) + x})
    return Seq(ActBn(nn.Conv2d(in_channels, h, patch_size, stride=patch_size)),
               *[Seq(Residual(ActBn(nn.Conv2d(h, h, kernel_size, groups=h, padding="same"))),
               ActBn(nn.Conv2d(h, h, 1))) for _ in range(depth)],
               nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(h, n_classes))

class ConvMixerBackbone(nn.Module):
    def __init__(self, in_channels = 1, output_dim = 2048):
        super(ConvMixerBackbone, self).__init__()
        self.in_channels = in_channels; self.output_dim = output_dim
        self.model = ConvMixer(in_channels=1,h=768,depth=32,n_classes=2048)
        self.linear = FC(2048,output_dim,activation=nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.linear(self.model(x['main'].unsqueeze(1)).flatten(start_dim=1))