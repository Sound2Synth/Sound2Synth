import torch
import torch.nn as nn
import torch.nn.functional as F
from pyheaven.torch_utils import *

class ConvBackbone(nn.Module):
    def __init__(self, in_channels = 1, output_dim = 2048):
        super(ConvBackbone, self).__init__()
        self.in_channels = in_channels; self.output_dim = output_dim
        # VGG 11 BN Backbone
        self.model = nn.Sequential(
            CONV(in_channels, 64,3,padding=1,norm_layer=nn.BatchNorm2d( 64),activation=nn.LeakyReLU(inplace=True)),
            nn.MaxPool2d(2,2),
            CONV(         64,128,3,padding=1,norm_layer=nn.BatchNorm2d(128),activation=nn.LeakyReLU(inplace=True)),
            nn.MaxPool2d(2,2),
            CONV(        128,256,3,padding=1,norm_layer=nn.BatchNorm2d(256),activation=nn.LeakyReLU(inplace=True)),
            CONV(        256,256,3,padding=1,norm_layer=nn.BatchNorm2d(256),activation=nn.LeakyReLU(inplace=True)),
            nn.MaxPool2d(2,2),
            CONV(        256,512,3,padding=1,norm_layer=nn.BatchNorm2d(512),activation=nn.LeakyReLU(inplace=True)),
            CONV(        512,512,3,padding=1,norm_layer=nn.BatchNorm2d(512),activation=nn.LeakyReLU(inplace=True)),
            nn.MaxPool2d(2,2),
            CONV(        512,512,3,padding=1,norm_layer=nn.BatchNorm2d(512),activation=nn.LeakyReLU(inplace=True)),
            CONV(        512,512,3,padding=1,norm_layer=nn.BatchNorm2d(512),activation=nn.LeakyReLU(inplace=True)),
            nn.AdaptiveMaxPool2d((2,2)),
        )
        self.linear = FC(2048,output_dim,activation=nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.linear(self.model(x['main'].unsqueeze(1)).flatten(start_dim=1))