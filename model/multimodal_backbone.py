import torch
import torch.nn as nn
import torch.nn.functional as F
from pyheaven.torch_utils import *
from utils import get_config

from .linear_backbone import LinearBackbone
from .lstm_backbone import LSTMBackbone
from .conv_backbone import ConvBackbone
from .pdc_backbone import PDCNN2Backbone

import numpy as np

class ModeDropout(nn.Module):
    def __init__(self, p=0.0, modes=list()):
        super().__init__(); self.p = p; self.modes = modes; self.n = sum(self.modes)
    
    def forward(self, x):
        if self.training and self.p>0.:
            index = 0; masked = 0
            for mode in self.modes:
                if np.random.rand()<self.p:
                    x[:,index:index+mode] *= 0.
                    masked += mode
            ratio = self.n/(self.n-masked) if masked < self.n else 0
            x *= ratio
        return x

class MultimodalBackbone(nn.Module):
    def __init__(self, output_dim = 2048, mode_dropout = 0.0, conv = True):
        super(MultimodalBackbone, self).__init__()
        self.output_dim = output_dim
        self.backbones = nn.ModuleDict({
            'main':     ConvBackbone(output_dim=512),
            'chroma':   PDCNN2Backbone(output_dim=512),
            'mfcc':     LSTMBackbone(input_dim=20,hidden_dim=256,num_layers=2,output_dim=128),
            'spec_amp': LinearBackbone(input_dim=get_config("default_torchaudio_args")["input_size"],hidden_dim=256,output_dim=128),
            'mel_amp':  LinearBackbone(input_dim=get_config("default_torchaudio_args")["input_size"],hidden_dim=256,output_dim=128),
            'spec_rms': LinearBackbone(input_dim=get_config("default_torchaudio_args")["input_size"],hidden_dim=256,output_dim=128),
            'mel_rms':  LinearBackbone(input_dim=get_config("default_torchaudio_args")["input_size"],hidden_dim=256,output_dim=128),
            'zcr':      LinearBackbone(input_dim=get_config("default_torchaudio_args")["input_size"],hidden_dim=256,output_dim=128),
            'rms':      LinearBackbone(input_dim=get_config("default_torchaudio_args")["input_size"],hidden_dim=256,output_dim=128),
            'flatness': LinearBackbone(input_dim=get_config("default_torchaudio_args")["input_size"],hidden_dim=256,output_dim=128),
        })
        if not conv:
            del self.backbones['main']
        self.dropout = ModeDropout(mode_dropout, modes=[layer.output_dim for key,layer in self.backbones.items()])
        self.linear = FC(sum(net.output_dim for net in self.backbones.values()), output_dim, activation=nn.LeakyReLU(inplace=True))

    def forward(self, x):
        features = {}
        for key in self.backbones:
            features[key] = self.backbones[key]({'main':x[key]})
        y = torch.cat([features[key] for key in sorted(features.keys())],dim=-1)
        return self.linear(self.dropout(y))