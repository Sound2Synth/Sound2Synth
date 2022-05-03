from .conv_backbone import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyheaven.torch_utils import *

class NaiveParametersClassifier(nn.Module):
    def __init__(self, division, input_dim = 2048, hidden_dim_per_group = 64):
        super(NaiveParametersClassifier, self).__init__()
        self.division = division
        self.input_dim = input_dim
        hidden_division = [hidden_dim_per_group for _ in division]
        self.hidden_dim_per_group = hidden_dim_per_group
        self.hidden_dim = sum(hidden_division)
        self.output_dim = sum(division)
        self.lin1 = FC(self.input_dim, self.hidden_dim, dropout=None, activation=nn.LeakyReLU(inplace=True))
        self.lin2 = GroupFC(hidden_division, hidden_division, dropout=None, activation=nn.LeakyReLU(inplace=True))
        self.lin3 = GroupFC(hidden_division, division, dropout=None, activation=nn.Identity())

    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(x)))

class OscillatorAttentionClassifier(NaiveParametersClassifier):
    def __init__(self, named_division, oscillators, input_dim = 2048, hidden_dim_per_group = 64):
        division = [x[1] for x in named_division.items()]
        super(OscillatorAttentionClassifier, self).__init__(division, input_dim, hidden_dim_per_group)
        self.att1 = FC(self.input_dim, len(oscillators)*hidden_dim_per_group, dropout=None, activation=nn.LeakyReLU(inplace=True))
        self.att2 = GroupFC([hidden_dim_per_group for _ in oscillators], [hidden_dim_per_group for _ in oscillators], dropout=None, activation=nn.Tanh())
        self.num_oscillators = len(oscillators); self.oscillator_group = [0 for _ in range(len(named_division))]
        for osc_id, params in enumerate(oscillators):
            for param_id in params:
                self.oscillator_group[param_id] = osc_id
    
    def forward(self, x):
        oscatt = self.att2(self.att1(x))
        att = torch.cat([oscatt[:,osc_id*self.hidden_dim_per_group:(osc_id+1)*self.hidden_dim_per_group].clone() for osc_id in self.oscillator_group], dim=-1).type_as(x)
        return self.lin3(self.lin2(att*self.lin1(x)))

class Net(nn.Module):
    def __init__(self, backbone, classifier):
        super(Net, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self,x):
        x = self.backbone(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x