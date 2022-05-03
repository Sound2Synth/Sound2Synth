import torch
import torch.nn as nn
import torch.nn.functional as F
from pyheaven.torch_utils import *

class LinearBackbone(nn.Module):
    def __init__(self,
        input_dim,
        hidden_dim = 2048,
        output_dim = 2048,
    ):
        super(LinearBackbone, self).__init__()
        self.input_dim = input_dim; self.hidden_dim = hidden_dim; self.output_dim = output_dim
        self.lin1 = FC(input_dim, hidden_dim, activation = nn.LeakyReLU(inplace=True))
        self.lin2 = FC(hidden_dim, output_dim, activation = nn.LeakyReLU(inplace=True))

    def forward(self, x):
        y = x['main']
        x = y.reshape(y.shape[0],-1)
        return self.lin2(self.lin1(x))