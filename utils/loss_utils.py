import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class DistributionalCrossEntropyDistance(nn.Module):
    def forward(self, pred, true):
        return -torch.sum(true*torch.log(torch.clip(F.softmax(pred,dim=-1), 1e-9, 1)), dim=-1) / (np.log(true.shape[-1]))

def gaussian_kernel(M, std):
    n =  - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

class SmoothedCrossEntropyDistance(nn.Module):
    def __init__(self, k=5, sigma=0.0):
        super().__init__(); self.k = k; self.sigma = sigma; self.cache = {}
    def kernel(self, n):
        if n not in self.cache:
            self.cache[n] = torch.exp(-(torch.arange(-self.k, self.k+1)/n/self.sigma)**2/2); self.cache[n] /= self.cache[n].sum()
        return self.cache[n]
    def weight(self, true):
        weights = F.conv1d(true.unsqueeze(1),self.kernel(true.shape[-1]).unsqueeze(0).unsqueeze(1).type_as(true),padding='same'); return F.normalize(weights.squeeze(1), p=1, dim=-1)
    def forward(self, pred, true):
        return -torch.sum(self.weight(true).detach()*torch.log(torch.clip(F.softmax(pred,dim=-1), 1e-9, 1)), dim=-1) / np.log(true.shape[-1])

class ClassificationLpDistance(nn.Module):
    def __init__(self, p=2.0):
        super().__init__(); self.p = p
    def forward(self, pred, true):
        n = true.shape[-1]; Z = (n-1)*(1/n)**self.p+(1-1/n)**self.p
        return torch.sum((F.softmax(pred,dim=-1)-true).abs()**self.p, dim=-1) / Z

class ClassificationAccuracy(nn.Module):
    def forward(self, pred, true):
        return torch.eq(pred.argmax(dim=-1),true.argmax(dim=-1)).float()

class RegressionLpDistance(nn.Module):
    def __init__(self, p=2.0):
        super().__init__(); self.p = p
    def forward(self, pred, true):
        return (pred.argmax(dim=-1)-true.argmax(dim=-1)).abs()**self.p / (true.shape[-1]-1)

LOSSES_MAPPING = {
    'maeloss': {
        'regl': ClassificationLpDistance(p=1.0),
        'clsl': ClassificationLpDistance(p=1.0),
    },
    'mseloss': {
        'regl': ClassificationLpDistance(p=2.0),
        'clsl': ClassificationLpDistance(p=2.0),
    },
    'celoss': {
        'regl': SmoothedCrossEntropyDistance(sigma=0.02),
        'clsl': DistributionalCrossEntropyDistance(),
    },
    'regceloss': {
        'regl': SmoothedCrossEntropyDistance(sigma=0.02),
        'clsl': None,
    },
    'clsceloss': {
        'regl': None,
        'clsl': DistributionalCrossEntropyDistance(),
    },
    'mixloss': {
        'regl': RegressionLpDistance(p=1.0),
        'clsl': DistributionalCrossEntropyDistance(),
    },
    'clsacc': {
        'regl': None,
        'clsl': ClassificationAccuracy(),
    },
    'regacc': {
        'regl': ClassificationAccuracy(),
        'clsl': None,
    },
    'regmae': {
        'regl': RegressionLpDistance(p=1.0),
        'clsl': None,
    }
}