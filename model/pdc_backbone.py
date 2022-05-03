import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyheaven.torch_utils import *

from torch.nn.parameter import Parameter
from torch.nn import init

class PrimeDilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=1, padding=0,
                 bins_per_octave=12, selected_primes=2, version='sym', variant=2):
        super(PrimeDilatedConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channel = in_channels if out_channels is None else out_channels
        self.version = version
        self.variant = variant
        self.B = bins_per_octave

        if selected_primes < 1:
            raise Exception(f"L should be a natural number. Given {selected_primes}.")

        primes, ratios = prime_ratios(selected_primes)

        self.k = [round(bins_per_octave * math.log(r, 2)) for r in ratios]

        if self.version.lower().startswith('a'):
            self.k = [0] + self.k
        elif self.version.lower().startswith('s'):
            k_minus = [-i for i in self.k]
            k_minus.reverse()
            self.k = k_minus + [0] + self.k
        else:
            raise Exception(f"Version should be 'A' for asymmetric or 'S' for symmetric. Given {version}.")

        self.k = list(set(self.k))
        self.k.sort()

        if self.variant == 1:
            self.weights = Parameter(torch.Tensor(len(self.k)))
            init.uniform_(self.weights, a=-1, b=1)
        elif self.variant == 2:
            self.weights = Parameter(torch.Tensor(len(self.k), in_channels))
            init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        elif self.variant == 3:
            self.conv = nn.Conv2d(in_channels * len(self.k), out_channels,
                                  kernel_size=kernel_size, padding=padding, bias=False)
        else:
            raise Exception(f"Variant should be 1, 2 or 3. Given {variant}.")

    def forward(self, h):
        channels = [harmonic_shift(h, j) for j in self.k]

        if self.variant == 1:
            out = torch.stack(channels, dim=4)
            out = out * self.weights
            out = torch.sum(out, dim=4)

        elif self.variant == 2:
            out = torch.stack(channels, dim=3)
            out = torch.transpose(out, 1, 4)
            out = out * self.weights
            out = torch.transpose(out, 1, 4)
            out = torch.sum(out, dim=3)

        elif self.variant == 3:
            out = torch.cat(channels, dim=1)
            out = self.conv(out)
        else:
            raise Exception(f"Variant should be 1, 2 or 3. Given {self.variant}.")

        return out


class HarmonicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 bins_per_octave=12, selected_primes=2, version='sym', variant=2):
        super(HarmonicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bins_per_octave = bins_per_octave

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)

        self.pdc = PrimeDilatedConv2d(out_channels, bins_per_octave=bins_per_octave,
                                      selected_primes=selected_primes, version=version, variant=variant)

    def forward(self, h):
        out = self.conv(h)
        out = self.pdc(out)

        return out


def prime_ratios(num=1):
    prime_list = [2]
    ratio_list = [2]
    v = 3
    s = 2

    while len(prime_list) < num:
        for i in prime_list:
            if v % i == 0:
                break
            elif i * i > v + 1:
                prime_list.append(v)
                ratio_list.append(v/s)
                break
        else:
            prime_list.append(v)
            ratio_list.append(v/s)

        v += 2
        if v > s * 2:
            s *= 2

    return prime_list, ratio_list


def harmonic_shift(spec, shift=0):
    if shift == 0:
        return spec

    batch, channels, bins, step = spec.shape
    zeros = torch.zeros((batch, channels, abs(shift), step), dtype=torch.float32, device=spec.device)

    if shift > 0:
        indices = torch.tensor(range(0, bins - shift), device=spec.device)
        out = torch.cat((zeros, torch.index_select(spec, 2, indices)), 2)
    else:
        shift = -shift
        indices = torch.tensor(range(shift, bins), device=spec.device)
        out = torch.cat((torch.index_select(spec, 2, indices), zeros), 2)

    return out

class VCNN2Backbone(nn.Module):
    def __init__(self, in_channels = 1, output_dim = 2048):
        super(VCNN2Backbone, self).__init__()
        self.in_channels = in_channels; self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((2,2)),
        )
        self.linear = FC(512,output_dim,activation=nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.linear(self.model(x['main'].unsqueeze(1)).flatten(start_dim=1))

class PDCNN2Backbone(nn.Module):
    def __init__(self, in_channels = 1, output_dim = 2048):
        super(PDCNN2Backbone, self).__init__()
        self.in_channels = in_channels; self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            PrimeDilatedConv2d(64, bins_per_octave=48, selected_primes=3, version='S', variant=2),
            PrimeDilatedConv2d(64, bins_per_octave=48, selected_primes=3, version='S', variant=2),
            PrimeDilatedConv2d(64, bins_per_octave=48, selected_primes=3, version='S', variant=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, 128, kernel_size=7, padding=3, bias=False),
            PrimeDilatedConv2d(128, bins_per_octave=12, selected_primes=3, version='S', variant=2),
            PrimeDilatedConv2d(128, bins_per_octave=12, selected_primes=3, version='S', variant=2),
            PrimeDilatedConv2d(128, bins_per_octave=12, selected_primes=3, version='S', variant=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((2,2)),
        )
        self.linear = FC(512,output_dim,activation=nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.linear(self.model(x['main'].unsqueeze(1)).flatten(start_dim=1))
