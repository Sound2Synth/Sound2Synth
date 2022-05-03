from utils.basic_utils import *
from utils.loss_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class BaseInterface(object):
    regression_nclasses = 256
    regression_parameters = list()
    classification_parameters = list()
    ordered_descriptors = MemberDict()

    def __init__(self, data):
        self.data = MemberDict(data)

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    def to_dict(self):
        return MemberDict(self.data)
    
    @classmethod
    def from_synth(cls, synth):
        data = {
            key:cls.ordered_descriptors[key]["serialize"](
                {"value":param.current_value,"param":param}
            )   for key, param in synth.parameters.items()
        }
        return cls({k:v for k,v in data.items() if v is not None})
    
    def to_synth(self, synth):
        for key, param in synth.parameters.items():
            param.current_value = self.ordered_descriptors[key]["unserialize"](
                {"value":self.data[key] if key in self.data else None,"param":param,"interface":self}
            )
    
    @classmethod
    def from_tensor(cls, data):
        index = 0; result = {}
        for key, classes in cls.classification_parameters:
            result[key] = float(torch.argmax(data[index:index+classes]))/(classes-1)
            index += classes
        for key in cls.regression_parameters:
            result[key] = float(torch.argmax(data[index:index+cls.regression_nclasses]))/cls.regression_nclasses
            index += cls.regression_nclasses
        return cls.from_dict(result)
    
    def to_tensor(self):
        return torch.cat([
            F.one_hot(torch.tensor([Clipped(int(round(self.data[key]*(classes-1))),0,classes-1)]), num_classes=classes)*1.0
            for key, classes in self.__class__.classification_parameters
        ] + [
            F.one_hot(torch.tensor([Clipped(int(round(self.data[key]*(self.__class__.regression_nclasses-1))),0,self.__class__.regression_nclasses-1)]), num_classes=self.__class__.regression_nclasses)*1.0
            for key in self.__class__.regression_parameters
        ], dim=-1).squeeze()

    @classmethod
    def dim(cls):
        return sum(x[1] for x in cls.classification_parameters) + cls.regression_nclasses*len(cls.regression_parameters)
    
    @classmethod
    def parameters(cls):
        return [x[0] for x in cls.classification_parameters] + cls.regression_parameters
    
    @classmethod
    def division(cls):
        return [x[1] for x in cls.classification_parameters] + [cls.regression_nclasses for x in cls.regression_parameters]
    
    @classmethod
    def named_division(cls):
        return {a[0]:a[1] for a in cls.classification_parameters+[(x,cls.regression_nclasses) for x in cls.regression_parameters]}

    @classmethod
    def profile(cls):
        return {
            "# Regression Parameters":len(cls.regression_parameters),
            "# Regression Classes":cls.regression_nclasses*len(cls.regression_parameters),
            "# Classification Parameters":len(cls.classification_parameters),
            "# Classification Classes":sum(x[1] for x in cls.classification_parameters),
            "# Parameters":len(cls.parameters()),
            "# Classes":cls.dim(),
        }

class ParameterSpaceLoss(nn.Module):
    def __init__(self, default_loss_type='celoss', regression_parameters=list(), classification_parameters=list(), regression_nclasses=list()):
        super().__init__(); self.default_loss_type = default_loss_type; self.regp = regression_parameters; self.clsp = classification_parameters; self.regn = regression_nclasses
    
    def forward(self, pred, true, weights, loss_type=None):
        loss_type = self.default_loss_type if loss_type is None else loss_type
        clsl = LOSSES_MAPPING[loss_type]['clsl']; regl = LOSSES_MAPPING[loss_type]['regl']
        index = 0; result = {}; keys = set()
        if clsl is not None:
            for key, classes in self.clsp:
                result[key] = clsl(pred[:,index:index+classes],true[:,index:index+classes])
                keys.add(key); index += classes
        index = sum(classes for _, classes in self.clsp)
        if regl is not None:
            for key in self.regp:
                result[key] = regl(pred[:,index:index+self.regn],true[:,index:index+self.regn])
                keys.add(key); index += self.regn
        losses = sum(result[key]*weights[key] for key in keys)/(sum(weights[key] for key in keys)+1e-9)
        return losses.mean()
