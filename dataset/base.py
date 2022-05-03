import utils
import utils.metrics as metrics
from utils.audio_utils import *
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, name=None, splits=None, with_params=False, with_gradient=False, augment=0.0):
        if name is not None:
            self.name = name; self.splits = splits
            self.path = pjoin(get_config("server_data_dir").split(':')[-1],name); data = ListFiles(self.path, ordered=True)
            self.wav = [pjoin(self.path,file) for file in data if (Suffix(file)== 'wav') and (file.split('.')[0] in self.splits)]
            self.par = [pjoin(self.path,file) for file in data if (Suffix(file)=='json') and (file.split('.')[0] in self.splits)]
            self.grd = [pjoin(self.path,file) for file in data if (Suffix(file)=='grad') and (file.split('.')[0] in self.splits)]
            self.with_params = with_params
            self.with_gradient = with_gradient
            self.augment = augment
            if with_params:
                prefixes = set(Prefix(file) for file in self.wav).intersection(set(Prefix(file) for file in self.par))
                self.wav = [file for file in self.wav if Prefix(file) in prefixes]
                self.par = [file for file in self.par if Prefix(file) in prefixes]
                self.grd = [file for file in self.grd if Prefix(file) in prefixes]
            else:
                self.par = [None for _ in self.wav]
            if with_gradient:
                prefixes = set(Prefix(file) for file in self.wav).intersection(set(Prefix(file) for file in self.grd))
                self.wav = [file for file in self.wav if Prefix(file) in prefixes]
                self.par = [file for file in self.par if Prefix(file) in prefixes]
                self.grd = [file for file in self.grd if Prefix(file) in prefixes]
            else:
                self.grd = [None for _ in self.wav]
        else:
            self.name = None; self.splits = splits; self.path = None
            self.wav = None; self.par = None; self.grd = None
            self.with_params = with_params; self.with_gradient = with_gradient
            self.augment = augment
    
    def __len__(self):
        return len(self.wav)
    
    @staticmethod
    def audio2data(audio, sample_rate):
        return (MemberDict({'main':audio[0]}), sample_rate), (None, None)

    def __getitem__(self, i):
        audio, sample_rate = tau.load(self.wav[i])
        parameters = "" if self.par[i] is None else self.par[i]
        gradients = "" if self.grd[i] is None else self.grd[i]
        return (MemberDict({'main':audio[0]}), sample_rate), (parameters, gradients)

    def __str__(self):
        return "%s(len=%d, with_params=%d, with_gradient=%d) [split: %s]"%(
            self.name,len(self),self.with_params,self.with_gradient,self.split
        ) if self.name is not None else "null"
