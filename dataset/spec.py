import utils
import utils.metrics as metrics
from utils.audio_utils import *
from .base import AudioDataset
import torch

class SpecDataset(AudioDataset):
    @staticmethod
    def audio2data(audio, sample_rate):
        spec = utils.AudioToSpec(audio[0]).transpose(-1,-2)
        spec = torch.clip(torch.log(spec+1e-5)/12.,-1,1)
        return (MemberDict({'main':spec}), sample_rate), (None, None)

    def __getitem__(self, i):
        item = AudioDataset.__getitem__(self, i)
        (features, sample_rate), label = item; audio = features['main']
        spec = utils.AudioToSpec(audio).transpose(-1,-2)
        spec = torch.clip(torch.log(spec+1e-5)/12.+torch.randn_like(spec)*self.augment,-1,1)
        return (MemberDict({'main':spec}), sample_rate), label

class MelDataset(AudioDataset):
    @staticmethod
    def audio2data(audio, sample_rate):
        mel = utils.AudioToMel(audio[0], sample_rate=sample_rate).transpose(-1,-2)
        mel = torch.clip(torch.log(mel+1e-5)/12.,-1,1)
        return (MemberDict({'main':mel}), sample_rate), (None, None)

    def __getitem__(self, i):
        item = AudioDataset.__getitem__(self, i)
        (features, sample_rate), label = item; audio = features['main']
        mel = utils.AudioToMel(audio, sample_rate=sample_rate).transpose(-1,-2)
        mel = torch.clip(torch.log(mel+1e-5)/12.+torch.randn_like(mel)*self.augment,-1,1)
        return (MemberDict({'main':mel}), sample_rate), label

class MFCCDataset(AudioDataset):
    @staticmethod
    def audio2data(audio, sample_rate):
        mfcc = utils.AudioToMFCC(audio[0], sample_rate=sample_rate).transpose(-1,-2)
        mfcc = torch.clip(mfcc/1000.,-1,1)
        return (MemberDict({'main':mfcc}), sample_rate), (None, None)

    def __getitem__(self, i):
        item = AudioDataset.__getitem__(self, i)
        (features, sample_rate), label = item; audio = features['main']
        mfcc = utils.AudioToMFCC(audio, sample_rate=sample_rate).transpose(-1,-2)
        mfcc = torch.clip(mfcc/1000.+torch.randn_like(mfcc)*self.augment,-1,1)
        return (MemberDict({'main':mfcc}), sample_rate), label
