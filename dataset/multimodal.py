import utils
import utils.metrics as metrics
from utils.audio_utils import *
from .base import AudioDataset
from .spec import SpecDataset, MelDataset, MFCCDataset
import torch
import numpy as np

class MultiModalDataset(MelDataset, MFCCDataset, SpecDataset, AudioDataset):
    @staticmethod
    def audio2features(audio, spec, mel, mfcc, sample_rate):
        features = MemberDict({})
        # STFT Spectrogram & MFCC
        features['main'] = spec
        features[ 'mel'] =  mel
        features['mfcc'] = mfcc

        # Amplitude Envelope (logged)
        features[ 'spec_amp'] =  spec.max(dim=-1,keepdim=True)[0]
        features[  'mel_amp'] =   mel.max(dim=-1,keepdim=True)[0]

        # Root Mean Square Energy (logged)
        features[ 'spec_rms'] = ( spec**2).mean(dim=-1,keepdim=True)**0.5
        features[  'mel_rms'] = (  mel**2).mean(dim=-1,keepdim=True)**0.5

        # Zero Crossing Rate
        basic_args = {
            'hop_length': get_config("default_torchaudio_args")["hop_length"],
        }
        features['zcr'] = torch.tensor(lbrs.feature.zero_crossing_rate(audio.numpy(), **basic_args)).transpose(-1,-2).float()
        features['rms'] = torch.tensor(lbrs.feature.rms(audio.numpy(), **basic_args)).transpose(-1,-2).float()

        # Spectral Flatness
        fft_args = {
            'n_fft': get_config("default_torchaudio_args")["n_fft"],
            'win_length': get_config("default_torchaudio_args")["win_length"],
            'hop_length': get_config("default_torchaudio_args")["hop_length"],
        }
        features['flatness'] = torch.tensor(lbrs.feature.spectral_flatness(audio.numpy(), **fft_args)).transpose(-1,-2).float()

        # Chroma STFT
        fmins = [32.703,65.406,130.81,261.63,523.25,1046.5,2093.0,4186.0,8372.0]
        oct_cqt = [
            torch.tensor(lbrs.feature.chroma_cqt(audio.numpy(), sr=sample_rate, n_octaves=1, fmin=fmins[oct-1], n_chroma=48, bins_per_octave=48)).transpose(-1,-2).float() for oct in range(1,10)
        ]
        features['chroma'] = torch.cat(oct_cqt, dim=-1)

        return features

    @staticmethod
    def audio2data(audio, sample_rate):
        audio_item = AudioDataset.audio2data(audio, sample_rate)
        (audio_features, sample_rate), _ = audio_item; _audio = audio_features['main']

        spec_item = SpecDataset.audio2data(audio, sample_rate)
        (spec_features, _), _ = spec_item; spec = spec_features['main']

        mel_item = MelDataset.audio2data(audio, sample_rate)
        ( mel_features, _), _ =  mel_item;  mel =  mel_features['main']

        mfcc_item = MFCCDataset.audio2data(audio, sample_rate)
        (mfcc_features, _), _ = mfcc_item; mfcc = mfcc_features['main']

        features = MultiModalDataset.audio2features(
            _audio, spec, mel, mfcc, sample_rate
        )

        return (features, sample_rate), (None, None)

    def __getitem__(self, i):
        audio_item = AudioDataset.__getitem__(self, i)
        (audio_features, sample_rate), label = audio_item; audio = audio_features['main']

        spec_item = SpecDataset.__getitem__(self, i)
        (spec_features, _), _ = spec_item; spec = spec_features['main']

        mel_item = MelDataset.__getitem__(self, i)
        ( mel_features, _), _ =  mel_item;  mel =  mel_features['main']

        mfcc_item = MFCCDataset.__getitem__(self, i)
        (mfcc_features, _), _ = mfcc_item; mfcc = mfcc_features['main']

        features = MultiModalDataset.audio2features(
            audio, spec, mel, mfcc, sample_rate
        )

        return (features, sample_rate), label