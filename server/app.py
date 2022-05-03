from pyheaven import *
import torch
import torchaudio as tau

import sys
sys.path.append('./')
from train import INTERFACE_MAPPING, DATASET_MAPPING, get_backbone, get_classifier, Net, Sound2SynthModel
from utils import get_config, AdjustAudioLength

def LoadModel(experiment):
    args = LoadJson(pjoin("experiment", experiment, "args.json"))
    train_args = MemberDict(args['train_args'])
    interface = INTERFACE_MAPPING[train_args.synth]
    net = Net(
        backbone = get_backbone(train_args.backbone, train_args),
        classifier = get_classifier(train_args.classifier, interface, train_args),
    )
    model = Sound2SynthModel(net, interface, args=train_args)
    checkpoint = pjoin("experiment", experiment, "checkpoints", "latest")
    if ExistFile(checkpoint):
        model.load_state_dict(torch.load(checkpoint,map_location=torch.device('cpu'))['state_dict'])
    else:
        raise FileNotFoundError(checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def LoadProcessedAudio(filepath):
    audio, sample_rate = tau.load(filepath)
    # Align sample rate
    target_sample_rate = get_config('default_torchaudio_args')['sample_rate']
    if sample_rate!=target_sample_rate:
        audio = tau.transforms.Resample(orig_freq=sample_rate,new_freq=target_sample_rate)(waveform=audio)
        sample_rate = target_sample_rate

    # Cut and pad to fixed length
    audio_length = get_config('default_torchaudio_args')['audio_length']
    audio = AdjustAudioLength(audio, audio_length, force_trim=True)

    return audio, sample_rate

def TransformInput(audio, sample_rate, experiment):
    args = LoadJson(pjoin("experiment", experiment, "args.json"))
    train_args = MemberDict(args['train_args'])
    dataset_type = DATASET_MAPPING[train_args.dataset_type]
    return dataset_type.audio2data(audio, sample_rate)

def Sound2Synth(filepath, experiments):
    for experiment in experiments:
        args = LoadJson(pjoin("experiment", experiment, "args.json"))
        train_args = MemberDict(args['train_args'])
        interface = INTERFACE_MAPPING[train_args.synth]
        _, _, algorithm = experiment.split('_')
        audio, sample_rate = LoadProcessedAudio(filepath)
        print(audio.shape)
        data = TransformInput(audio, sample_rate, experiment)
        print(data[0][0]['main'].shape)
        input_tensor = {k:v.unsqueeze(0) for k,v in data[0][0].items()}
        print(input_tensor.keys())
        model = LoadModel(experiment); pred = model(input_tensor)
        assert(len(pred)==1); prd = pred[0]
        preset = interface.from_tensor(prd).to_dict()
        preset['ALGORITHM'] = int(algorithm)/31.
        return preset