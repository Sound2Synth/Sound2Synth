from .basic_utils import *
import torch
import torchaudio.functional as F
import torchaudio.transforms as T

def AudioToSpec(audio,
    n_fft       = get_config("default_torchaudio_args")["n_fft"],
    win_length  = get_config("default_torchaudio_args")["win_length"],
    hop_length  = get_config("default_torchaudio_args")["hop_length"],
    power       = get_config("default_torchaudio_args")["power"],
    center      = get_config("default_torchaudio_args")["center"],
    pad_mode    = get_config("default_torchaudio_args")["pad_mode"],
):
    return T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=power,
        center=center,
        pad_mode=pad_mode,
    )(audio)

def SpecToAudio(spectrogram,
    n_fft       = get_config("default_torchaudio_args")["n_fft"],
    win_length  = get_config("default_torchaudio_args")["win_length"],
    hop_length  = get_config("default_torchaudio_args")["hop_length"],
    power       = get_config("default_torchaudio_args")["power"],
):
    return T.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=power,
    )(spectrogram)

def AudioToMel(audio, 
    sample_rate = get_config("default_torchaudio_args")['sample_rate'],
    n_fft       = get_config("default_torchaudio_args")["n_fft"],
    win_length  = get_config("default_torchaudio_args")["win_length"],
    hop_length  = get_config("default_torchaudio_args")["hop_length"],
    power       = get_config("default_torchaudio_args")["power"],
    center      = get_config("default_torchaudio_args")["center"],
    pad_mode    = get_config("default_torchaudio_args")["pad_mode"],
    norm        = get_config("default_torchaudio_args")["norm"],
    onesided    = get_config("default_torchaudio_args")["onesided"],
    n_mels      = get_config("default_torchaudio_args")["n_mels"],
    mel_scale   = get_config("default_torchaudio_args")["mel_scale"],
):
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=power,
        center=center,
        pad_mode=pad_mode,
        norm=norm,
        onesided=onesided,
        n_mels=n_mels,
        mel_scale=mel_scale,
    )(audio)

def AudioToMFCC(audio, 
    sample_rate = get_config("default_torchaudio_args")['sample_rate'],
    n_mfcc      = get_config("default_torchaudio_args")['n_mfcc'],
    n_fft       = get_config("default_torchaudio_args")["n_fft"],
    win_length  = get_config("default_torchaudio_args")["win_length"],
    hop_length  = get_config("default_torchaudio_args")["hop_length"],
    power       = get_config("default_torchaudio_args")["power"],
    center      = get_config("default_torchaudio_args")["center"],
    pad_mode    = get_config("default_torchaudio_args")["pad_mode"],
    norm        = get_config("default_torchaudio_args")["norm"],
    onesided    = get_config("default_torchaudio_args")["onesided"],
    n_mels      = get_config("default_torchaudio_args")["n_mels"],
    mel_scale   = get_config("default_torchaudio_args")["mel_scale"],
):
    return T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft"      : n_fft,
            "win_length" : win_length,
            "hop_length" : hop_length,
            "power"      : power,
            "center"     : center,
            "pad_mode"   : pad_mode,
            "norm"       : norm,
            "onesided"   : onesided,
            "n_mels"     : n_mels,
            "mel_scale"  : mel_scale,
        }
    )(audio)

def TrimAudio(audio, trim_scale=1e-5):
    new_audio = audio.T; trim = len(new_audio)-1
    while trim>0 and (new_audio[trim]**2).mean() <= trim_scale**2:
        trim -= 1
    new_audio = new_audio[:trim+1,:]; return new_audio.T

def PadAudio(audio, target_length):
    new_audio = audio.T
    if len(new_audio) > target_length:
        raise ValueError(f"The audio is too long ({len(new_audio)}) to adjust to target length ({target_length}).")
    if len(new_audio) < target_length:
        pad = torch.zeros_like(new_audio[0])
        new_audio = torch.cat([new_audio,torch.stack([pad for _ in range(target_length-len(new_audio))],dim=0)],dim=0)
    return new_audio.T

def AdjustAudioLength(audio, target_length, trim_scale=1e-5, force_trim=False):
    new_audio = TrimAudio(audio, trim_scale=trim_scale).T
    if len(new_audio) > target_length and force_trim:
        new_audio = new_audio[:target_length,:]
    elif len(new_audio) > target_length:
        raise ValueError(f"The trimmed audio is too long ({len(new_audio)}) to adjust to target length ({target_length}). Try to increase 'trim_scale' or set 'force_trim' to true.")
    return PadAudio(new_audio.T, target_length)

def AlignAudioLength(audio1, audio2, mode='pad', trim_scale=1e-5, fixed=None):
    modes = ['first','second','trim','mid','pad','fixed']
    assert (mode in modes), f"Supported modes: {modes}"
    if mode == 'first':
        return audio1.clone(), AdjustAudioLength(audio2, target_length=audio1.shape[1])
    elif mode == 'second':
        return AdjustAudioLength(audio1, target_length=audio2.shape[1]), audio2.clone()
    elif mode == 'trim':
        audio1, audio2 = TrimAudio(audio1, trim_scale=trim_scale), TrimAudio(audio2, trim_scale=trim_scale)
        n = min(audio1.shape[1],audio2.shape[1])
        return AdjustAudioLength(audio1, target_length=n, force_trim=True), AdjustAudioLength(audio2, target_length=n, force_trim=True)
    elif mode == 'mid':
        audio1, audio2 = TrimAudio(audio1, trim_scale=trim_scale), TrimAudio(audio2, trim_scale=trim_scale)
        n = max(audio1.shape[1], audio2.shape[1])
        return PadAudio(audio1, n), PadAudio(audio2, n)
    elif mode == 'pad':
        n = max(audio1.shape[1], audio2.shape[1])
        return PadAudio(audio1, n), PadAudio(audio2, n)
    elif mode == 'fixed':
        return AdjustAudioLength(audio1, target_length=fixed, force_trim=True), AdjustAudioLength(audio2, target_length=fixed, force_trim=True)

def initialize_midi_settings(
    midi_settings_dir   = get_config('midi_settings_dir'),
    pitch_range         = get_config("default_midi_settings_args")['pitch_range'],
    velocity            = get_config("default_midi_settings_args")['velocity'],
    ticks_per_beat      = get_config("default_midi_settings_args")['ticks_per_beat'],
    duration_beats      = get_config("default_midi_settings_args")['duration_beats'],
    recording_beats     = get_config("default_midi_settings_args")['recording_beats'],
    bpm                 = get_config("default_midi_settings_args")['bpm'],
):
    CreateFolder(midi_settings_dir)
    for pitch in range(pitch_range[0],pitch_range[1]):
        midi = mido.MidiFile(type=0)
        track = mido.MidiTrack()
        midi.tracks.append(track)

        midi.ticks_per_beat = ticks_per_beat

        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
        track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=0))
        track.append(mido.Message('note_off', note=pitch, velocity=velocity, time=duration_beats*ticks_per_beat))
        track.append(mido.MetaMessage('end_of_track', time=(recording_beats-duration_beats)*ticks_per_beat))

        midi.save(pjoin(midi_settings_dir,f'{pitch}.mid'))