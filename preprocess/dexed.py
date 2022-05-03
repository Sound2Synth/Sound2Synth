from utils import *
from dataset import *

import tqdm
import numpy as np
np.random.seed(20001101)
def CreateDexedPresetAugmentDataset(presets, per_iterations, max_steps, dataset, split='dexed01', algorithm=0, preserve=True):
    from interface.dexed import DexedInterface, regression_parameters, classification_parameters
    local_dst = p2s(pjoin(get_config("data_dir"),dataset),f=True); CreateFolder(local_dst)
    remote_dst = p2s(pjoin(get_config("server_data_dir"),dataset),f=True)
    h = RSYNC(local_dst,remote_dst,args="-q",wait=False); h.wait(timeout=5)
    rsync_failed = set()
    print(DexedInterface.profile())
    with Synthesizer("Dexed") as S:
        c = 0
        pbar = TQDM(presets)
        for preset_file in pbar:
            flag = preserve
            pbar.set_description(preset_file)
            spbar = tqdm.tqdm(total=per_iterations*max_steps)
            for i in range(per_iterations):
                if preset_file is not None:
                    S.load_plist(preset_file)
                params = DexedInterface.from_synth(S).to_dict()
                keyset = list(params.keys())
                for j in range(max_steps):
                    m = 0.; err = 0
                    if not flag:
                        key = np.random.choice(keyset)
                        while key.startswith("OP") and (not key.endswith("SWITCH")) and params[key[:4]+"SWITCH"]<0.5:
                            key = np.random.choice(keyset)
                        params[key] = np.random.rand()
                        flag = False
                    preset = DexedInterface.from_dict(params)
                    preset.set_algorithm(algorithm/31)
                    preset.to_synth(S)
                    audio, sample_rate = S.render()
                    if sample_rate!=48000:
                        audio = tau.transforms.Resample(orig_freq=sample_rate,new_freq=48000)(waveform=audio)
                        sample_rate = 48000
                    m = float(audio.abs().max())
                    if m>0.01:
                        spbar.set_description("i[%02d]j[%02d]c[%06d][%.3f]"%(i,j,c,m))
                        name = "{:s}.{:06d}.{:06d}".format(split,c,sample_rate)
                        SaveJson(params, pjoin(local_dst,name+".json"), indent=4)
                        tau.save(pjoin(local_dst,name+ ".wav"),audio,sample_rate)
                        try:
                            h = RSYNC(pjoin(local_dst,name+".json"),pjoin(remote_dst,name+".json"),args="-q",wait=False); h.wait(timeout=5)
                            h = RSYNC(pjoin(local_dst,name+ ".wav"),pjoin(remote_dst,name+ ".wav"),args="-q",wait=False); h.wait(timeout=5)
                            Delete(pjoin(local_dst,name+ ".wav"),rm=True); Delete(pjoin(local_dst,name+".json"),rm=True)
                        except:
                            rsync_failed.add(name)
                        S.clear_cache()
                        c += 1; spbar.update()
    while len(rsync_failed):
        for name in TQDM(list(rsync_failed)):
            try:
                h = RSYNC(pjoin(local_dst,name+".json"),pjoin(remote_dst,name+".json"),args="-q",wait=False); h.wait(timeout=5)
                h = RSYNC(pjoin(local_dst,name+ ".wav"),pjoin(remote_dst,name+ ".wav"),args="-q",wait=False); h.wait(timeout=5)
                rsync_failed.remove(name); Delete(pjoin(local_dst,name+ ".wav"),rm=True); Delete(pjoin(local_dst,name+".json"),rm=True)
            except:
                pass

def CreateDexedRandomDataset(per_iterations, max_steps, dataset, split, algorithm):
    CreateDexedPresetAugmentDataset([None],per_iterations,max_steps,dataset,split,algorithm,False)

if __name__=="__main__":
    initialize_midi_settings()

    presets_by_algorithm = [[] for _ in range(32)]
    with Synthesizer("Dexed") as S:
        for preset_file in EnumFiles(get_config("data_dir")):
            if Format(preset_file)=="plist":
                S.load_plist(preset_file)
                algorithm = int(round(S.parameters['ALGORITHM'].current_value*31))
                presets_by_algorithm[algorithm].append(preset_file)
    for algorithm, presets in enumerate(presets_by_algorithm):
        CreateDexedPresetAugmentDataset(
            presets = presets,
            per_iterations = 1,
            max_steps = 1,
            dataset = "Dexed",
            split = "LARGE%02d"%algorithm,
            algorithm = algorithm,
            preserve = True,
        )

    for algorithm in TQDM(32):
        CreateDexedRandomDataset(
            per_iterations = 100,
            max_steps = 10,
            dataset = "Dexed",
            split = "RAND%02d"%algorithm,
            algorithm = algorithm,
        )