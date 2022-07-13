# Sound2Synth: Interpreting Sound via FM Synthesizer Parameters Estimation

This is the code repo for the paper "Sound2Synth: Interpreting Sound via FM Synthesizer Parameters Estimation" (IJCAI 2022 AI, THE ARTS AND CREATIVITY (SPECIAL TRACK)).

## Demo

Demo clips, demo videos, and demo songs are held on dropbox, please obtain them from [this link](https://www.dropbox.com/sh/fs9fu0y6iw45u90/AACfievRzwBMoOmGU4zLxxDYa?dl=0).

<br/>

## Setup

**System Requirements**: MacOS >= 10.13. Python >= 3.7.

1. Download source code (this repo):
```bash
git clone https://github.com/Sound2Synth/Sound2Synth.git
```

2. Install requirements:
```bash
cd ./Sound2Synth/
pip3 install -r requirements.txt
```

3. Configure paths and parameters in `config.json`. Please find and replace keywords including `<USERNAME>`, `<SERVER>`. To use `wandb` logging, you may also need to change `<WANDB_ENTITY>`.

4. Fix `PyAU` dependency according to your needs. For more details, please refer to [this document](./PyAU/README.txt). It is recommended to run all programs related to synthesizer access with the following environment (implemented in `run.sh`):
```bash
export PYTHONPATH="./"
export QT_MAC_WANTS_LAYER=1
```

UPD. 2022.07.13:

We have released a plug-in for replacing the PyAU requirement. Please check out [Tracktion4Sound2Synth](https://github.com/Sound2Synth/Tracktion4Sound2Synth).

<br/>

## Dataset Generation

1. This repo does not hold Dexed preset libraries. Please obtain organized Dexed preset libraries from online communities or any other way. For example, you can download some presets from [this link](https://www.audiopluginguy.com/free-dexed-plus-tonnes-patches/). Put the obtained presets under `data_dir` as what is set in the `config.json` file.

2. Modify and run `preprocess/dexed.py` to preprocess datasets according to your needs.
```bash
bash run.sh preprocess/dexed.py
```

<br/>

## Training

An extra remote server (besides your MacOS environment) with GPU is recommended for training the model, you need to set up the server path and ssh config before the training procedure begins.

1. Please check out the arguments specified in `manager.py`, `train.py`, and `test.py`. `manager.py` is responsible for deploying code and data to a remote server, initiating the training process, monitoring training procedures, and summarizing results. Each training procedure will be synced and saved locally on the MacOS under `examples/`, with checkpoints, validation statistics, and inference (test) statistics stored.

2. Modify and run `manager.py` according to your needs. For example, the following command trains a multi-modal network using all 3 dataset generation methods on Algorithm $0$ (notice that dataset must be fully preprocessed before training):
```bash
python3 manager.py --algorithm 0 --split aug-rand --backbone multi-spec --add-to-ensemble --classifier parameter --back-sync --clear --cuda <YOUR_CUDA_DEVICES>
```

<br/>

## Application

Full Sound2Synth application consists of three parts:

1. A Dexed plug-in client. Please refer to the `Sound2Synth Dexed Plug-In` [repository](https://github.com/Sound2Synth/Sound2Synth-Plug-Ins/) for accessing the plug-in and installation guide. This plug-in is modified from the original Dexed plug-in. It contains only a web protocol for sending synthesizer parameter estimation requests, instead of a full Sound2Synth pipeline.

2. A trained Sound2Synth model or an ensemble of trained Sound2Synth models. Please refer to the "Training" section for building such models. Pre-trained checkpoints may be released in the future.

3. A Sound2Synth server. It contains code for responding and processing Dexed plug-in client requests, calling models, and returning parameters to the client. This is contained in the `server/` folder. To initiate a local Sound2Synth server, you can simply use:
```bash
bash server.sh
```

This bash command starts a local server at `https://127.0.0.1:1234/` by default, you can modify the address in server code according to your needs. Notice that you need a trained model or model ensemble to function the server. All failures of calling the server will be printed in the console for you to debug the server instance.

To use the plug-in, first load the Dexed plug-in client in any DAW that supports it. Then input the server address and `PING` the server to connect. After successfully building a connection, you can use `MATCH` to upload sound waveform, and the parameters of the Dexed plug-in will automatically be re-assigned after the computation finishes. For clearer instruction, please refer to the demo video we provided in `Sound2Synth Demos`.

An out-of-box plug-in is still under development and will be released. Please stay tuned.

<br/>

## Checkpoints & Results

Trained checkpoints for directly using the Sound2SYnth plug-in server will be uploaded soon. The checkpoints and experiment results will be maintained and updated for progress.

<br/>

## Citation

```
@inproceedings{sound2synth,
  title     = {Sound2Synth: Interpreting Sound via FM Synthesizer Parameters Estimation},
  author    = {Chen, Zui and Jing, Yansen and Yuan, Shengcheng and Xu, Yifei and Wu, Jian and Zhao, Hang},
  booktitle = {AI, the Arts and Creativity – Special Track of the 31st International Joint Conference on Artificial Intelligence},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  year      = {2022},
}
```