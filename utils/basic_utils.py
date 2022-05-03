import mido
import torchaudio as tau
import librosa as lbrs
from pyheaven import *

DEFAULT_CONFIG_PATH = "config.json"

def get_config(key, default=None, config_path=DEFAULT_CONFIG_PATH):
    config = LoadJson(config_path); return config[key] if key in config else default

def set_config(key, value, override=True, config_path=DEFAULT_CONFIG_PATH):
    config = LoadJson(config_path); config[key] = value if override or (key not in config) else config[key]; SaveJson(config,config_path,indent=4)
