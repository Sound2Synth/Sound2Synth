from .base import AudioDataset
from .spec import SpecDataset, MelDataset, MFCCDataset
from .multimodal import MultiModalDataset

DATASET_MAPPING = {
    'audio': AudioDataset,
    'spec': SpecDataset,
    'mel': MelDataset,
    'mfcc': MFCCDataset,
    'multimodal': MultiModalDataset,
}