from .base import BaseInterface, ParameterSpaceLoss
from .dexed import DexedInterface

INTERFACE_MAPPING = {
    "Dexed": DexedInterface,
}