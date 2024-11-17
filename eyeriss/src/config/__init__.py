from typing import Dict, Tuple

from .config import (
    Config,
    BaseConfig,
)
from .scanchain import ScanChain


class ConfigurationManager:
    pass


filter_settings: Dict = {
    "size": (3, 3),
    "channels": 3,
    "bits": 32,
}

image_settings: Dict = {
    "size": (512, 512),
    "channels": 3,
    "bits": 32,
}

filter_spad_settings: Dict = {
    "words": 1,
    "wordSize": 96,
}

image_spad_settings: Dict = {
    "words": 1,
    "wordSize": 16384,
}

psum_spad_settings: Dict = {
    "words": 1,
    "wordSize": 16320,
}
