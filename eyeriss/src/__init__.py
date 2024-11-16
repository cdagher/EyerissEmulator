from config import Config
from interface import CLI
from data import Data
from noc import NoC
from memory import MemoryBlock, Hierarchy
from pe import PE

from enum import Enum


def relu(x):
    return max(0, x)
