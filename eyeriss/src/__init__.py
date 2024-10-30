from config import Config
from interface import CLI
from data import Data
from noc import NoC
from memory import Memory, Hierarchy
from pe import PE

from enum import Enum


class Instruction(Enum):
    READ = 1
    WRITE = 2
    COMPUTE = 3
    PE_WRITE_FILTER = 4
    PE_WRITE_IFMAP = 5
    PE_Write_IFMAP = 6
    PE_READ_PSUM = 7


def relu(x):
    return max(0, x)
