from typing import Optional, Tuple

from src.instr import BaseInstr
from src.addr import Address

import numpy as np

class Data:
    _instr: BaseInstr   # instruction
    _addr: Address      # address
    _data: np.ndarray   # data array

    def __init__(
            self,
            instr: BaseInstr,
            addr: Optional[Address] = None,
            data: Optional[np.ndarray] = None):
        self._instr = instr
        self._addr = addr
        self._data = data

    @property
    def instr(self) -> BaseInstr:
        return self._instr
    
    @property
    def addr(self) -> Address:
        return self._addr

    @property
    def data(self) -> np.ndarray:
        return self._data
    
    def __repr__(self):
        return f"Data(instr={self._instr}, data={self._data})"
    
    def __eq__(self, other):
        if isinstance(other, Data):
            return self._instr == other._instr and self._data == other._data
        return False
    
    def __hash__(self):
        return hash((self._instr, self._data))
    
    def __lt__(self, other):
        if isinstance(other, Data):
            return self._instr < other._instr
        return NotImplemented
    
    def __str__(self) -> str:
        return f"Data(instr={self._instr}\n\taddr={self._addr}\n\tdata={self._data})"
    