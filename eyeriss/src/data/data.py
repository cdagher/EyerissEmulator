from typing import Optional

from src.instr import BaseInstr

import numpy as np

class Data:
    _instr: BaseInstr
    _data: np.ndarray

    def __init__(self, instr: int, data: Optional[int] = None):
        self._instr = instr
        self._data = data

    @property
    def instr(self) -> BaseInstr:
        return self._instr
    
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
        return f"Data(instr={self._instr}\n\tdata={self._data})"
    