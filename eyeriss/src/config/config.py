from abc import ABC, abstractmethod

class Config(ABC):
    
    _rows: int
    _cols: int
    _latency: float
    _energy: float

    def __init__(
            self,
            rows: int,
            cols: int,
            latency: float,
            energy: float
    ):

        self._rows = rows
        self._cols = cols
        self._latency = latency
        self._energy = energy

    @property
    def size(self):
        return self._rows, self._cols
    
    @property
    def rows(self):
        return self._rows
    
    @property
    def cols(self):
        return self._cols
    
    @property
    def latency(self):
        return self._latency
    
    @property
    def energy(self):
        return self._energy
    
class BaseConfig(Config):
    def __init__(self):
        super().__init__(12, 14, 0, 0)
