from typing import List, Tuple, Dict

from src.memory import GlobalBuffer
from src.data import Data
from src.pe import PE

class NoC:
    _pes: List[PE]
    _pe_map: Dict[Tuple[int, int], PE]
    _rows: int
    _cols: int
    _latency: float
    _energy: float

    _gb: GlobalBuffer

    def __init__(
            self,
            rows: int,
            cols: int,
            latency: float,
            energy: float
            # TODO: Add PE map config
        ):

        self._rows = rows
        self._cols = cols
        self._latency = latency
        self._energy = energy

        self._pes = [[PE(i * cols + j) for j in range(cols)] for i in range(rows)]
        self._pe_map = {
            (i, j): self._pes[i][j]
            for i in range(rows)
            for j in range(cols)
        }

        self._gb = GlobalBuffer(1024, 16)

    def __getitem__(self, key: Tuple[int, int]) -> PE:
        return self._pe_map[key]
    
    def __setitem__(self, key: Tuple[int, int], value: PE):
        self._pe_map[key] = value

    def __iter__(self):
        for row in self._pes:
            for pe in row:
                yield pe

    def __len__(self):
        return self._rows * self._cols
    
    @property
    def size(self):
        return self._rows, self._cols
    
    @property
    def latency(self):
        return self._latency
    
    @property
    def energy(self):
        return self._energy
    
    @property
    def gb(self):
        return self._gb
    
    def start(self):
        for pe in self:
            pe.start()

    def join(self):
        for pe in self:
            pe.join()

    def terminate(self):
        for pe in self:
            pe.terminate()

    def close(self):
        for pe in self:
            pe.close()

    def multicast(self, data: Data):
        """
        Multicasts data to all PEs.

        Args:
            data (Data): Data to multicast.
        """
        
        for pe in self:
            pe.put(data)

    def p2p(self, src: Tuple[int, int], dst: Tuple[int, int]):
        """
        Sends data from one PE to another.

        Args:
            src (Tuple[int, int]): Source PE coordinates.
            dst (Tuple[int, int]): Destination PE coordinates.
        """

        data = self[src].get()
        self[dst].put(data)
        