from typing import List, Tuple, Dict, Optional

from src.memory import GlobalBuffer
from src.data import Data
from src.pe import (
    PE,
    SpatialArray,
)
from src.instr import (
    TerminateInstr,
)

class NoC:
    _sa: SpatialArray
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

        self._sa = SpatialArray(rows, cols)

        self._gb = GlobalBuffer(1024, 16)

    def __getitem__(self, key: Tuple[int, int]) -> PE:
        return self._sa[key]
    
    def __setitem__(self, key: Tuple[int, int], value: PE):
        self._sa[key] = value

    def __iter__(self):
        for pe in self._sa:
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
    
    # def start(self):
    #     for pe in self:
    #         pe.start()

    # def join(self):
    #     for pe in self:
    #         pe.join()

    # def terminate(self):
    #     for pe in self:
    #         pe.put(Data(TerminateInstr()))
    #         pe.terminate()

    # def close(self):
    #     for pe in self:
    #         pe.close()

    def multicast(
            self,
            data: Data,
            to: Optional[List[PE]] = None):
        """
        Multicasts data to all PEs.

        Args:
            data (Data): Data to multicast.
        """
        
        if to is not None:
            for pe in self:
                pe(data)

        else:
            for pe in self:
                pe(data)

    def diagonal_connection(self, src: Tuple[int, int]) -> PE | None:
        """
        Gets the diagonal PE from the source PE.

        If the source PE is in the top row, the diagonal PE is None.
        If the source PE is in the rightmost column, the diagonal PE is None.
        
        returns:
            PE: Diagonal PE. None if the source PE is in the top row or rightmost column.
        """

        row, col = src
        if row == 0 or col == self._cols - 1:
            return None
        return self[(row - 1, col + 1)]
    
    def diagonal_connections(self, src: Tuple[int, int]) -> List[PE]:
        """
        Gets the diagonal PEs from the source PE.

        If the source PE is in the top row, the diagonal PEs are None.
        If the source PE is in the rightmost column, the diagonal PEs are None.
        
        returns:
            List[PE]: Diagonal PEs. Empty list if the source PE is in the top row or rightmost column.
        """

        row, col = src
        if row == 0 or col == self._cols - 1:
            return []
        return [self[row-1, col+1]] + self.diagonal_connections((row - 1, col + 1))
