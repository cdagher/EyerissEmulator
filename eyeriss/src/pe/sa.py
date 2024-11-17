from .pe import PE
from typing import List, Tuple

class SpatialArray:
    _rows: int
    _cols: int
    _pes: List[List[PE]]

    def __init__(self, rows: int, cols: int):
        self._rows = rows
        self._cols = cols
        self._pes = [[PE(i * cols + j) for j in range(cols)] for i in range(rows)]

    def __getitem__(self, key: Tuple[int, int]) -> PE:
        return self._pes[key[0]][key[1]]
    
    def __setitem__(self, key: Tuple[int, int], value: PE):
        self._pes[key[0]][key[1]] = value

    def __iter__(self):
        for row in self._pes:
            for pe in row:
                yield pe

    def __len__(self):
        return self._rows * self._cols
    
    @property
    def size(self):
        return self._rows, self._cols
    
    def __str__(self) -> str:
        return f"SpatialArray(rows={self._rows}, cols={self._cols})"
    
    def __repr__(self) -> str:
        return str(self)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, SpatialArray):
            return self._rows == other._rows and self._cols == other._cols
        return False
    
    def __hash__(self) -> int:
        return hash((self._rows, self._cols))