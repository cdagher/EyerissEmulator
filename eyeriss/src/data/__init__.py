from typing import Tuple, Optional

from data import Data
from dataflow import DataFlow, RSDataFlow
from gating import Gate


class Movement:
    pass

class Address:
    """
    Address class representing a memory address.

    Attributes:
        _address (Tuple[int, ...]): A tuple representing the address in memory.

    The elements of the tuple correspond to different dimensions of the memory, 
    eg. (page, block, word), (block, word), or (word).

    Methods:
        address() -> Tuple[int, ...]:
            Returns the address as a tuple.
    """
    _address: Tuple[int, ...]

    def __init__(self, address: Optional[Tuple[int, ...]] = None):
        self._address = address

    @property
    def address(self) -> Tuple[int, ...]:
        return self._address
    
    @property
    def shape(self) -> int:
        return len(self._address) if self._address else 0
    
    def __repr__(self):
        return f"Address({self._address})"
    
    def __eq__(self, other):
        if isinstance(other, Address):
            return self._address == other._address
        return False
    
    def __hash__(self):
        return hash(self._address)
    
    def __str__(self) -> str:
        return f"Address({self._address})"
    
    def __lt__(self, other):
        if isinstance(other, Address):
            return self._address < other._address
        return NotImplemented
    
    def __len__(self):
        return len(self._address) if self._address else 0
    
    def __getitem__(self, index: int) -> int:
        return self._address[index]
    
    def __iter__(self):
        return iter(self._address) if self._address else iter([])
    
    def __contains__(self, item: int) -> bool:
        return item in self._address if self._address else False
    
    def __add__(self, other: 'Address') -> 'Address':
        if isinstance(other, Address):
            return Address(tuple(a + b for a, b in zip(self._address, other._address)))
        return NotImplemented
