from typing import Tuple, Optional

from .interface import (
    CLI,
)
from .config import (
    Config,
    BaseConfig,
)
from .data import Data
from .noc import NoC
from .pe import PE
from .addr import Address

from enum import Enum

import numpy as np

from src.memory import (
    GlobalBuffer
)
from src.instr import (
    BaseInstr,
    ComputeInstr,
    PEWriteFilterInstr,
    PEWriteIfmapInstr,
    PEWritePsumInstr,
    PEReadPsumInstr,
    PEAddPsumInstr,
    GLBReadFilterInstr,
    GLBWriteFilterInstr,
    GLBReadIFMAPInstr,
    GLBWriteIFMAPInstr,
    GLBReadPSUMInstr,
    GLBWritePSUMInstr,
    GLBReadOfMapInstr
)

    
class Eyeriss:
    _noc: NoC
    _glb: GlobalBuffer

    _filter_set:  bool              # has the filter been set yet
    _filter_size: Tuple[int, int]   # filter size
    _image_set:   bool              # has the image been set yet
    _image_size:  Tuple[int, int]   # image size

    def __init__(self, rows: int, cols: int):
        self._noc = NoC(rows, cols, 0, 0)
        self._glb = GlobalBuffer()

        self._filter_set = False
        self._filter_size = (0, 0)
        self._image_set = False
        self._image_size = (0, 0)

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.rows, config.cols)

    def noc(self) -> NoC:
        return self._noc
    
    def glb(self) -> GlobalBuffer:
        return self._glb
    
    def set_filter(self, filter: np.ndarray):
        frows, fcols = filter.shape
        self._filter_size = (frows, fcols)

        if frows > self._noc.size[0] or fcols > self._noc.size[1]:
            raise ValueError("Filter size must not exceed PE array size")
        
        # send filter rows to all PEs in the corresponding rows
        for i in range(frows):
            dest = [self._noc[i, j] for j in range(self._noc.size[1])]
            self._noc.multicast(
                data=Data(PEWriteFilterInstr(Address((0, 0)), filter[i])),
                to=dest
            )

        self._filter_set = True

    def set_image(self, image: np.ndarray):
        irows, _ = image.shape
        self._image_size = (irows, image.shape[1])

        # check if the image rows exceed the PE array size
        if irows > sum(self._noc.size):
            raise ValueError(f"Image rows must not exceed PE array size\
                             \n\t{irows} > {sum(self._noc.size)}")
        
        # check if the image rows exceed the PE array columns after wrapping
        if irows - self._filter_size[0] > self._noc.size[1]:
            raise ValueError(f"Image rows must not exceed PE array columns\
                             \n\t{irows} - {self._filter_size[0]} > {self._noc.size[1]}")
        
        # fill PEs with zeros to clear previous data
        for pe in self._noc:
            pe(Data(PEWriteIfmapInstr(Address((0, 0)), np.zeros(image.shape[1]))))

        # send image rows to the first column of PEs
        for i in range(min(irows, self._filter_size[0])):
            dest = [self._noc[i, 0]]
            
            # append the PEs in the diagonal to the destination list
            diag = self._noc.diagonal_connections((i, 0))
            dest.extend(diag)
            
            # multicast the image row to the PEs in the destination list
            self._noc.multicast(
                data=Data(PEWriteIfmapInstr(Address((0, 0)), image[i])),
                to=dest
            )

        # if the whole image has been sent, return
        if irows <= self._filter_size[0]:
            self._image_set = True
            return

        # image has more rows than PE array. Send remainder to the bottom row
        for i in range(self._filter_size[0], irows):
            row = self._noc.size[0]-1
            col = i - self._noc.size[0]
            dest = [self._noc[row, col]]

            # append the PEs in the diagonal to the destination list
            diag = self._noc.diagonal_connections((row, col))
            dest.extend(diag)

            self._noc.multicast(
                data=Data(PEWriteIfmapInstr(Address((0, 0)), image[i])),
                to=dest
            )

        self._image_set = True

    def is_ready(self) -> bool:
        return self._filter_set and self._image_set
    
    def reset(self):
        self._filter_set = False
        self._image_set = False

    def compute(
            self,
            image: Optional[np.ndarray] = None,
            filter: Optional[np.ndarray] = None
        ) -> np.ndarray:
        if filter is not None:
            self.set_filter(filter)
        if image is not None:
            self.set_image(image)

        if not self.is_ready():
            raise ValueError("Filter and image must be set before computing")

        # compute the dot product of the filter and image
        for pe in self._noc:
            # print(f"Sending compute istr to PE {pe.id}")
            pe(Data(ComputeInstr()))

        # perform 2d convolution by computing the dot product of the filter and
        # image rows. Pass psums vertically to the previous row
        for i in range(self._filter_size[0], 1, -1):
            i -= 1 # adjust for 0-based indexing

            # for each PE in the row, send the psum to the previous row
            for j in range(self._noc.size[1]):
                pe = self._noc[i, j]
                # pe(Data(PEReadPsumInstr(Address((0, 0)))))
                psum = pe(Data(PEReadPsumInstr(Address((0, 0)))))
                dest = self._noc[i-1, j]
                dest(Data(PEAddPsumInstr(Address((0, 0)), psum)))

        # get the final psums from the first row of PEs
        psums = []
        conv_size = self._image_size[0] - self._filter_size[0] + 1
        nsums = min(conv_size, self._noc.size[1])
        for j in range(nsums):
            pe = self._noc[0, j]
            # pe(Data(PEReadPsumInstr(Address((0, 0)))))
            psum = pe(Data(PEReadPsumInstr(Address((0, 0)))))
            psums.append(psum)

        return np.array(psums)

    def __getitem__(self, key: Tuple[int, int]) -> PE:
        return self._noc[key]
    
    def __setitem__(self, key: Tuple[int, int], value: PE):
        self._noc[key] = value

    def __iter__(self):
        for pe in self._noc:
            yield pe

    def __len__(self):
        return len(self._noc)
    
    @property
    def size(self):
        return self._noc.size
    
    def __str__(self) -> str:
        return f"Eyeriss(size={self.size})"
    
    def __repr__(self) -> str:
        return str(self)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Eyeriss):
            return self.size == other.size
        return False
    
    def __hash__(self) -> int:
        return hash(self.size)
    
    def __call__(
            self,
            image: Optional[np.ndarray] = None,
            filter: Optional[np.ndarray] = None
        ) -> Data:

        return self.compute(image, filter)


def relu(x):
    return max(0, x)
