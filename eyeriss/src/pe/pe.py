from typing import override

import multiprocessing as mp

from src.instr import (
    TerminateInstr,
    ComputeInstr,
    PEWriteFilterInstr,
    PEWriteIfmapInstr,
    PEWritePsumInstr,
    PEReadPsumInstr,
    PEAddPsumInstr,
)
from src.memory import SPAD
from src.data import Data
from src.config import (
    filter_settings,
    image_settings,
    filter_spad_settings,
    image_spad_settings,
    psum_spad_settings,
)
from src.addr import Address

import numpy as np

import warnings


class ControlUnit:
    def __init__(self):
        # Initialize control unit parameters
        pass

    def control(self):
        # Control logic for the processing element
        pass

class _ElementProc(mp.Process):
    """
    Processing Element (PE) that performs computations on input data.

    Attributes:
        _id (int): Unique identifier for the processing element.
        _filter (SPAD): Filter memory for storing weights.
        _ifmap (SPAD): Input feature map memory.
        _psum (SPAD): Partial sum memory.
        _control_unit (ControlUnit): Control unit for managing operations.

    Methods:
        __call__(input: mp.Queue[Data], output: mp.Queue[Data]):
            Processes input data and produces output data.
    """

    _id: int
    _filter: SPAD
    _ifmap: SPAD
    _psum: SPAD
    _inQueue: mp.Queue  # input queue of Data
    _outQueue: mp.Queue # output queue of Data

    def __init__(
            self,
            id: int,
            inQueue: mp.Queue,
            outQueue: mp.Queue):
        super().__init__()

        warnings.warn(
            "_ElementProc is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )

        self._id = id
        # self._filter = SPAD(words=224, wordSize=16)
        # self._ifmap = SPAD(words=12, wordSize=16)
        # self._psum = SPAD(words=24, wordSize=16)
        self._filter = SPAD(**filter_spad_settings)
        self._ifmap = SPAD(**image_spad_settings)
        self._psum = SPAD(**psum_spad_settings)
        
        self._inQueue = inQueue
        self._outQueue = outQueue

    @override
    def run(self):
        """
        Processes input data and produces output data.

        Args:
            input (mp.Queue[Data]): Input queue for receiving data.
            output (mp.Queue[Data]): Output queue for sending data.
        """

        run = True

        # Process input data and produce output data
        while run:
            data = self._inQueue.get()
            print(f"PE {self._id}: Received data {data}")
            if data.instr == PEWriteFilterInstr:
                self._filter.write(data.addr, data.data)
            elif data.instr == PEWriteIfmapInstr:
                self._ifmap.write(data.addr, data.data)
            elif data.instr == PEReadPsumInstr:
                psum_data = self._psum.read(data.addr)
                self._outQueue.put(psum_data)
            elif data.instr == PEWritePsumInstr:
                self._psum.write(data.addr, data.data)
            elif data.instr == PEAddPsumInstr:
                psum = self._psum.read(data.addr)
                self._psum.write(data.addr, psum + data.data)
            elif data.instr == ComputeInstr:
                self.compute()
            elif data.instr == TerminateInstr:
                self._psum.terminate()
                self._filter.terminate()
                self._ifmap.terminate()
                
                self._psum.close()
                self._filter.close()
                self._ifmap.close()

                run = False

    def _conv1d(self, row, weight):
        # Perform 1D convolution logic
        result = np.zeros((self._ifmap.shape[0] - self._filter.shape[0] + 1,))
        for x in range(0, len(row) - 1 + len(weight)):
            y = x + len(weight)
            if y > len(row):
                break
            r = row[x:y] * weight
            result[x] = np.sum(r)
        return result

    def _conv(self, imageRow, filterWeight):
        # Perform convolution logic
        return self._conv1d(imageRow, filterWeight)

    def compute(self):
        # Perform computation logic
        if self._ifmap.is_empty() or self._filter.is_empty():
            print(f"PE {self._id}: IFMAP or filter is empty")
            return
        if self._psum.is_empty():
            print(f"PE {self._id}: Running convolution")
            psum = self._conv(self._ifmap.read((0, 0)), self._filter.read((0, 0)))
            self._psum.write((0, 0), psum)

class PE:
    """
    Processing Element (PE) wrapper for multiprocessing.

    Attributes:
        _element (Element): Instance of the Element class.
        _proc (mp.Process): Process for running the Element.
        _input (mp.Queue): Input queue for sending data to the Element.
        _output (mp.Queue): Output queue for receiving data from the Element.

    Methods:
        start(): Starts the processing element process.
        join(): Waits for the processing element process to finish.
        put(data: Data): Sends data to the processing element.
        get() -> Data: Receives data from the processing element.
        terminate(): Terminates the processing element process.
        close(): Closes the input and output queues.
    """

    _id: int

    _filter: SPAD
    _ifmap: SPAD
    _psum: SPAD

    def __init__(self, id: int):
        self._id = id
        self._filter = SPAD(**filter_spad_settings)
        self._ifmap = SPAD(**image_spad_settings)
        self._psum = SPAD(**psum_spad_settings)

    @property
    def id(self):
        return self._id
    
    def filter(self) -> SPAD:
        return self._filter
    
    def ifmap(self) -> SPAD:
        return self._ifmap
    
    def psum(self) -> SPAD:
        return self._psum

    def _conv1d(self, imageRow: np.ndarray, filterRow: np.ndarray):
        # Perform 1D convolution logic
        result = np.zeros((imageRow.shape[0] - filterRow.shape[0] + 1,))
        for x in range(0, len(imageRow) - 1 + len(filterRow)):
            y = x + len(filterRow)
            if y > len(imageRow):
                break
            r = imageRow[x:y] * filterRow
            result[x] = np.sum(r)
        return result

    def _conv(self, imageRow: np.ndarray, filterRow: np.ndarray):
        # Perform convolution logic
        return self._conv1d(imageRow, filterRow)

    def compute(self):
        # Perform computation logic
        if self._ifmap.is_empty() or self._filter.is_empty():
            print(f"PE {self._id}: IFMAP or filter is empty")
            return
        if self._psum.is_empty():
            # print(f"PE {self._id}: Running convolution")
            psum = self._conv(
                self._ifmap.read(Address((0, 0))),
                self._filter.read(Address((0, 0)))
            )
            self._psum.write(Address((0, 0)), psum)

    def __call__(self, data: Data):
        
        instr = data.instr

        if isinstance(instr, PEWriteFilterInstr):
            self._filter.write(instr.address, instr.data)
            return None
        
        elif isinstance(instr, PEWriteIfmapInstr):
            self._ifmap.write(instr.address, instr.data)
            return None
        
        elif isinstance(instr, PEReadPsumInstr):
            psum_data = self._psum.read(instr.address)
            return psum_data
        
        elif isinstance(instr, PEWritePsumInstr):
            self._psum.write(instr.address, instr.data)
            return None
        
        elif isinstance(instr, PEAddPsumInstr):
            psum = self._psum.read(instr.address)
            self._psum.write(instr.address, psum + instr.data)
            return None
        
        elif isinstance(instr, ComputeInstr):
            self.compute()
            return None
        
        elif isinstance(instr, TerminateInstr):
            return None
        
    def __str__(self):
        return f"PE({self._id})"
    
    def __repr__(self):
        return str(self)