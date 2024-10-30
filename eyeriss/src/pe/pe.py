from typing import override

import multiprocessing as mp

from src import Instruction

from src.memory import SPAD
from src.data import Data

import numpy as np


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
    _inQueue: mp.Queue[Data]
    _outQueue: mp.Queue[Data]

    def __init__(
            self,
            id: int,
            inQueue: mp.Queue,
            outQueue: mp.Queue):
        self._id = id
        self._filter = SPAD(words=224, wordSize=16)
        self._ifmap = SPAD(words=12, wordSize=16)
        self._psum = SPAD(words=24, wordSize=16)
        
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
        # Process input data and produce output data
        while True:
            data = self._inQueue.get()
            if data.instr == Instruction.PE_WRITE_FILTER:
                self._filter.write(..., data.data)
            elif data.instr == Instruction.PE_WRITE_IFMAP:
                self._ifmap.write(..., data.data)
            elif data.instr == Instruction.PE_READ_PSUM:
                psum_data = self._psum.read(...)
                self._outQueue.put(psum_data)
            elif data.instr == Instruction.COMPUTE:
                self.compute()
                self._outQueue.put(self._psum.read(...))  # Placeholder for actual result

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

    def _conv2d(self, imageRow, filterWeight, imageNum, filterNum):
        # Perform 2D convolution logic
        result = np.zeros((imageRow.shape[0] - filterWeight.shape[0] + 1,
                           imageRow.shape[1] - filterWeight.shape[1] + 1))

        if filterNum == 1 and imageNum == 1:
            return self._conv1d(imageRow, filterWeight)

        if filterNum == 1:
            pics = np.hsplit(imageRow, imageNum)

            for x in pics:
                rx = self._conv1d(x, filterWeight)
                result[x] = rx

            return result
        
        if imageNum == 1:
            filterWeight = np.reshape(filterWeight, (int(filterWeight.shape[0] / filterNum), filterNum))
            fits = np.array(filterWeight.T)

            for x in fits:
                rx = self._conv1d(imageRow, x)
                result[x] = rx

            result = result.T
            result = np.reshape(result, (1, result.shape[0], result.shape[1]))
            return result

    def _conv(self, imageRow, filterWeight, imageNum, filterNum):
        # Perform convolution logic
        return self._conv2d(imageRow, filterWeight, imageNum, filterNum)

    def compute(self):
        # Perform computation logic
        if self._ifmap.is_empty() or self._filter.is_empty():
            return
        if self._psum.is_empty():
            self._psum.write(..., self._conv(self._ifmap.read(), self._filter.read(), 1, 1))

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

    _element: _ElementProc
    _input: mp.Queue
    _output: mp.Queue

    def __init__(self, id: int):
        self.writer = mp.Queue()
        self.reader = mp.Queue()
        self._element = _ElementProc(id, self.writer, self.reader)

    def start(self):
        self._element.start()

    def join(self):
        self._element.join()

    def put(self, data: Data):
        self.reader.put(data)

    def get(self) -> Data:
        return self.writer.get()
    
    def terminate(self):
        self._element.terminate()

    def close(self):
        self.reader.close()
        self.writer.close()
