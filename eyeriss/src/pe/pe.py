import multiprocessing as mp

from src.memory import SPAD
from src.data import Data


class ControlUnit:
    def __init__(self):
        # Initialize control unit parameters
        pass

    def control(self):
        # Control logic for the processing element
        pass

class _Element:
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
    _control_unit: ControlUnit

    def __init__(self, id: int):
        self._id = id
        self._filter = SPAD(words=224, wordSize=16)
        self._ifmap = SPAD(words=12, wordSize=16)
        self._psum = SPAD(words=24, wordSize=16)
        self._control_unit = ControlUnit()

    def __call__(self, input: mp.Queue[Data], output: mp.Queue[Data]):
        # Process input data and produce output data
        pass

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

    _element: _Element
    _proc: mp.Process
    _input: mp.Queue
    _output: mp.Queue

    def __init__(self, id: int):
        self.element = _Element(id)
        self.writer = mp.Queue()
        self.reader = mp.Queue()
        self._proc = mp.Process(target=self.element, args=(self.reader, self.writer))

    def start(self):
        self._proc.start()

    def join(self):
        self._proc.join()

    def put(self, data: Data):
        self.reader.put(data)

    def get(self) -> Data:
        return self.writer.get()
    
    def terminate(self):
        self._proc.terminate()

    def close(self):
        self.reader.close()
        self.writer.close()
