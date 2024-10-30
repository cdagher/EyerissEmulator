from typing import override

import multiprocessing as mp

import numpy as np

import time
from time import sleep


class _MemoryProc(mp.Process):
    """
    Memory process for handling read and write operations.

    Attributes:
        _queue_in (mp.Queue): Input queue for receiving commands.
        _queue_out (mp.Queue): Output queue for sending data.
        _read_latency (float): Latency for read operations.
        _bits (np.ndarray): Memory storage represented as a numpy array.
        _lock (mp.Lock): Lock for synchronizing access to memory.

    Methods:
        run(): Main loop for processing commands.
        words(): Returns the number of words in memory.
        wordSize(): Returns the size of each word in memory.
    """
    _queue_in: mp.Queue
    _queue_out: mp.Queue

    _read_latency: float

    _bits: np.ndarray
    _lock = mp.Lock()

    def __init__(
            self,
            words: int,
            wordSize: int,
            read_latency: float,
            queue_in: mp.Queue,
            queue_out: mp.Queue
        ):
        super().__init__()
        self._read_latency = read_latency

        self._queue_in = queue_in
        self._queue_out = queue_out

        self._bits = np.zeros((words, wordSize), dtype=np.bool)

    @property
    def words(self):
        return self._bits.shape[0]
    
    @property
    def wordSize(self):
        return self._bits.shape[1]

    @override
    def run(self):
        while True:
            # Wait for a command from the main process
            command = self._queue_in.get()
            if command[0] == "w":
                address, data = command[1], command[2]
                bits = np.array(list(bin(data)[2:].zfill(self.wordSize())), dtype=np.bool)
                with self._lock:
                    self._bits[address] = bits
            elif command[0] == "r":
                address = command[1]
                with self._lock:
                    data = self._bits[address]
                # Simulate read latency
                sleep(self._read_latency)
                self._queue_out.put(data)
class Memory:
    """
    Memory class for managing read and write operations.

    Attributes:
        _words (int): Number of words in memory.
        _wordSize (int): Size of each word in memory.
        _read_latency (float): Latency for read operations.
        _proc (_MemoryProc): Memory process for handling operations.
        _inqueue (mp.Queue): Input queue for sending commands to memory.
        _outqueue (mp.Queue): Output queue for receiving data from memory.

    Methods:
        read(address: int) -> int:
            Reads data from the specified address.
        write(address: int, data: int):
            Writes data to the specified address.
        start(): Starts the memory process.
        join(): Waits for the memory process to finish.
        terminate(): Terminates the memory process.
        close(): Closes the input and output queues.
        size() -> tuple:
            Returns the size of the memory (number of words and word size).
        bits() -> int:
            Returns the total number of bits in memory.
    """

    _words: int
    _wordSize: int

    _read_latency: float # read latency in seconds
    _energy: float       # energy in Joules

    _proc: _MemoryProc
    _inqueue: mp.Queue
    _outqueue: mp.Queue

    def __init__(
            self,
            words: int,
            wordSize: int,
            read_latency: float,
            energy: float
        ):
        self._words = words
        self._wordSize = wordSize

        self._read_latency = read_latency
        self._energy = energy

        self._inqueue = mp.Queue()
        self._outqueue = mp.Queue()
        self._proc = _MemoryProc(words, wordSize, self._read_latency, self._inqueue, self._outqueue)

    def read(self, address: int) -> int:
        self._inqueue.put(("r", address))
        return self._outqueue.get()
    
    def read_int(self, address: int) -> int:
        return int(self.read(address), 2)

    def write(self, address: int, data: int):
        self._inqueue.put(("w", address, data))

    def start(self):
        self._proc.start()

    def join(self):
        self._proc.join()

    def terminate(self):
        self._proc.terminate()

    def close(self):
        self._inqueue.close()
        self._outqueue.close()

    @property
    def size(self):
        return self._words, self._wordSize
    
    @property
    def shape(self):
        return (self._words, self._wordSize)
    
    @property
    def bits(self):
        return self._words * self._wordSize
    
    @property
    def read_latency(self):
        return self._read_latency

    @property
    def energy(self):
        return self._energy

class SPAD(Memory):
    """
    Scratchpad memory (SPAD) class.

    Attributes:
        READ_LATENCY (float): Read latency for SPAD.
    """

    READ_LATENCY = 1e-9 # 1 ns
    ENERGY = 1e-12 # 1 pJ

    def __init__(self, words: int, wordSize: int):
        super().__init__(words, wordSize, self.READ_LATENCY, self.ENERGY)

class GlobalBuffer(Memory):
    """
    Global buffer memory class.

    Attributes:
        READ_LATENCY (float): Read latency for global buffer.
    """

    READ_LATENCY = 5e-9 # 1 ns
    ENERGY = 5e-12 # 1 pJ

    def __init__(self, blocks: int = 25, blockSize: int = 4096):
        super().__init__(blocks, blockSize, self.READ_LATENCY, self.ENERGY)

class DRAM(Memory):
    """
    Dynamic random-access memory (DRAM) class.

    Attributes:
        READ_LATENCY (float): Read latency for DRAM.
    """

    READ_LATENCY = 1e-6 # 1 us
    ENERGY = 500e-12 # 500 pJ

    def __init__(self, words: int, wordSize: int):
        super().__init__(words, wordSize, self.READ_LATENCY, self.ENERGY)
