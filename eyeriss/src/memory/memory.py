from typing import List, override

import multiprocessing as mp

import numpy as np

import time
from time import sleep

from src.data import Data, Address

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
    _queue_in: mp.Queue[Data]
    _queue_out: mp.Queue[Data]

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
class MemoryBlock:
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
        shape() -> tuple:
            Returns the size of the memory (number of words and word size).
        bits() -> int:
            Returns the total number of bits in memory.
        read_latency() -> float:
            Returns the read latency for the memory in seconds.
        energy() -> float:
            Returns the energy consumption for the memory.
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
    def shape(self):
        return self._words, self._wordSize
    
    @property
    def bits(self):
        return self._words * self._wordSize
    
    @property
    def read_latency(self):
        return self._read_latency

    @property
    def energy(self):
        return self._energy

class Memory:
    """
    Memory class for managing multiple memory blocks.

    Attributes:
        _blocks (List[MemoryBlock]): List of memory blocks.

    Methods:
        read(address: Address) -> int:
            Reads data from the specified address.
        write(address: Address, data: np.ndarray):
            Writes data to the specified address.
        shape() -> tuple:
            Returns the shape of the memory (number of blocks and block size).
        bits() -> int:
            Returns the total number of bits in memory.
        read_latency() -> float:
            Returns the read latency for the memory in seconds.
        energy() -> float:
            Returns the energy consumption for the memory.
    """

    _blocks: List[MemoryBlock]

    def __init__(self, blocks: List[MemoryBlock]):
        self._blocks = blocks

    def read(self, address: Address) -> int:
        # Implement read logic for multiple blocks
        pass

    def write(self, address: Address, data: np.ndarray):
        # Implement write logic for multiple blocks
        pass

    def __getitem__(self, index: int) -> MemoryBlock:
        return self._blocks[index]

    def __len__(self) -> int:
        return len(self._blocks)
    
    @property
    def shape(self):
        return (len(self._blocks), self._blocks[0].shape)
    
    @property
    def bits(self):
        return sum(block.bits for block in self._blocks)
    
    @property
    def read_latency(self):
        return self._blocks[0].read_latency

    @property
    def energy(self):
        return self._blocks[0].energy

class SPAD(Memory):
    """
    Scratchpad memory (SPAD) class.

    Attributes:
        READ_LATENCY (float): Read latency for SPAD.
    """

    READ_LATENCY = 1e-9 # 1 ns
    ENERGY = 1e-12 # 1 pJ

    def __init__(self, words: int, wordSize: int):
        block = MemoryBlock(words, wordSize, self.READ_LATENCY, self.ENERGY)
        super().__init__([block])

class GlobalBuffer(Memory):
    """
    Global buffer memory class.

    Attributes:
        READ_LATENCY (float): Read latency for global buffer.
    """

    READ_LATENCY = 5e-9 # 5 ns
    ENERGY = 5e-12 # 5 pJ

    def __init__(self, blocks: int = 25, blockSize: int = 4096, wordSize: int = 16):
        blocks = [MemoryBlock(blockSize, wordSize, self.READ_LATENCY, self.ENERGY) for _ in range(blocks)]
        super().__init__(blocks)

class DRAM(Memory):
    """
    Dynamic random-access memory (DRAM) class.

    Attributes:
        READ_LATENCY (float): Read latency for DRAM.
    """

    READ_LATENCY = 1e-6 # 1 us
    ENERGY = 500e-12 # 500 pJ

    _pages: int

    def __init__(self, pages: int = 4, blocks: int = 16, blockSize: int = 4096, wordSize: int = 16):
        self._pages = pages
        page_blocks = [[MemoryBlock(blockSize, wordSize, self.READ_LATENCY, self.ENERGY) for _ in range(blocks)] for _ in range(pages)]
        super().__init__(page_blocks)

    @override
    def read(self, address: Address) -> int:
        if address.shape != 3:
            raise ValueError("Address must have 3 dimensions: (page, block, word)")
        page, block, word = address
        pg_blocks = self._blocks[0].shape[0]

        return self._blocks[page * pg_blocks].read(Address((block, word)))

    @override
    def write(self, address: Address, data: np.ndarray):
        if address.shape != 3:
            raise ValueError("Address must have 3 dimensions: (page, block, word)")
        page, block, word = address
        pg_blocks = self._blocks[0].shape[0]

        self._blocks[page * pg_blocks].write(Address((block, word)), data)

    @override
    def __getitem__(self, index: int) -> List[MemoryBlock]:
        return self._blocks[index]

