from abc import ABC
from typing import List, Dict, override

from src.addr import Address

class BaseInstr(ABC):
    _name: str
    _opcode: int

    def __init__(self, name: str, opcode: int):
        self._name = name
        self._opcode = opcode

    @property
    def name(self):
        return self._name
    
    @property
    def opcode(self):
        return self._opcode
    
    def __str__(self):
        return f"{self._name}({self._opcode})"

    def __repr__(self):
        return f"{self._name}({self._opcode})"
    
    def __eq__(self, other):
        if isinstance(other, BaseInstr):
            return self._opcode == other._opcode
        return False
    
    def __hash__(self):
        return hash(self._opcode)

class TerminateInstr(BaseInstr):
    def __init__(self):
        super().__init__("TERMINATE", 0)

class BaseRWInstr(BaseInstr, ABC):
    _address: Address

    def __init__(self, pre: str, opcode: int, address: int):
        super(BaseRWInstr, self).__init__(f"{pre}_READ", opcode)
        self._address = Address(address)

    @property
    def address(self):
        return self._address

class BaseReadInstr(BaseRWInstr, ABC):
    def __init__(self, pre: str, opcode: int, address: int):
        super(BaseReadInstr, self).__init__(f"{pre}_READ", opcode, address)

    @override
    def __str__(self):
        return f"{self._name}({self._opcode}, address={self._address})"
    
    @override
    def __repr__(self):
        return f"{self._name}({self._opcode}, address={self._address})"
    
    @override
    def __eq__(self, other):
        if isinstance(other, BaseReadInstr):
            return super().__eq__(other) and self._address == other._address
        return False
    
    @override
    def __hash__(self):
        return hash((self._opcode, self._address))

class BaseWriteInstr(BaseRWInstr, ABC):
    _data: int

    def __init__(self, pre: str, opcode: int, address: int, data: int):
        super(BaseWriteInstr, self).__init__(f"{pre}_WRITE", opcode, address)
        self._data = data

    @property
    def address(self):
        return self._address
    
    @property
    def data(self):
        return self._data
    
    @override
    def __str__(self):
        return f"{self._name}({self._opcode}, address={self._address}, data={self._data})"
    
    @override
    def __repr__(self):
        return f"{self._name}({self._opcode}, address={self._address}, data={self._data})"
    
    @override
    def __eq__(self, other):
        if isinstance(other, BaseWriteInstr):
            return super().__eq__(other) and self._address == other._address and self._data == other._data
        return False
    
    @override
    def __hash__(self):
        return hash((self._opcode, self._address, self._data))

class ComputeInstr(BaseInstr):
    def __init__(self):
        super().__init__("COMPUTE", 1)

class PEWriteFilterInstr(BaseWriteInstr):
    def __init__(self, address: Address, data: int):
        super().__init__("PE_FILTER", 4, address, data)

class PEWriteIfmapInstr(BaseWriteInstr):
    def __init__(self, address: Address, data: int):
        super().__init__("PE_IFMAP", 5, address, data)

class PEReadPsumInstr(BaseReadInstr):
    def __init__(self, address: Address):
        super().__init__("PE_PSUM", 6, address)

class PEWritePsumInstr(BaseWriteInstr):
    def __init__(self, address: Address, data: int):
        super().__init__("PE_PSUM", 7, address, data)

class PEAddPsumInstr(BaseWriteInstr):
    def __init__(self, address: Address, data: int):
        super().__init__("PE_PSUM", 8, address, data)

class GLBReadFilterInstr(BaseReadInstr):
    def __init__(self, address: Address):
        super().__init__("GLB_FILTER", 9, address)
        
class GLBWriteFilterInstr(BaseWriteInstr):
    def __init__(self, address: Address, data: int):
        super().__init__("GLB_FILTER", 10, address, data)

class GLBReadIFMAPInstr(BaseReadInstr):
    def __init__(self, address: Address):
        super().__init__("GLB_IFMAP", 11, address)

class GLBWriteIFMAPInstr(BaseWriteInstr):
    def __init__(self, address: Address, data: int):
        super().__init__("GLB_IFMAP", 12, address, data)

class GLBReadPSUMInstr(BaseReadInstr):
    def __init__(self, address: Address):
        super().__init__("GLB_PSUM", 13, address)

class GLBWritePSUMInstr(BaseWriteInstr):
    def __init__(self, address: Address, data: int):
        super().__init__("GLB_PSUM", 14, address, data)

class GLBReadOfMapInstr(BaseReadInstr):
    def __init__(self, address: Address):
        super().__init__("GLB_OFMAP", 15, address)


class InstructionSet:
    _instructions: List[BaseInstr]

    def __init__(self):
        self._instructions = [
            TerminateInstr(),
            ComputeInstr(),
            PEWriteFilterInstr(None, 0),   # Placeholder values
            PEWriteIfmapInstr(None, 0),    # Placeholder values
            PEReadPsumInstr(None),         # Placeholder values
            PEWritePsumInstr(None, 0),     # Placeholder values
            PEAddPsumInstr(None, 0),       # Placeholder values
            GLBReadFilterInstr(None),      # Placeholder values
            GLBWriteFilterInstr(None, 0),  # Placeholder values
            GLBReadIFMAPInstr(None),       # Placeholder values
            GLBWriteIFMAPInstr(None, 0),   # Placeholder values
            GLBReadPSUMInstr(None),        # Placeholder values
            GLBWritePSUMInstr(None, 0),    # Placeholder values
            GLBReadOfMapInstr(None),       # Placeholder values
        ]

    def get_instruction(self, opcode: int) -> BaseInstr:
        return self.instructions.get(opcode, None)
    
    def list_instructions(self) -> List[BaseInstr]:
        return self._instructions

    def __str__(self):
        return "\n".join(str(instr) for instr in self.instructions.values())
    
    def __repr__(self):
        return f"InstructionSet({self.instructions})"
    
    def __contains__(self, instr: BaseInstr) -> bool:
        return instr.opcode in self.instructions
    
    def __getitem__(self, opcode: int) -> BaseInstr:
        return self.instructions[opcode]
    
    def __len__(self) -> int:
        return len(self.instructions)
    
    @property
    def instructions(self) -> Dict[int, BaseInstr]:
        return {instr.opcode: instr for instr in self._instructions}  # return a dictionary for easier use
