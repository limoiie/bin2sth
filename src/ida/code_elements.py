from typing import List

from src.utils.auto_json import auto_json

try:
    from as_json import AsJson
except ImportError:
    from src.ida.as_json import AsJson


@auto_json
class Arch(AsJson):
    def __init__(self, typ=None, bits=None, endian=None):
        self.type = typ
        self.bits = bits
        self.endian = endian

    def __str__(self):
        return str(self.__dict__)


class Block(AsJson):
    def __init__(self):
        self.label = ''
        self.ea = 0
        self.src = []  # asm lines
        self.ref_data = {}  # data ref as address -> hex string

    def __str__(self):
        return str(self.__dict__)


class Function(AsJson):
    blocks: List[Block]

    def __init__(self):
        self.label = ''
        self.ea = 0
        self.segm = ''
        self.flags = 0
        self.blocks = []
        self.cfg = {}

    def __str__(self):
        return str(self.__dict__)


class Program(AsJson):
    arch: Arch
    funcs: List[Function]

    def __init__(self, prog=None, prog_ver=None, cc=None, cc_ver=None,
                 arch=None, opt=None, obf=None, funcs=None, cg=None):
        self.prog, self.prog_ver = prog, prog_ver
        self.arch = arch
        self.cc, self.cc_ver = cc, cc_ver
        self.opt = opt
        self.obf = obf
        self.funcs = funcs
        self.cg = cg

    def __str__(self):
        return str(self.__dict__)
