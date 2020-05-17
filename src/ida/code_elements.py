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


class Block(AsJson):
    def __init__(self):
        self.label = ''
        self.ea = 0
        self.src = []  # asm lines
        self.ref_data = {}  # data ref as address -> hex string


class Function(AsJson):
    blocks: List[Block]

    def __init__(self):
        self.label = ''
        self.ea = 0
        self.segm = ''
        self.flags = 0
        self.blocks = []
        self.cfg = {}


class Program(AsJson):
    arch: Arch
    funcs: List[Function]

    def __init__(self, prog=None, cc=None, arch=None, opt=None,
                 obf=None, funcs=None, cg=None):
        self.prog, self.prog_ver = prog if prog else (None, None)
        self.arch = arch
        self.cc, self.cc_ver = cc if cc else (None, None)
        self.opt = opt
        self.obf = obf
        self.funcs = funcs
        self.cg = cg
