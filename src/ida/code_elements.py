import copy
import json


class Serializable(object):
    """ `json` is the object in dict type  """

    def __getitem__(self, item):
        return getattr(self, item)

    def serialize(self):
        return copy.deepcopy(self.__dict__)

    def deserialize(self, data):
        self.__dict__ = copy.deepcopy(data)
        return self

    @staticmethod
    def to_json(obj):
        if obj is None:
            return None
        return obj.serialize()

    @staticmethod
    def to_json_list(objs):
        if objs is None:
            return []
        return [Serializable.to_json(obj) for obj in objs]

    @staticmethod
    def from_json(Obj, data):
        obj = Obj()
        obj.deserialize(data)
        return obj

    @staticmethod
    def from_json_list(Obj, data):
        return [Serializable.from_json(Obj, e) for e in data]

    @staticmethod
    def dump(file, obj):
        with open(file, 'w') as f:
            json.dump(obj.serialize(), f)

    @staticmethod
    def load(file, Obj):
        with open(file, 'r') as f:
            js = json.load(f)
            return Serializable.from_json(Obj, js)


class Arch(Serializable):
    def __init__(self, typ=None, bits=None, endian=None):
        self.type = typ
        self.bits = bits
        self.endian = endian


class Block(Serializable):
    def __init__(self):
        self.label = ''
        self.ea = 0
        self.src = []  # asm lines
        self.ref_data = {}  # data ref as address -> hex string


class Function(Serializable):
    def __init__(self):
        self.label = ''
        self.ea = 0
        self.segm = ''
        self.flags = 0
        self.blocks = []
        self.cfg = {}

    def serialize(self):
        dic = super(Function, self).serialize()
        dic['blocks'] = self.to_json_list(self.blocks)
        return dic

    def deserialize(self, data):
        super(Function, self).deserialize(data)
        self.blocks = self.from_json_list(Block, data['blocks'])


class Program(Serializable):
    def __init__(self, prog=None, cc=None, arch=None, opt=None,
                 obf=None, funcs=None, cg=None):
        self.prog, self.prog_ver = prog
        self.arch = arch
        self.cc, self.cc_ver = cc
        self.opt = opt
        self.obf = obf
        self.funcs = funcs
        self.cg = cg

    def serialize(self):
        dic = super(Program, self).serialize()
        dic['arch'] = self.to_json(self.arch)
        dic['funcs'] = self.to_json_list(self.funcs)
        return dic

    def deserialize(self, data):
        super(Program, self).deserialize(data)
        self.arch = self.from_json(Arch, data['arch'])
        self.funcs = self.from_json_list(Function, data['funcs'])
