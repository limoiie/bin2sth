class Bcf:
    def __init__(self):
        self.enable = False
        self.loop = 1

    def __str__(self):
        return f'bcf={self.enable} bcf_loop={self.loop}'


class Fla:
    def __init__(self):
        self.enable = False
        self.split = False
        self.num = 1

    def __str__(self):
        return f'fla={self.enable} split={self.split} split_num={self.num}'


class Sub:
    def __init__(self):
        self.enable = False
        self.loop = 1

    def __str__(self):
        return f'sub={self.enable} sub_loop={self.loop}'


class ObfuscationFlag:
    def __init__(self):
        self.sub = Sub()
        self.bcf = Bcf()
        self.fla = Fla()
        self.flag = ''

    def __str__(self):
        return f'flags: {self.sub} {self.bcf} {self.fla}'


def apply_obfuscator_flags(obf_flags, flags):
    if obf_flags.sub.enable:
        flags += f' -mllvm -sub -mllvm -sub_loop={obf_flags.sub.loop}'
    if obf_flags.bcf.enable:
        flags += f' -mllvm -bcf -mllvm -bcf_loop={obf_flags.bcf.loop}'
    if obf_flags.fla.enable:
        flags += f' -mllvm -fla'
        if obf_flags.fla.split_inst_into_tokens:
            flags += f' -mllvm -split -mllvm -split_num={obf_flags.fla.num}'
    return flags


def parse_obfuscation_flags(obf):
    obfs = obf.split_inst_into_tokens(',')
    flags = ObfuscationFlag()
    flags.flag = obf.replace(',', '-') if obf else 'none'
    for obf in obfs:
        if len(obf) == 0:
            continue
        if obf[0] == 's':
            flags.sub.enable = True
            if len(obf) > 1:
                flags.sub.loop = int(obf[1:])
        if obf[0] == 'b':
            flags.bcf.enable = True
            if len(obf) > 1:
                flags.bcf.loop = int(obf[1:])
        if obf[0] == 'f':
            flags.fla.enable = True
        if obf[0] == 't':
            flags.fla.split = True
            if len(obf) > 1:
                flags.fla.num = int(obf[1:])
    return flags
