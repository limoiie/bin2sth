import pysnooper


cc_cxx = {
    'gcc': 'g++',
    'clang': 'clang++'
}


class CompilerCfg:
    # @pysnooper.snoop()
    def __init__(self, cc, cc_ver):
        self.cc, self.cc_ver = cc, cc_ver
        for c in cc_cxx:
            if c in self.cc:
                self.cxx = self.cc.replace(c, cc_cxx[c])
                self.cxx_ver = cc_ver
                return
        else:
            raise ValueError(f'Unknown compiler - {cc}! Support compilers:'
                             f'{cc_cxx.keys()}')

    def fill_flags(self, maker):
        pass


def make_compiler_cfg(cc: str):
    i = cc.rfind('-')
    cc, cc_ver = cc[:i], cc[i+1:]
    if i == -1:
        raise ValueError(f'You must specify the compiler version explicitly!'
                         f'Such as gcc-7')

    cfg = CompilerCfg(cc, cc_ver)
    return cfg
