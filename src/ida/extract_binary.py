import sys
import os
import re

import fire
import idaapi
import idautils
import idc

sys.path.append('.')

from code_elements import Arch, Program, Function, Block

import ida_utils as iu


# fill in the args from command line
sys.argv = idc.ARGV

print('Asm2Vec script for idapro is now running...')
print('Waiting for idapro...')
idaapi.autoWait()
print('start persisting...')


def make_callgraph():
    cg = {}
    for _, callee_ea in iu.all_functions():
        # foreach address who refs to callee_ea
        for ea in idautils.CodeRefsTo(callee_ea, 0):
            caller = idaapi.get_func(ea)
            if caller:
                caller_ea = caller.start_ea
                cg[caller_ea] = cg.get(caller_ea, [])
                cg[caller_ea].append(callee_ea)
    return cg


def make_arch():
    self = Arch()
    info = idaapi.get_inf_structure()
    self.type = info.procName.lower()
    if self.type.startswith('mips'):
        self.type = 'mips'

    self.bits = 'b64' if info.is_64bit() else 'b32'
    self.endian = 'be' if iu.is_be(info) else 'le'
    return self


def make_inst(head):
    # addr = to_hex(head)
    mnem = idc.GetMnem(head)
    if mnem == '':
        return []
    mnem = idc.GetDisasm(head).split()[0]

    opds = []
    # get opds one by one
    for i in range(5):
        opd = idc.GetOpnd(head, i)
        if opd != '' and not opd.startswith('unk_'):
            nea = idc.get_name_ea_simple(opd)
            if idc.get_segm_name(nea) == '.rodata':
                # string will be hashed into a short term
                s = idc.get_strlit_contents(nea)
                if s:
                    opd = 'str%x' % (hash(s) % 0x1000000)
                    opds.append(opd)
            opds.append(opd)

    inst = [mnem] + opds
    return ', '.join(inst)


def make_block(block, arch, cfg):
    """
  Create block. Instructions are extracted, the refereed data is
  collected and the CFG is filled
  """
    self = Block()
    self.label = 'loc_' + iu.to_hex_(block.start_ea)

    self.ea = block.start_ea
    if arch.type == 'arm':
        self.ea += idc.GetReg(self.ea, 'T')
    # self.end_ea = block.end_ea

    # parser asm lines
    for i, head in enumerate(idautils.Heads(block.start_ea, block.end_ea)):
        # parse instruction from head
        self.src.append(make_inst(head))

        # get data ref from here
        for ref in idautils.DataRefsFrom(head):
            self.ref_data[i] = iu.to_hex_inv(idc.Qword(ref))

    # fill succs into cfg
    cfg[self.ea] = [succ.start_ea for succ in block.succs()]

    return self


def make_func(func_ea, segm, arch):
    self = Function()
    func = idaapi.get_func(func_ea)
    self.label = idc.GetFunctionName(func_ea)
    self.flags = func.flags
    self.ea = func.start_ea
    self.segm = segm
    self.cfg = {}

    # collect blocks
    flow_chart = idaapi.FlowChart(func)
    self.blocks = [
        make_block(block, arch, self.cfg)
        for block in flow_chart
    ]
    return self


def extract(prog_name, prog_ver, cc, cc_ver, opt, obf):
    prog = (prog_name, prog_ver)
    cc = (cc, cc_ver)
    arch = make_arch()
    funcs = [
        make_func(func_ea, segm, arch)
        for segm, func_ea in iu.all_functions()
    ]
    cg = make_callgraph()
    obj = Program(prog, cc, arch, opt, obf, funcs, cg)

    file = '%s.tmp.json' % iu.filepath()
    obj.dump(file, obj)


def auto_extract(filepath):
    _, filename = os.path.split(filepath)
    res = re.match(r'(\w*?)@(.*?)-(O[0123])-(\w*)-(.*?)-(\w*)', filename)
    prog_name, prog_ver, cc, cc_ver, opt, obf = res.groups()
    # print(f'{prog_name}, {prog_ver}, {cc}, {cc_ver}, {opt}, {obf}')
    extract(prog_name, prog_ver, cc, cc_ver, opt, obf)


fire.Fire(auto_extract)

idc.Exit(0)
