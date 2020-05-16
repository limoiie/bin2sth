import functools
import re
from typing import Iterable, Optional, List

import rx
from rx import operators as ops, Observable

import numpy as np

import src.utils.rx.operators as ops_
from src.preprocesses.corpus import CorpusBuilder
from src.ida.code_elements import Program, Function, Block
from src.utils.logger import get_logger
from src.preprocesses.vocab import AsmVocabBuilder
from src.utils.rx.internal.basic import second

logger = get_logger('preprocess')


def split_inst_into_tokens(line):
    def f(l):
        return len(l) > 0
    tokens = re.split(r'[+\-*@$^&\\\[\]:(),;\s]', line.strip())
    return list(filter(f, tokens))


def label_func(prog, func):
    return prog.prog + prog.prog_ver + func.label


class BinBag:
    progs: Iterable[Program]

    def __init__(self, progs):
        self.progs = progs


def to_obs(stream):
    if isinstance(stream, Observable):
        return stream
    assert isinstance(stream, Iterable)
    return rx.from_iterable(stream)


class Pp:
    def __call__(self, bag: BinBag):
        return self.p(bag)

    def p(self, bag: BinBag) -> BinBag:
        bag.progs = to_obs(bag.progs).pipe(
            ops.map(lambda prog: self.p_prg(prog)))
        return bag

    def p_prg(self, prog: Program) -> Program:
        prog.funcs = self.p_funs(to_obs(prog.funcs), prog)
        return prog

    def p_fun(self, fun: Function, prog: Program) -> Optional[Function]:
        fun.blocks = self.p_blks(fun.blocks, fun, prog)
        # fun.blocks = self.p_blks(to_obs(fun.blocks), fun, prog)
        return fun

    def p_blk(self, blk: Block, fun: Function, prog: Program) \
            -> Optional[Block]:
        return blk

    def p_funs(self, funs: Observable, prog: Program) -> Observable:
        return funs.pipe(
            ops.map(lambda fun: self.p_fun(fun, prog)),
            ops.filter(lambda fun: fun is not None)
        )

    # def p_blks(self, blks: Observable, fun: Function, prog: Program) \
    #         -> Observable:
    #     return blks.pipe(
    #         ops.map(lambda blk: self.p_blk(blk, fun, prog)),
    #         ops.filter(lambda blk: blk is not None)
    #     )

    def p_blks(self, blks: List[Block], fun: Function, prog: Program) \
            -> List[Block]:
        p_blk = functools.partial(self.p_blk, fun=fun, prog=prog)
        return list(filter(None, map(p_blk, blks)))

    def as_map(self):
        return ops.map(self)


class PpFullLabel(Pp):
    def p_fun(self, fun: Function, prog: Program) -> Optional[Function]:
        fun.label = label_func(prog, fun)
        return fun


def tokenize(stmts):
    return list(map(split_inst_into_tokens, stmts))


class PpTokenizer(Pp):
    def p_fun(self, fun: Function, prog: Program)\
            -> Optional[Function]:
        if hasattr(fun, 'stmts'):
            fun.stmts = tokenize(fun.stmts)
            return fun
        return super().p_fun(fun, prog)

    def p_blk(self, blk: Block, fun: Function, prog: Program) \
            -> Optional[Block]:
        blk.src = tokenize(blk.src)
        return blk


# depend on PpMergeBlocks
class PpFilterFunc(Pp):
    def __init__(self, minlen):
        self.minlen = minlen

    def p_funs(self, funs: Observable, prog: Program) -> Observable:
        def count(fun):
            return sum(map(lambda b: len(b.src), fun.blocks)), fun
        return funs.pipe(
            ops.map(count),
            ops.filter(lambda cf: cf[0] >= self.minlen),
            ops.map(second)
        )


class PpOneHotEncoder(Pp):
    def __init__(self, vocab):
        self.vocab = vocab

    def p_fun(self, fun: Function, prog: Program) \
            -> Optional[Function]:
        if hasattr(fun, 'stmts'):
            fun.stmts = self.vocab.onehot_encode(fun.stmts)
            return fun
        return super().p_fun(fun, prog)

    def p_blk(self, blk: Block, fun: Function, prog: Program) \
            -> Optional[Block]:
        blk.src = self.vocab.onehot_encode(blk.src)
        return blk


# insert field <stmts> into Function
class PpMergeBlocks(Pp):
    def p_blks(self, blks: List[Block], fun: Function, prog: Program) \
            -> List[Block]:
        fun.stmts = sum(map(lambda blk: blk.src, blks), [])
        return blks


# add <funcs> into <BinBag>
class PpMergeFuncs(Pp):
    def p(self, bag: BinBag) -> BinBag:
        bag.funcs = to_obs(bag.progs).pipe(
            ops.flat_map(
                lambda prog: to_obs(prog.funcs)))
        return bag


# depends on PpMergeBlocks ==> PpMergeProgs
class PpOutStmts(Pp):
    def p(self, bag: BinBag) -> BinBag:
        bag.o_stmts = bag.funcs.pipe(
            ops.map(lambda f: f.stmts))
        return bag


# insert <cfg_adj> into :class Function
# insert <stmts_list> into :class Function
class PpCfgAdj(Pp):
    """
    Extract a list of block stmts and construct the adj matrix from cfg
    """
    def p_fun(self, fun: Function, prog: Program) -> Optional[Function]:
        n = len(fun.blocks)
        fun.cfg_adj = np.zeros((n, n), dtype=float)
        blk2idx = {blk.label: i for i, blk in enumerate(fun.blocks)}
        for from_, to_s_ in fun.cfg.items():
            for to_ in to_s_:
                fun.cfg_adj[blk2idx[from_]][blk2idx[to_]] = 1.0
        fun.stmts_list = [blk.src for blk in fun.blocks]
        return fun


# depends on PpMergeBlocks ==> PpMergeProgs
class PpPadding(Pp):
    def __init__(self, maxlen, ph):
        self.maxlen = maxlen
        self.ph = ph

    def p_fun(self, fun: Function, prog: Program) \
            -> Optional[Function]:
        fun.stmts = self.pad_stmts(fun.stmts)
        return fun

    def pad_stmts(self, stmts):
        if len(stmts) >= self.maxlen:
            return stmts[:self.maxlen]
        # todo: consider return the real lens too so that we
        #  can pack them before feeding them into lstm, or
        #  something like that
        return [self.ph] * (self.maxlen - len(stmts)) + stmts


# depends on PpOutStmts ==> PpMergeBlocks ==> PpMergeProgs
class PpVocab(Pp):
    def __init__(self, min_freq=0, max_vocab=10000):
        self.builder = AsmVocabBuilder(min_freq, max_vocab)
        self.builder.reset()

    def p(self, bag: BinBag) -> BinBag:
        bag.vocab = bag.o_stmts.pipe(
            ops_.tap(self.builder.scan),
            ops_.end_act(self.builder.build)
        )
        return bag


# depends on PpMergeFuncs
class PpCorpus(Pp):
    def __init__(self, builder=CorpusBuilder()):
        self.builder = builder
        self.builder.reset()

    def p(self, bag: BinBag) -> BinBag:
        bag.corpus = bag.funcs.pipe(
            ops_.tap(self.builder.scan),
            ops_.end_act(self.builder.build)
        )
        return bag


def unk_idx_list(l):
    # 0 is the index of self.vocab.unk
    return [0] * l
