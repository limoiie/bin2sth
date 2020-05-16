from src.ida.code_elements import Function
from src.preprocesses.corpus import Corpus, CorpusBuilder


class CfgCorpus(Corpus):
    """
    A little different with :class Corpus:
      - one more field for adj matrix of cfg
      - a list of block stmts instead of a flatten func body,
      which is self.idx2ins
    """
    def __init__(self):
        # doc is binary#function
        super().__init__()
        self.cfg_adj = []


class CfgCorpusBuilder(CorpusBuilder):
    def __init__(self):
        super().__init__()
        self.corpus = CfgCorpus()

    def reset(self):
        super().__init__()
        self.corpus.cfg_adj = []

    def scan(self, fun: Function):
        fun.stmts = fun.stmts_list
        super().scan(fun)
        self.corpus.cfg_adj.append(fun.cfg_adj)
