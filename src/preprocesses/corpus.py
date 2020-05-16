from src.ida.code_elements import Function
from src.utils.logger import get_logger


class Corpus:
    """
    Stores function names and ids
    """
    logger = get_logger('Corpus')

    def __init__(self):
        # doc is binary#function
        self.doc2idx = {}
        self.idx2doc = []
        self.idx2ins = []
        self.n_docs = 0

    def onehot_encode(self, fun_name):
        doc = fun_name
        if doc not in self.doc2idx:
            raise RuntimeError(f'No doc with name {doc} in corpus! '
                               f'Consider rebuilding it.')
        return self.doc2idx[doc]


class CorpusBuilder:
    def __init__(self):
        self.corpus = Corpus()

    def reset(self):
        self.corpus.doc2idx = {}
        self.corpus.idx2doc = []
        self.corpus.idx2ins = []
        self.corpus.n_docs = 0

    def scan(self, fun: Function):
        doc, stmts = fun.label, fun.stmts
        self.corpus.idx2doc.append(doc)
        self.corpus.doc2idx[doc] = self.corpus.n_docs
        self.corpus.idx2ins.append(stmts)
        self.corpus.n_docs += 1

    def build(self):
        return self.corpus
