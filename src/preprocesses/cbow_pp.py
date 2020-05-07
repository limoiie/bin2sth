import random

import numpy as np

from src.corpus import Corpus
from src.dataset import UnSupervisedDataset
from src.preprocess import unk_idx_list
from src.vocab import AsmVocab


class CBowDatasetBuilder:

    def __init__(self, vocab: AsmVocab, corpus: Corpus):
        self.vocab = vocab
        self.corpus = corpus
        self.data = []

    def build(self, window, ss=None):
        self.data = []
        # encode in one-hot if not
        if not is_collection_of(self.corpus.idx2ins, [int]):
            self.corpus.idx2ins = self.vocab.onehot_encode(self.corpus.idx2ins)

        # convert each func into a sequence of training data
        for func_id, stmts in enumerate(self.corpus.idx2ins):
            for word, context in make_one_doc(stmts, window):
                self.data.append((func_id, word, np.array(context)))

        # sub-sample tokens if need
        if ss is not None:
            self.__sub_sample(ss)

        return UnSupervisedDataset(self.data)

    def __sub_sample(self, ss_freq):
        """sub sample tokens"""
        ws = self.vocab.sub_sample_ratio(ss_freq)
        if ws is not None:
            def f(w):  # w is of (fun_id, center_word, ctx_words)
                return random.random() < ws[w[1]]
            self.data = list(filter(f, self.data))


def make_one_doc(insts, window):
    n_insts = len(insts)
    for i, inst in enumerate(insts):
        prev_inst = insts[i - 1][:window] if i > 0 else []
        next_inst = insts[i + 1][:window] if i + 1 < n_insts else []
        lw = unk_idx_list(window - len(prev_inst))
        rw = unk_idx_list(window - len(next_inst))

        context = lw + prev_inst + next_inst + rw
        for word in inst:
            yield word, context


def is_collection_of(m, cls, early_break=True):
    """
    Is :param m a collection of instances of :param cls
    :param m: the collection
    :param cls: a list of classes of the expected instance
    :param early_break: just check one non-collection item
    """
    typ = type(m)
    if typ in cls:
        return True
    if typ in (list, tuple, set):
        for i in m:
            return is_collection_of(i, cls, early_break)
    return False
