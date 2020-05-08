import random
from itertools import filterfalse

import numpy as np

from src.corpus import Corpus
from src.dataset import UnSupervisedDataset
from src.preprocess import unk_idx_list
from src.vocab import AsmVocab


class CBowDatasetBuilder:

    def __init__(self, vocab: AsmVocab):
        self.vocab = vocab
        self.data = []

    def build(self, corpus, window, ss=None):
        self.data = []
        # encode in one-hot if not
        if not _is_collection_of(corpus.idx2ins, [int]):
            corpus.idx2ins = self.vocab.onehot_encode(corpus.idx2ins)

        # convert each func into a sequence of training data
        for func_id, stmts in enumerate(corpus.idx2ins):
            per_doc = []
            for word, context in _make_one_doc(stmts, window):
                per_doc.append((func_id, word, context))
            if per_doc:
                self.data.append(per_doc)

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

            def p(doc):
                if len(doc) < 10:
                    return doc
                return list(filter(f, doc))

            self.data = list(filter(None, map(p, self.data)))


def sync_corpus(train_corpus: Corpus, query_corpus: Corpus):
    """
    Sync two corpus into a consistant state, where they have the same
    funcs and the same index identifies the same func
    """
    idx2doc = list(set(train_corpus.idx2doc) & set(query_corpus.idx2doc))
    n_docs = len(idx2doc)

    def sync(corpus):
        idx2ins = []
        doc2idx = {}
        for i, doc in enumerate(idx2doc):
            idx2ins.append(corpus.idx2ins[i])
            doc2idx[doc] = i
        corpus.idx2doc = idx2doc
        corpus.idx2ins = idx2ins
        corpus.doc2idx = doc2idx
        corpus.n_docs = n_docs
        return corpus

    return sync(train_corpus), sync(query_corpus)


def _make_one_doc(insts, window):
    n_insts = len(insts)
    for i, inst in enumerate(insts):
        prev_inst = insts[i - 1][:window] if i > 0 else []
        next_inst = insts[i + 1][:window] if i + 1 < n_insts else []
        lw = unk_idx_list(window - len(prev_inst))
        rw = unk_idx_list(window - len(next_inst))

        context = lw + prev_inst + next_inst + rw
        for word in inst:
            yield word, context


def _is_collection_of(m, cls, early_break=True):
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
            return _is_collection_of(i, cls, early_break)
    return False
