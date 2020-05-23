import random

import torch

from src.models.builder import ModelBuilder
from src.preprocesses.corpus import Corpus
from src.preprocesses.dataset.dataset import ReloadableDataset
from src.preprocesses.preprocess import unk_idx_list
from src.preprocesses.vocab import AsmVocab


class PVDMDataset(ReloadableDataset):
    pass


@ModelBuilder.register(PVDMDataset)
class PVDMDatasetBuilder(ModelBuilder):
    vocab: AsmVocab
    corpus: Corpus
    base_corpus: Corpus

    def __init__(self, vocab, corpus, base_corpus, window, ss=None):
        self.vocab = vocab
        self.corpus = corpus
        self.base_corpus = base_corpus
        self.window = window
        self.ss = ss
        self.data = []

    def build(self):
        sync_corpus(self.corpus, self.base_corpus)

        self.data = []
        # convert each func into a sequence of training data
        for func_id, stmts in enumerate(self.corpus.idx2ins):
            per_doc = []
            for word, context in make_one_doc(stmts, self.window):
                per_doc.append((
                    torch.tensor(func_id),
                    torch.tensor(word),
                    torch.tensor(context)
                ))
            if per_doc:
                self.data.append(per_doc)

        maker = SubSampleDataMaker(
            self.data, self.vocab.sub_sample_ratio(self.ss))
        return PVDMDataset(maker)


class SubSampleDataMaker:
    def __init__(self, data, ws):
        self.data = data
        self.ws = ws

    def __call__(self):
        return self.__sub_sample()

    def __sub_sample(self):
        """sub sample tokens"""
        if self.ws is not None:
            def entry_filter(w):  # w is of (fun_id, center_word, ctx_words)
                return random.random() < self.ws[w[1]]

            def p(doc):
                if len(doc) < 10:  # do not sub-sample the small function
                    return doc
                return list(filter(entry_filter, doc))

            return list(filter(None, map(p, self.data)))
        return self.data


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

    sync(train_corpus)


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
