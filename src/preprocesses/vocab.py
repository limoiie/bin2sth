import numpy as np

from src.utils.logger import get_logger


class AsmVocab:
    logger = get_logger('AsmVocab')
    unk = '<unk>'

    def __init__(self, min_freq=0, _max_vocab=10000):
        self.min_freq = min_freq
        self.total_tkn = 0
        self.size = 0
        self.counter = {}
        self.idx2tkn = []
        self.idx2frq = []
        self.tkn2idx = {}

    def onehot_encode(self, insts):
        """ Convert insts in tokens to insts in indexes """
        if type(insts) is str:
            return self.tkn2idx.get(insts)

        idx_insts = []
        for inst in insts:
            encoding = self.onehot_encode(inst)
            if encoding:
                idx_insts.append(encoding)
        return idx_insts

    def word_freq_ratio(self):
        word_ssr = np.array(self.idx2frq)
        return word_ssr / word_ssr.sum()

    def sub_sample_ratio(self, ss):
        word_ssr = self.word_freq_ratio()
        word_ssr[0] = 1
        word_ssr = np.sqrt(ss / word_ssr) + ss / word_ssr
        word_ssr = np.clip(word_ssr, 0, 1)
        word_ssr[0] = 0
        return word_ssr


class AsmVocabBuilder:
    logger = get_logger('AsmVocab')
    unk = '<unk>'

    def __init__(self, min_freq=0, _max_vocab=10000):
        self.vocab: AsmVocab = AsmVocab(min_freq, _max_vocab)
        self.counter = {}

    def reset(self):
        self.vocab.total_tkn = 0
        self.vocab.size = 0
        self.vocab.tkn2idx = {}
        self.vocab.idx2tkn = []
        self.vocab.idx2frq = []

    def scan(self, doc):
        self.__count_tokens(doc)

    def build(self):
        """
        Build the vocab from <docs>. <docs> is a list of <doc>, each
        <doc> is a list of insts, each inst is a list of opds/oprs
        """
        self.__filter_by_freq()
        self.vocab.size = len(self.counter)
        self.vocab.total_tkn = sum(self.counter.values())
        self.__map_idx2other()

        self.logger.debug(f'unique tokens: {len(self.vocab.idx2tkn)}')
        self.logger.debug(f'total  tokens: {self.vocab.total_tkn}')
        return self.vocab

    def __count_tokens(self, item):
        if type(item) in [str, bytes]:
            c = self.counter.get(item, 0)
            self.counter[item] = c + 1
        else:
            for sub in item:
                self.__count_tokens(sub)

    def __filter_by_freq(self):
        filter_out_keys = []
        for token, freq in self.counter.items():
            if freq < self.vocab.min_freq:
                filter_out_keys.append(token)
        for token in filter_out_keys:
            del self.counter[token]

    def __map_idx2other(self):
        self.vocab.size += 1
        self.vocab.idx2tkn = [''] * self.vocab.size
        self.vocab.idx2frq = [0] * self.vocab.size

        p = self.unk
        self.vocab.idx2tkn[0] = p
        self.vocab.idx2frq[0] = 0
        self.vocab.tkn2idx[p] = 0

        name_freq_pairs = list(self.counter.items())

        for idx, (name, freq) in enumerate(name_freq_pairs):
            idx += 1
            self.vocab.idx2tkn[idx] = name
            self.vocab.idx2frq[idx] = freq
            self.vocab.tkn2idx[name] = idx
