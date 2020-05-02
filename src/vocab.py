from src.utils.logger import get_logger

import numpy as np


class AsmVocab:
    logger = get_logger('AsmVocab')

    def __init__(self, min_freq=0, _max_vocab=10000, unk='</unk>'):
        self.min_freq = min_freq
        self.unk = unk
        self.total_tkn = 0
        self.size = 0
        self.counter = {}
        self.idx2tkn = []
        self.idx2frq = []
        self.tkn2idx = {}

    def build(self, docs):
        """ 
        Build the vocab from <docs>. <docs> is a list of <doc>, each 
        <doc> is a list of insts, each inst is a list of opds/oprs
        """
        self.__prepare()
        self.__count_tokens(docs)    
        self.__filter_by_freq()
        self.size = len(self.counter)
        self.total_tkn = sum(self.counter.values())
        self.__map_idx2other()

        self.logger.debug(f'unique tokens: {len(self.idx2tkn)}')
        self.logger.debug(f'total  tokens: {self.total_tkn}')

    def __prepare(self):
        self.total_tkn = 0
        self.size = 0
        self.counter.clear()
        self.idx2tkn.clear()
        self.idx2frq.clear()
        self.tkn2idx.clear()

    def __count_tokens(self, docs):
        for doc in docs:
            for ins in doc:
                for token in ins:
                    c = self.counter.get(token, 0)
                    self.counter[token] = c + 1

    def __filter_by_freq(self):
        filter_out_keys = []
        for token, freq in self.counter.items():
            if freq < self.min_freq:
                filter_out_keys.append(token)
        for token in filter_out_keys:
            del self.counter[token]

    def __map_idx2other(self):
        self.size += 1
        self.idx2tkn = [''] * self.size
        self.idx2frq = [0] * self.size

        p = self.unk
        self.idx2tkn[0] = p
        self.idx2frq[0] = 0
        self.tkn2idx[p] = 0

        name_freq_pairs = list(self.counter.items())
        name_freq_pairs.sort(key=lambda e: e[1], reverse=True)

        for idx, (name, freq) in enumerate(name_freq_pairs):
            idx += 1
            self.idx2tkn[idx] = name
            self.idx2frq[idx] = freq
            self.tkn2idx[name] = idx

    def onehot_encode(self, insts):
        """ Convert insts in tokens to insts in indexes """
        idx_insts = []
        for inst in insts:
            idx_inst = []
            for token in inst:
                if token in self.tkn2idx:
                    idx_inst.append(self.tkn2idx[token])
            if idx_inst:
                idx_insts.append(idx_inst)
        return idx_insts


def compute_word_freq_ratio(vocab):
    word_ssr = np.array(vocab.idx2frq)
    return word_ssr / word_ssr.sum()    


def compute_sub_sample_ratio(wf, ss):
    word_ssr = wf
    word_ssr[0] = 1
    word_ssr = np.sqrt(ss / word_ssr) + ss / word_ssr
    word_ssr = np.clip(word_ssr, 0, 1)
    word_ssr[0] = 0
    return word_ssr
