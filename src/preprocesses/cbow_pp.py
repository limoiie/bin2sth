import logging

from src.corpus import Corpus
from src.preprocess import unk_idx_list
from src.vocab import AsmVocab


class CBowDataEnd:
    """
    Convert documents into the sequence of (center_word, words_window)
    for the support of training CBow model. The words therein have
    been substituted with the one-hot encoding by using the given vocab
    """
    logger = logging.getLogger('CBowDataEnd')

    def __init__(self, window, vocab: AsmVocab, corpus: Corpus):
        self.window = window
        self.vocab = vocab
        self.corpus = corpus
        self.data = []
        assert is_collection_of(corpus.idx2ins, [int])

    def __unk_list(self, l):
        return [self.vocab.unk] * l

    def __build_one_doc(self, insts):
        n_insts = len(insts)
        for i, inst in enumerate(insts):
            prev_inst = insts[i - 1][:self.window] if i > 0 else []
            next_inst = insts[i + 1][:self.window] if i + 1 < n_insts else []
            lw = unk_idx_list(self.window - len(prev_inst))
            rw = unk_idx_list(self.window - len(next_inst))

            context = lw + prev_inst + next_inst + rw
            for word in inst:
                yield word, context

    def build(self):
        """ Process docs into a sequence of training data entries. """
        self.logger.debug('building training data...')

        data = []
        for func_id, stmts in enumerate(self.corpus.idx2ins):
            for word, context in self.__build_one_doc(stmts):
                data.append((func_id, word, context))
        self.data = data

        self.logger.debug('building training data done')


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
    if typ is list:
        for i in m:
            return is_collection_of(i, cls, early_break)
    return False
