import logging
import re

from src.ida.code_elements import Program
from src.vocab import AsmVocab
from src.corpus import Corpus


def split(line):
    def f(l):
        return len(l) > 0
    return filter(f, re.split(r'[+\-*@$^&\\\[\]:(),;\s]', line.strip()))


def label_func(prog, func):
    return prog['prog'] + prog['prog_ver'] + func['label']


class DocIter:
    def __call__(self, prog):
        raise NotImplementedError

    def per(self, progs):
        """  map with __call__ and reduce with unite """
        for prog in progs:
            yield self(prog)


class DIProxy(DocIter):
    def __init__(self, iters):
        self.iters = iters

    def __call__(self, prog):
        for doc_iter in self.iters:
            prog = doc_iter(prog)
        return prog

    def per(self, progs):
        for doc_iter in self.iters:
            progs = doc_iter.per(progs)
        return progs


class DIPure(DocIter):
    """  
    Convert Program dict into a sequence of (func_label, func_stmts) 
    """

    def __call__(self, prog):
        if isinstance(prog, Program):
            prog = prog.__dict__

        try:
            for func in prog['funcs']:
                label, stmts = label_func(prog, func), []
                for block in func['blocks']:
                    stmts += block['src']
                yield label, stmts
        except KeyError:
            raise Exception('prog should be an inst of <Program> or <Dict>!\n\
                Check if the fields name of <Program> have been changed.')


def tokenize(stmts):
    for stmt in stmts:
        yield split(stmt)


class DITokenizer(DocIter):
    """ 
    Tokenize statements in the <prog>. <prog> should be a list of
    <func>s; each <func> should consist of a label and the stmts 
    in lines
    """
    def __call__(self, prog):
        
        for label, stmts in prog:
            yield label, tokenize(stmts)


class DIStmts(DocIter):
    """ Collect function bodys """
    def __call__(self, prog):
        for _, stmts in prog:
            yield stmts


class DIUnite(DocIter):
    """ 
    Unite the list of progs into a list of docs, where each prog 
    contains a list of docs and each doc corresponding to a function
    """
    def __call__(self, prog):
        for doc in prog:
            yield doc

    def per(self, progs):
        for prog in progs:
            for doc in prog:
                yield doc


class DICorpus(DocIter):
    """ Collect function labels """
    def __call__(self, prog):
        for label, _ in prog:
            yield label


def unk_idx_list(l):
    # 0 is the index of self.vocab.unk
    return [0] * l


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
            stmts = self.vocab.onehot_encode(stmts)
            for word, context in self.__build_one_doc(stmts):
                data.append((func_id, word, context))
        self.data = data

        self.logger.debug('building training data done')
