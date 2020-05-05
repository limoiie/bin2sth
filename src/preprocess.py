import re

from src.ida.code_elements import Program


def split(line):
    def f(l):
        return len(l) > 0
    tokens = re.split(r'[+\-*@$^&\\\[\]:(),;\s]', line.strip())
    return list(filter(f, tokens))


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


def inst_tokenize(stmts):
    """Take the whole instruction as a token"""
    for stmt in stmts:
        yield [stmt]


class DIInstTokenizer(DocIter):
    def __call__(self, prog):

        for label, stmts in prog:
            yield label, inst_tokenize(stmts)


class DIStmts(DocIter):
    """ Collect function bodys """
    def __call__(self, prog):
        for _, stmts in prog:
            yield stmts


class DIMergeProgs(DocIter):
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


class DIOneHotEncode(DocIter):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, prog):
        for label, stmts in prog:
            yield label, self.vocab.onehot_encode(stmts)


def unk_idx_list(l):
    # 0 is the index of self.vocab.unk
    return [0] * l
