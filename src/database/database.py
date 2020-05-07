from pymongo import MongoClient

from src.corpus import Corpus
from src.database.program_dao import BinArgs, load_progs_jointly
from src.preprocess import DIOneHotEncode, DIInstTokenizer
from src.preprocess import DIProxy, DIPure, DIStmts
from src.preprocess import DITokenizer, DIMergeProgs
from src.preprocesses.cbow_pp import CBowDatasetBuilder
from src.preprocesses.nmt_inspired_pp import DIPadding, NMTInsDataEnd
from src.vocab import AsmVocab


def get_database_client():
    client = MongoClient('localhost', 27017)
    return client


def load_vocab(db, args: BinArgs):
    # TODO: cache into db
    progs = load_progs_jointly(db, args)
    pp = DIProxy([DIPure(), DITokenizer(), DIStmts(), DIMergeProgs()])
    docs = pp.per(progs)
    vocab = AsmVocab()
    vocab.build(docs)
    return vocab


def load_corpus(db, args: BinArgs, vocab):
    progs = load_progs_jointly(db, args)
    pp = DIProxy([
        DIPure(), DITokenizer(), DIOneHotEncode(vocab), DIMergeProgs()
    ])
    docs = pp.per(progs)
    corpus = Corpus()
    corpus.build(docs)
    return corpus


def load_inst_vocab(db, args: BinArgs):
    # TODO: cache into db
    progs = load_progs_jointly(db, args)
    pp = DIProxy([DIPure(), DIInstTokenizer(), DIStmts(), DIMergeProgs()])
    docs = pp.per(progs)
    vocab = AsmVocab()
    vocab.build(docs)
    return vocab


def load_corpus_with_padding(db, args: BinArgs, vocab, maxlen):
    """
    This corpus takes the whole instruction as a word
    """
    progs = load_progs_jointly(db, args)
    pp = DIProxy([
        DIPure(), DIOneHotEncode(vocab), DIPadding(maxlen, 0), DIMergeProgs()
    ])
    docs = pp.per(progs)
    corpus = Corpus()
    corpus.build(docs)
    return corpus


def load_cbow_data_end(db, vocab_args, args: BinArgs, window, ss):
    # TODO: cache into db
    vocab = load_vocab(db, vocab_args)
    corpus = load_corpus(db, args, vocab)
    # data = CBowDataEnd(window, vocab, corpus)
    dataset_maker = CBowDatasetBuilder(vocab, corpus)
    dataset = dataset_maker.build(window, ss)
    return vocab, corpus, dataset


def load_nmt_data_end(db, vocab_args, args1, args2, maxlen):
    vocab = load_inst_vocab(db, vocab_args)
    corpus1 = load_corpus_with_padding(db, args1, vocab, maxlen)
    corpus2 = load_corpus_with_padding(db, args2, vocab, maxlen)
    data = NMTInsDataEnd(vocab, corpus1, corpus2)
    data.build()
    return data
