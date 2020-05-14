import rx
from pymongo import MongoClient

import src.preprocesses.preprocess as pp
from src.database.program_dao import BinArgs, load_progs_jointly
from src.preprocesses.cbow_pp import CBowDatasetBuilder, sync_corpus
from src.preprocesses.nmt_inspired_pp import NMTInsDataEnd
from src.preprocesses.preprocess import BinBag


def get_database_client():
    client = MongoClient('localhost', 27017)
    return client


def load_vocab(db, args: BinArgs):
    """Load the vocab where the word unit is operands/operators"""
    # TODO: cache into db
    progs = load_progs_jointly(db, args)
    return rx.just(BinBag(progs)).pipe(
        pp.PpMergeBlocks().as_map(),
        pp.PpTokenizer().as_map(),
        pp.PpMergeFuncs().as_map(),
        pp.PpOutStmts().as_map(),
        pp.PpVocab().as_map()
    ).run().vocab.run()


def load_corpus(db, args: BinArgs, vocab):
    progs = load_progs_jointly(db, args)
    return rx.just(BinBag(progs)).pipe(
        pp.PpMergeBlocks().as_map(),
        pp.PpFilterFunc(minlen=5).as_map(),
        pp.PpTokenizer().as_map(),
        pp.PpOneHotEncoder(vocab).as_map(),
        pp.PpMergeFuncs().as_map(),
        pp.PpCorpus().as_map()
    ).run().corpus.run()


def load_inst_vocab(db, args: BinArgs):
    """Load the vocab where the word unit is instruction"""
    # TODO: cache into db
    progs = load_progs_jointly(db, args)
    return rx.just(BinBag(progs)).pipe(
        pp.PpMergeBlocks().as_map(),
        pp.PpMergeFuncs().as_map(),
        pp.PpOutStmts().as_map(),
        pp.PpVocab().as_map()
    ).run().vocab.run()


def load_corpus_with_padding(db, args: BinArgs, vocab, maxlen, minlen=5):
    """
    Load corpus where the whole instruction is taken as the word-unit.
    Besides, each function body will be padding with `0', which is
    the index of `<unk>' tags, to form a unified instruction sequence
    (body) size.
    """
    progs = load_progs_jointly(db, args)
    return rx.just(BinBag(progs)).pipe(
        pp.PpMergeBlocks().as_map(),
        pp.PpFilterFunc(minlen=minlen).as_map(),
        pp.PpOneHotEncoder(vocab).as_map(),
        pp.PpPadding(maxlen, 0).as_map(),
        pp.PpMergeFuncs().as_map(),
        pp.PpCorpus().as_map()
    ).run().corpus.run()


def load_cbow_data(db, vocab_args, train_args, query_args, window, ss):
    """
    Load cbow-related data which is used for training and evaluation.
    """
    vocab = load_vocab(db, vocab_args)
    train_corpus = load_corpus(db, train_args, vocab)
    query_corpus = load_corpus(db, query_args, vocab)

    # sync two corpus for the conviency of evaluation
    train_corpus, query_corpus = sync_corpus(train_corpus, query_corpus)

    dataset_maker = CBowDatasetBuilder(vocab)
    train_ds = dataset_maker.build(train_corpus, window, ss)
    query_ds = dataset_maker.build(query_corpus, window, ss)
    return vocab, train_corpus, query_corpus, train_ds, query_ds


def load_nmt_data_end(db, vocab_args, args1, args2, maxlen):
    vocab = load_inst_vocab(db, vocab_args)
    corpus1 = load_corpus_with_padding(db, args1, vocab, maxlen, 0)
    corpus2 = load_corpus_with_padding(db, args2, vocab, maxlen, 0)
    data = NMTInsDataEnd(vocab, corpus1, corpus2)
    data.build()
    return data
