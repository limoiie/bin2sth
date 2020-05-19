import rx
from bson import ObjectId
from gridfs import GridFS
from pymongo import MongoClient

import src.preprocesses.preprocess as pp
from src.database.beans.check_point import CheckPoint
from src.database.dao import Dao, to_filter
from src.ida.code_elements import Program
from src.training.args.train_args import BinArgs, RuntimeArgs, TrainArgs
from src.dataset.pvdm_dataset import PVDMDatasetBuilder, sync_corpus
from src.dataset.genn_ufe_dataset import GENNDatasetBuilder
from src.dataset.word2vec_datset import Word2VecDatasetBuilder
from src.preprocesses.cfg_corpus import CfgCorpusBuilder
from src.dataset.nmt_inspired_dataset import NMTInsDataEnd
from src.preprocesses.corpus import CorpusBuilder
from src.preprocesses.preprocess import BinBag
from src.utils.auto_json import AutoJson
from src.utils.json_utils import obj_update
from src.utils.logger import get_logger

logger = get_logger('database')


def get_database_client():
    client = MongoClient('localhost', 27017)
    return client


def get_database():
    return get_database_client().test_database


def adjust_dataset_args(args):
    if not args.find_corpus:
        args.find_corpus = args.base_corpus
    obj_update(args.base_corpus, args.find_corpus)
    return args


def prepare_args(db, data_args, model_args, epochs, n_batch, init_lr):
    # dataset args are loaded from file since they are too complex
    # to be passed through command line
    ds = adjust_dataset_args(AutoJson.load(data_args))
    rt = RuntimeArgs(epochs, n_batch, init_lr)
    m = AutoJson.load(model_args)
    args = TrainArgs(ds, rt, m)

    dao = Dao.instance(TrainArgs, db, GridFS(db))
    # fetch from db to insert the primary key `_id` into args
    return dao.find_or_store(to_filter(args), args)


def load_progs_jointly(db, args: BinArgs):
    """
    Joint product the args to form a set of binaries and then load the
    info of these binaries into a list.
    """
    prog_dao = Dao.instance(Program, db, GridFS(db))
    for (prog, prog_ver), (cc, cc_ver), arch, opt, obf in args.joint():
        prog = Program(prog=prog, prog_ver=prog_ver, cc=cc, cc_ver=cc_ver,
                       arch=arch, opt=opt, obf=obf)
        # fixme: update database so that program can be
        ps = prog_dao.find(to_filter(prog, with_cls=False))
        if ps is None:
            raise ValueError(f'No such Program info in the database: \
                prog={prog}, prog_ver={prog_ver}, cc={cc}, cc_ver={cc_ver}, \
                arch={arch}, opt={opt}, obf={obf}')
        for p in ps:
            yield p


def load_vocab(db, args: BinArgs):
    """Load the vocab where the word unit is operands/operators"""
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
        pp.PpFullLabel().as_map(),
        pp.PpMergeBlocks().as_map(),
        pp.PpFilterFunc(minlen=5).as_map(),
        pp.PpTokenizer().as_map(),
        pp.PpOneHotEncoder(vocab).as_map(),
        pp.PpMergeFuncs().as_map(),
        pp.PpCorpus(CorpusBuilder()).as_map()
    ).run().corpus.run()


def load_cfg_corpus(db, args: BinArgs, vocab):
    progs = load_progs_jointly(db, args)
    return rx.just(BinBag(progs)).pipe(
        pp.PpFullLabel().as_map(),
        pp.PpFilterFunc(minlen=5).as_map(),
        # do not merge blocks
        pp.PpTokenizer().as_map(),
        pp.PpOneHotEncoder(vocab).as_map(),
        pp.PpCfgAdj().as_map(),
        pp.PpMergeFuncs().as_map(),
        pp.PpCorpus(CfgCorpusBuilder()).as_map()  # build cfg corpus
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
    bb = BinBag(progs)
    bb = rx.just(bb).pipe(
        pp.PpFullLabel().as_map(),
        pp.PpMergeBlocks().as_map(),
        pp.PpFilterFunc(minlen=minlen).as_map(),
        pp.PpOneHotEncoder(vocab).as_map(),
        pp.PpPadding(maxlen, 0).as_map(),
        pp.PpMergeFuncs().as_map(),
        pp.PpCorpus(CorpusBuilder()).as_map()
    ).run()
    return bb.corpus.run()


def load_word2vec_data(db, vocab_args, corpus_args, window, ss):
    vocab = load_vocab(db, vocab_args)
    corpus = load_corpus(db, corpus_args, vocab)

    builder = Word2VecDatasetBuilder(vocab)
    train_ds = builder.build(corpus, window, ss)
    return vocab, corpus, train_ds


def load_pvdm_data(db, vocab_args, train_args, query_args, window, ss):
    """
    Load cbow-related data which is used for training and evaluation.
    """
    vocab = load_vocab(db, vocab_args)
    train_corpus = load_corpus(db, train_args, vocab)
    query_corpus = load_corpus(db, query_args, vocab)

    # sync two corpus for the conviency of evaluation
    train_corpus, query_corpus = sync_corpus(train_corpus, query_corpus)

    dataset_maker = PVDMDatasetBuilder(vocab)
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


def load_genn_ufe_data(db, vocab_args, args1, args2):
    vocab = load_vocab(db, vocab_args)
    corpus1 = load_cfg_corpus(db, args1, vocab)
    corpus2 = load_cfg_corpus(db, args2, vocab)
    # do not sync corpus!! next line will do that
    ds_builder = GENNDatasetBuilder()
    train_ds = ds_builder.build(corpus1, corpus2)
    return vocab, corpus1, corpus2, train_ds


def load_model(db, require):
    id_, mod = require['training_id'], require['module']
    dao = Dao.instance(CheckPoint, db, GridFS(db))
    cp = dao.find_one({'training_id': ObjectId(id_)})
    if cp and mod in cp.checkpoints:
        return cp.checkpoints[mod]
    raise RuntimeError(f'No such checkpoint found: '
                       f'training_id: {id_},'
                       f'module: {mod}')


def dump_model(db, id_, cp):
    dao = Dao.instance(CheckPoint, db, GridFS(db))
    if not dao.id_one({'training_id': id_}):
        return dao.store_one(CheckPoint(id_, cp))
    raise RuntimeError(f'There already is a checkpoint for this training '
                       f'process: {id_}')


def remove_checkpoint(db, training_id):
    dao = Dao.instance(CheckPoint, db, GridFS(db))
    dao.delete({'training_id': training_id})
