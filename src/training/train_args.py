from typing import List

from src.ida.code_elements import Arch
from src.utils.auto_json import auto_json, AutoJson
from src.utils.json_utils import obj_update
from src.utils.list_joint import flat, joint_list


@auto_json
class BinArgs:
    archs: List[Arch]

    def __init__(self, progs=None, prog_vers=None, ccs=None, cc_vers=None,
                 archs=None, opts=None, obfs=None):
        self.progs, self.prog_vers = progs, prog_vers
        self.ccs, self.cc_vers = ccs, cc_vers
        self.archs, self.opts, self.obfs = archs, opts, obfs

    def joint(self):
        """ Joint product the args to form a set of binaries """
        progs = flat(self.progs, self.prog_vers)
        ccs = flat(self.ccs, self.cc_vers)
        return joint_list([progs, ccs, self.archs, self.opts, self.obfs])


@auto_json
class DatasetArgs:
    vocab: BinArgs
    base_corpus: BinArgs
    find_corpus: BinArgs

    def __init__(self, vocab=None, base_corpus=None, find_corpus=None):
        self.vocab = vocab
        self.base_corpus = base_corpus
        self.find_corpus = find_corpus


@auto_json
class RuntimeArgs:

    def __init__(self, epochs=None, n_batch=None, init_lr=None):
        self.epochs = epochs
        self.n_batch = n_batch
        self.init_lr = init_lr


@auto_json
class TrainArgs:
    """
    Used to identify model and training results, such as loss and metrics
    """
    rt: RuntimeArgs

    def __init__(self, ds_args=None, rt_args=None, m_args=None, tag='default'):
        self.ds = ds_args
        self.rt: RuntimeArgs = rt_args
        self.m = m_args
        self.tag = tag


def wrap_dataset_args(args):
    if not args.find_corpus:
        args.find_corpus = args.base_corpus
    obj_update(args.base_corpus, args.find_corpus)
    return args


def prepare_args(data_args, model_args, epochs, n_batch, init_lr):
    # dataset args are loaded from file since they are too complex
    # to be passed through command line
    ds_args = AutoJson.load(data_args)
    m_args = AutoJson.load(model_args)
    rt_args = RuntimeArgs(epochs, n_batch, init_lr)
    return TrainArgs(wrap_dataset_args(ds_args), rt_args, m_args)
