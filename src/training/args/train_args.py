import json
from typing import List
import hashlib

from src.ida.code_elements import Arch
from src.utils.auto_json import auto_json, AutoJson
from src.utils.collection_op import joint_list, flat


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

    def __str__(self):
        return str(self.__dict__)


@auto_json
class DatasetArgs:
    vocab: BinArgs
    base_corpus: BinArgs
    find_corpus: BinArgs

    def __init__(self, vocab=None, base_corpus=None, find_corpus=None):
        self.vocab = vocab
        self.base_corpus = base_corpus
        self.find_corpus = find_corpus

    def __str__(self):
        return str(self.__dict__)


@auto_json
class RuntimeArgs:

    def __init__(self, epochs=None, n_batch=None, init_lr=None):
        self.epochs = epochs
        self.n_batch = n_batch
        self.init_lr = init_lr

    def __str__(self):
        return str(self.__dict__)


@auto_json
class ModelArgs:

    def __init__(self, args=None):
        self.args = args
        self.models = dict()
        self.models_to_load = args['models'] if args else []
        self.models_to_save = args['models_to_save'] if args else []
        self.clean_args()

    def clean_args(self):
        args = dict()
        for key in self.models_to_load:
            args[key] = self.args[key]
        self.args = args


@auto_json
class TrainArgs:
    """
    Used to identify model and training results, such as loss and metrics
    """
    rt: RuntimeArgs

    def __init__(self, ds_args=None, rt_args=None, m_args=None, tag='default'):
        self.ds = ds_args
        self.rt = rt_args
        self.m = m_args
        self.tag = tag
        self.hash_id = hash_args(self.__dict__)

    def __str__(self):
        return str(self.__dict__)


def hash_args(obj):
    dic = AutoJson.to_dict(obj)
    dumps = json.dumps(dic, sort_keys=True)
    return hashlib.md5(dumps.encode()).hexdigest()
